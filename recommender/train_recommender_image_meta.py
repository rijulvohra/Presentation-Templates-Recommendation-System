# Required imports
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from PIL import Image

import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import json
import glob
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class RecommenderDataset(Dataset):
    def __init__(self, args, type_path="val"):
        self.inputs = []
        self.targets = []
        self.image_meta_feature_map = {}
        self.args = args
        self.type_path = type_path

        self._load_feature_extractor_models()
        self._build()

        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def __getitem__(self, index):
        return {
            "inputs": self.inputs[index],
            "targets": self.targets[index]
        }

    def __len__(self):
        return len(self.inputs)

    def _load_feature_extractor_models(self):
        tc_model_path = os.path.join(args.model_dir, "distilbert_tc_model")
        self.text_feat_extractor = DistilBertForSequenceClassification.from_pretrained(tc_model_path)
        self.text_feat_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # Remove the classifier layer and dropout layer
        self.text_feat_extractor.classifier = nn.Identity()
        self.text_feat_extractor.dropout = nn.Identity()

        print("LOG: Loading models done...")

    def _getTextualFeatures(self, input_text):
        encoded_input = self.text_feat_tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.args.text_max_seq_length,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        ids = encoded_input["input_ids"]
        mask = encoded_input["attention_mask"]
        text_features = self.text_feat_extractor(input_ids=ids,
                                                 attention_mask=mask)[0].squeeze().detach().cpu()

        # free up memory
        del ids
        del mask
        del encoded_input

        return text_features

    def _getImageMetaFeatures(self, input_image_path, input_image_meta_text):
        if self.image_meta_feature_map.get(input_image_path) is not None:
            image_meta_features = self.image_meta_feature_map[input_image_path]
        else:
            encoded_input = self.text_feat_tokenizer.encode_plus(
                input_image_meta_text,
                add_special_tokens=True,
                max_length=self.args.text_max_seq_length,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            ids = encoded_input["input_ids"]
            mask = encoded_input["attention_mask"]
            image_meta_features = self.text_feat_extractor(input_ids=ids,
                                                           attention_mask=mask)[0].squeeze().detach().cpu()
            self.image_meta_feature_map[input_image_path] = image_meta_features

            # free up memory
            del ids
            del mask
            del encoded_input

        return image_meta_features

    def _build(self):

        if self.type_path == "train":
            data_path = os.path.join(self.args.data_dir, "train.csv")
        elif self.type_path == "val":
            data_path = os.path.join(self.args.data_dir, "val.csv")
        else:
            raise ValueError("Invalid value for type path!")

        df_data = pd.read_csv(data_path)
        for idx, cur_row in tqdm(df_data.iterrows(), total=df_data.shape[0], desc="LOG: Loading data"):
            try:
                cur_text = cur_row["text"]
                cur_image_path = os.path.join(self.args.image_data_dir, cur_row["topic"], cur_row["image_label"])
                cur_image_meta_text = cur_row["description"]
                cur_text_features = self._getTextualFeatures(cur_text).squeeze()
                cur_image_meta_features = self._getImageMetaFeatures(cur_image_path, cur_image_meta_text).squeeze()
                cur_inputs = torch.cat((cur_text_features, cur_image_meta_features))
                cur_targets = int(cur_row["match"])

                self.inputs.append(cur_inputs)
                self.targets.append(cur_targets)
            except Exception as e:
                print(e)


class RecommenderModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=None):
        super(RecommenderModel, self).__init__()
        # 1 layer model
        if hidden_size is None:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, num_classes)
            )
        else:
            # 2 layer model
            self.classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_classes)
            )

    def forward(self, input):
        output = self.classifier(input)
        return output


def computeMetrics(y_true, y_pred):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="micro")
    metrics["recall"] = recall_score(y_true, y_pred, average="micro")
    metrics["f1_score"] = f1_score(y_true, y_pred, average="micro")
    return metrics


def train(model, train_dl, val_dl, args):
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_criterion = nn.CrossEntropyLoss()
    training_logs = []

    best_val_loss = None
    model.to(device)
    for cur_epoch in tqdm(range(args.num_epochs)):

        # Train
        avg_train_loss = 0.0
        num_train_steps = 0
        model.train()
        for idx, cur_batch in enumerate(tqdm(train_dl)):
            cur_inputs = cur_batch["inputs"].to(device)
            cur_targets = cur_batch["targets"].to(device)

            optimizer.zero_grad()
            cur_logits = model(cur_inputs)
            loss = loss_criterion(cur_logits, cur_targets)
            avg_train_loss += loss.item()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            num_train_steps += 1

        avg_train_loss /= num_train_steps

        # Eval
        avg_val_loss = 0.0
        num_val_steps = 0
        model.eval()
        y_pred_all = []
        y_true_all = []
        for idx, cur_batch in enumerate(tqdm(val_dl)):
            cur_inputs = cur_batch["inputs"].to(device)
            cur_targets = cur_batch["targets"].to(device)

            with torch.no_grad():
                cur_logits = model(cur_inputs)

            loss = loss_criterion(cur_logits, cur_targets)
            avg_val_loss += loss.item()

            cur_logits_cpu = cur_logits.detach().cpu().numpy()
            y_pred_flat = list(np.argmax(cur_logits_cpu, axis=1).flatten())
            y_pred_all += y_pred_flat
            y_true_all += list(cur_targets.cpu().numpy())
            num_val_steps += 1

        avg_val_loss /= num_val_steps
        cur_metrics = computeMetrics(y_true_all, y_pred_all)
        print("Metrics: ", cur_metrics)

        # Store the best checkpoint
        if best_val_loss is None:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(args.output_dir,
                                           "best_model_epoch={}_val_loss={}".format(cur_epoch, round(avg_val_loss, 2)))
            print("Saving model at {}".format(model_save_path))
            torch.save(model.state_dict(), model_save_path)

        elif avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(args.output_dir,
                                           "best_model_epoch={}_val_loss={}".format(cur_epoch, round(avg_val_loss, 2)))
            print("Saving model at {}".format(model_save_path))
            torch.save(model.state_dict(), model_save_path)

        # Store the training log
        cur_logs = {}
        cur_logs["epoch_num"] = cur_epoch
        cur_logs["avg_train_loss"] = avg_train_loss
        cur_logs["avg_val_loss"] = avg_val_loss
        cur_logs["metrics"] = cur_metrics
        training_logs.append(cur_logs)

    # Write the logs
    log_path = os.path.join(args.output_dir, "training_logs.json")
    with open(log_path, "w") as f:
        json.dump(training_logs, f)


if __name__ == "__main__":
    # Hyperparams:
    args_dict = {
        "train_batch_size": 8,
        "val_batch_size": 8,
        "input_image_size": 224,
        "num_classes": 2,
        "num_epochs": 10,
        "learning_rate": 0.01,
        "text_max_seq_length": 256,
        "output_dir": "models/recommender_model",
        "model_dir": "models",
        "image_data_dir": "templates_data/full",
        "data_dir": "recommender_data_v2"
    }
    args = argparse.Namespace(**args_dict)

    train_dataset = RecommenderDataset(args, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    val_dataset = RecommenderDataset(args, "val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    recommender = RecommenderModel(input_size=1536, num_classes=2, hidden_size=500)
    train(recommender, train_dataloader, val_dataloader, args)

