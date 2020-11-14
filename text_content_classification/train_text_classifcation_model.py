import argparse
import os
import pandas as pd
import numpy as np
import re
import torch
import datetime
import time
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import json


def cleanText(input_text):
    cleaned_text = input_text.lower()
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', cleaned_text)
    cleaned_text = re.sub(r"\b[a-zA-Z]\b", ' ', cleaned_text)
    cleaned_text = re.sub(' +', ' ', cleaned_text)
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text


class TopicDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_seq_length):
        self.len = data_frame.shape[0]
        self.data = data_frame
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __getitem__(self, index):
        text = str(self.data.iloc[index]["text"])
        encoded_vals = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True
        )

        ids = encoded_vals["input_ids"]
        mask = encoded_vals["attention_mask"]

        return {
            "ids": torch.tensor(ids),
            "mask": torch.tensor(mask),
            "targets": torch.tensor(self.data.iloc[index]["topic_label"])
        }

    def __len__(self):
        return self.len


def format_time(elapsed_seconds):
    elapsed_rounded = int(round((elapsed_seconds)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def computeMetrics(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    f1_val = f1_score(y_true, y_pred, average="micro")
    return accuracy, f1_val

def saveModel(model, model_suffix, args):
    print("Saving model = ", "model" + str(model_suffix))
    # Save the model:
    model.save_pretrained(os.path.join(args.output_dir, "model" + str(model_suffix)))

    # Save the model configuration:
    with open(os.path.join(args.output_dir, "model" + str(model_suffix) + "_params.json"), "w") as f:
        json.dump(vars(args), f)

def train(model, train_data_loader, val_data_loader, args):
    # Get the device info:
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, eps=1e-8)

    total_training_steps = len(train_data_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_training_steps)

    training_logs = []
    inital_time = time.time()
    minimum_val_loss = None
    best_model_state_dict = None
    best_model_epoch = None
    for cur_epoch in tqdm(range(args.num_epochs)):
        print("\n\nEpoch number = ", cur_epoch, "\n")

        print("\n\nTraining...")
        start_time = time.time()
        total_train_loss = 0

        model.train()
        for cur_step, cur_batch in enumerate(train_data_loader):
            cur_input_ids = cur_batch["ids"].to(device)
            cur_attention_masks = cur_batch["mask"].to(device)
            cur_labels = cur_batch["targets"].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids=cur_input_ids,
                            attention_mask=cur_attention_masks,
                            labels=cur_labels)
            total_train_loss += loss.item()

            # Backprop the loss
            loss.backward()

            # Perform Gradient Clipping to 1.0:
            clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters:
            optimizer.step()

            # Update learning rate:
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_data_loader)
        total_train_time = format_time(time.time() - start_time)
        print("Training time for Epoch ", cur_epoch, " = ", total_train_time)
        print("Average training loss = ", avg_train_loss)


        print("\n\nValidation...")
        start_time = time.time()
        total_val_loss = 0
        y_preds_total = []
        y_labels_total = []

        # Setting the model in eval mode:
        model.eval()
        for cur_step, cur_batch in tqdm(enumerate(val_data_loader)):
            cur_input_ids = cur_batch["ids"].to(device)
            cur_attention_masks = cur_batch["mask"].to(device)
            cur_labels = cur_batch["targets"].to(device)

            with torch.no_grad():
                cur_loss, cur_logits = model(cur_input_ids,
                                            attention_mask=cur_attention_masks,
                                            labels=cur_labels)

                total_val_loss += cur_loss.item()
                cur_logits_cpu = cur_logits.detach().cpu().numpy()
                cur_labels_cpu = cur_labels.to("cpu").numpy()

                y_preds_flat = list(np.argmax(cur_logits_cpu, axis=1).flatten())
                y_labels_flat = list(cur_labels_cpu.flatten())

                y_preds_total += y_preds_flat
                y_labels_total += y_labels_flat

        avg_val_loss = total_val_loss / len(val_data_loader)
        val_accuracy, val_f1_score = computeMetrics(y_preds_total, y_labels_total)
        total_val_time = format_time(time.time() - start_time)
        print("Validation time for Epoch ", cur_epoch, " = ", total_val_time)
        print("Average validation loss = ", avg_val_loss)
        print("Validation accuracy = ", val_accuracy)
        print("Validation F1 score (micro) = ", val_f1_score)

        # Saving the model with minimum validation loss (best model till the current epoch)
        if minimum_val_loss is None:
            minimum_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()
            best_model_epoch = cur_epoch
            saveModel(model, "_best", args)
        elif avg_val_loss < minimum_val_loss:
            minimum_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()
            best_model_epoch = cur_epoch
            saveModel(model, "_best", args)

        training_logs.append(
        {
                "epoch": cur_epoch,
                "avg_train_loss": avg_train_loss,
                "train_time": total_train_time,
                "avg_val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1_score,
                "val_time": total_val_time,
        })

    print("Training done...")
    total_time = format_time(time.time() - inital_time)
    print("Total time taken = ", total_time)

    # Save the training logs:
    with open(os.path.join(args.output_dir, "training_logs.json"), "w") as f:
        json.dump(training_logs, f)

    # # Save the model:
    # print("Best model epoch = ", best_model_epoch)
    # model.load_state_dict(best_model_state_dict)
    # saveModel(model, "_best", args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="../data/topic_classification_data",
                        help='Path for Data files')
    parser.add_argument('--output_dir', type=str, default="../outputs/distil_bert_ckpts_v2",
                        help='Path to save the checkpoints')
    parser.add_argument('--model_name_or_path', type=str, default="distilbert-base-uncased",
                        help='Model name or Path')
    parser.add_argument('--tokenizer_name_or_path', type=str, default="distilbert-base-uncased",
                        help='Tokenizer name or Path')
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--num_output_classes', type=int, default=11)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)


    args = parser.parse_known_args()[0]
    print(args)

    # Get the device info:
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    # Read the input data:
    df_train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    df_test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    # Input data processing:
    df_train["text"] = df_train.apply(lambda x: cleanText(x["text"]), axis=1)
    df_val["text"] = df_val.apply(lambda x: cleanText(x["text"]), axis=1)
    df_test["text"] = df_test.apply(lambda x: cleanText(x["text"]), axis=1)

    tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer_name_or_path)

    df_train_set = TopicDataset(df_train, tokenizer, args.max_seq_length)
    train_data_loader = DataLoader(df_train_set, batch_size=args.train_batch_size, shuffle=True)

    df_val_set = TopicDataset(df_val, tokenizer, args.max_seq_length)
    val_data_loader = DataLoader(df_val_set, batch_size=args.eval_batch_size, shuffle=True)

    df_test_set = TopicDataset(df_test, tokenizer, args.max_seq_length)
    test_data_loader = DataLoader(df_test_set, batch_size=args.eval_batch_size, shuffle=True)

    # Model training:
    model = DistilBertForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                num_labels=args.num_output_classes,
                                                                output_attentions=False,
                                                                output_hidden_states=False)
    model.to(device)
    train(model, train_data_loader, val_data_loader, args)

