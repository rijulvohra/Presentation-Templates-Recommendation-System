# Required imports
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn

import argparse
from tqdm import tqdm
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if torch.cuda.device_count() > 0:
    device = "cuda"
else:
    device = "cpu"

def computeMetrics(y_true, y_pred):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="micro")
    metrics["recall"] = recall_score(y_true, y_pred, average="micro")
    metrics["f1_score"] = f1_score(y_true, y_pred, average="micro")
    return metrics


def train(model, train_dl, val_dl, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_criterion = nn.NLLLoss()
    training_logs = []

    best_val_loss = None
    model.to(device)
    for cur_epoch in tqdm(range(args.num_epochs)):

        # Train
        avg_train_loss = 0.0
        num_train_steps = 0
        model.train()
        for idx, cur_batch in enumerate(tqdm(train_dl)):
            cur_images = cur_batch[0].to(device)
            cur_labels = cur_batch[1].to(device)

            optimizer.zero_grad()
            output = model(cur_images)
            loss = loss_criterion(output, cur_labels)
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
            cur_images = cur_batch[0].to(device)
            cur_labels = cur_batch[1].to(device)

            with torch.no_grad():
                cur_logits = model(cur_images)

            cur_logits_cpu = cur_logits.detach().cpu().numpy()
            y_pred_flat = list(np.argmax(cur_logits_cpu, axis=1).flatten())
            y_pred_all += y_pred_flat
            y_true_all += list(cur_labels.cpu().numpy())
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
        "train_batch_size": 16,
        "val_batch_size": 16,
        "input_image_size": 224,
        "num_classes": 11,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "output_dir": "../models"
    }
    args = argparse.Namespace(**args_dict)

    # Data processing:
    base_dir = "../templates_data"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "dev")

    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=args.input_image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=args.input_image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = datasets.ImageFolder(train_dir, transform=image_transforms["train"])
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataset = datasets.ImageFolder(val_dir, transform=image_transforms["val"])
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    # Modeling:
    model = models.resnet50(pretrained=True)
    fc_in_features = model.fc.in_features
    model.fc = nn.Linear(fc_in_features, args.num_classes)
    train(model, train_dataloader, val_dataloader, args)
