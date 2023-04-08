import os
import sys
import json

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from loaders import data_loader
from architectures import cnn_model
from deep_training import ModelTraining

from utils.enums import Tasks
from utils.augmentation import between_class, mixup_data, mixup_criterion
from utils.utils import write_results, create_results_folder, get_file_path
from utils.metrics import accuracy_mixup, accuracy_score, f1_score


def clf_label(opt):
    if opt.dataset == "leaf":
        if opt.model_task == Tasks.MULTITASK:
            return "multitask"
        elif opt.model_task == Tasks.BIOTIC_STRESS:
            return "biotic_stress"
        else:
            return "severity"
    else:
        return "symptom"


class MultiTaskClassifier(ModelTraining):
    """Multi Task Classifier."""

    def __init__(self, options, images_dir: str):
        self.opt = options
        self.opt.num_classes = (5, 5)
        self.opt.images_dir = images_dir
        create_results_folder(self.opt.results_path, self.opt.experiment_name)

    def train(self, train_loader, model, criterion, optimizer, data_augmentation=None):
        # tell to pytorch that we are training the model
        model.train()

        metrics = {"loss": 0.0, "biotic_stress_acc": 0.0, "severity_acc": 0.0}
        total = 0

        pbar = tqdm(train_loader)
        for images, labels_dis, labels_sev in pbar:
            # Loading images on gpu
            if torch.cuda.is_available():
                images, labels_dis, labels_sev = (
                    images.cuda(),
                    labels_dis.cuda(),
                    labels_sev.cuda(),
                )

            # Apply data augmentation
            if data_augmentation == "bc+":
                images, labels_dis_a, labels_sev_a = between_class(images, labels_dis, labels_sev)
                labels_dis, labels_sev = (
                    torch.max(labels_dis_a, 1)[1],
                    torch.max(labels_sev_a, 1)[1],
                )

            elif data_augmentation == "mixup":
                images, labels_dis_a, labels_dis_b, labels_sev_a, labels_sev_b, lam = mixup_data(
                    images, labels_dis, labels_sev
                )

            # Pass images through the network
            outputs_dis, outputs_sev = model(images)

            # Compute error
            if data_augmentation == "bc+":
                loss_dis = criterion(torch.softmax(outputs_dis, dim=1).log(), labels_dis_a)
                loss_sev = criterion(torch.softmax(outputs_sev, dim=1).log(), labels_sev_a)

            elif data_augmentation == "mixup":
                loss_dis = mixup_criterion(criterion, outputs_dis, labels_dis_a, labels_dis_b, lam)
                loss_sev = mixup_criterion(criterion, outputs_sev, labels_sev_a, labels_sev_b, lam)

            else:
                loss_dis = criterion(outputs_dis, labels_dis)
                loss_sev = criterion(outputs_sev, labels_sev)

            # Clear gradients parameters
            model.zero_grad()

            # Getting gradients

            (loss_dis + loss_sev).backward()

            # Clipping gradient
            clip_grad_norm_(model.parameters(), 5)

            # Updating parameters
            optimizer.step()

            # Compute metrics
            # Loss
            metrics["loss"] += (loss_dis + loss_sev).data.cpu() / 2 * len(images)

            # Biotic stress metrics
            pred = torch.max(outputs_dis.data, 1)[1]
            y_pred = pred.cpu().int()

            if data_augmentation == "mixup":
                metrics["biotic_stress_acc"] += accuracy_mixup(
                    y_pred,
                    labels_dis_a.cpu().int(),
                    labels_dis_b.cpu().int(),
                    lam,
                ) * len(images)
            else:
                y_true = labels_dis.cpu().int()
                metrics["biotic_stress_acc"] += accuracy_score(y_true, y_pred) * len(images)

            # Severity metrics
            pred = torch.max(outputs_sev.data, 1)[1]
            y_pred = pred.cpu().int()

            if data_augmentation == "mixup":
                metrics["severity_acc"] += accuracy_mixup(
                    y_pred,
                    labels_sev_a.cpu().int(),
                    labels_sev_b.cpu().int(),
                    lam,
                ) * len(images)
            else:
                y_true = labels_sev.cpu().int()
                metrics["severity_acc"] += accuracy_score(y_true, y_pred) * len(images)

            # Update progress bar
            total += len(images)
            dis_acc = 100.0 * metrics["biotic_stress_acc"] / total
            sev_acc = 100.0 * metrics["severity_acc"] / total
            pbar.set_description("[Dis ACC: %.2f, Sev ACC: %.2f]" % (dis_acc, sev_acc))

        for x in metrics:
            if x != "loss":
                metrics[x] = 100.0 * metrics[x] / len(train_loader.dataset)
            else:
                metrics[x] = metrics[x] / len(train_loader.dataset)
            metrics[x] = float(metrics[x])

        return metrics

    def validation(self, val_loader, model, criterion):
        # tell to pytorch that we are evaluating the model
        model.eval()

        metrics = {"loss": 0.0, "biotic_stress_acc": 0.0, "severity_acc": 0.0, "mean_fs": 0.0}
        total = 0
        with torch.no_grad():
            pbar = tqdm(val_loader)
            for images, labels_dis, labels_sev in pbar:
                # Loading images on gpu
                if torch.cuda.is_available():
                    images, labels_dis, labels_sev = (
                        images.cuda(),
                        labels_dis.cuda(),
                        labels_sev.cuda(),
                    )

                # pass images through the network
                outputs_dis, outputs_sev = model(images)

                # calculate loss
                loss_dis = criterion(outputs_dis, labels_dis)
                loss_sev = criterion(outputs_sev, labels_sev)

                # Compute metrics
                ## Loss
                metrics["loss"] += (loss_dis + loss_sev).data.cpu() / 2 * len(images)

                # Biotic stress metrics
                pred = torch.max(outputs_dis.data, 1)[1]
                y_pred = pred.cpu().int()
                y_true = labels_dis.cpu().int()

                metrics["biotic_stress_acc"] += accuracy_score(y_true, y_pred) * len(images)
                metrics["mean_fs"] += f1_score(y_true, y_pred, average="macro") * len(images) * 0.5

                # Severity metrics
                pred = torch.max(outputs_sev.data, 1)[1]
                y_pred = pred.cpu().int()
                y_true = labels_sev.cpu().int()

                metrics["severity_acc"] += accuracy_score(y_true, y_pred) * len(images)
                metrics["mean_fs"] += f1_score(y_true, y_pred, average="macro") * len(images) * 0.5

                # Update progress bar
                total += len(images)
                dis_acc = 100.0 * metrics["biotic_stress_acc"] / total
                sev_acc = 100.0 * metrics["severity_acc"] / total
                pbar.set_description("[Dis ACC: %.2f, Sev ACC: %.2f]" % (dis_acc, sev_acc))

        for x in metrics:
            if x != "loss":
                metrics[x] = 100.0 * metrics[x] / len(val_loader.dataset)
            else:
                metrics[x] = metrics[x] / len(val_loader.dataset)
            metrics[x] = float(metrics[x])

        return metrics

    def print_info(self, **kwargs):
        data_type = kwargs.get("data_type")
        metrics = kwargs.get("metrics")
        epoch = kwargs.get("epoch")
        epochs = kwargs.get("epochs")

        print(
            "[Epoch:%3d/%3d][%s][LOSS: %4.2f][Dis ACC: %5.2f][Sev ACC: %5.2f]"
            % (
                epoch + 1,
                epochs,
                data_type,
                metrics["loss"],
                metrics["biotic_stress_acc"],
                metrics["severity_acc"],
            )
        )

    def run_training(self):
        print(clf_label(self.opt) + " - training: " + self.opt.experiment_name)

        # Dataset
        train_loader, val_loader, _ = data_loader(self.opt)

        # Model
        model = cnn_model(self.opt.model, self.opt.pretrained, self.opt.num_classes)

        # Criterion
        criterion_train = (
            nn.CrossEntropyLoss()
            if self.opt.data_augmentation != "bc+"
            else torch.nn.KLDivLoss(reduction="batchmean")
        )
        criterion_val = nn.CrossEntropyLoss()

        # Optimizer
        if self.opt.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.9, weight_decay=self.opt.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, weight_decay=self.opt.weight_decay
            )

        record = {}
        record["model"] = self.opt.model
        record["batch_size"] = self.opt.batch_size
        record["weight_decay"] = self.opt.weight_decay
        record["optimizer"] = self.opt.optimizer
        record["pretrained"] = self.opt.pretrained
        record["data_augmentation"] = self.opt.data_augmentation
        record["epochs"] = self.opt.epochs
        record["train_loss"] = []
        record["val_loss"] = []
        record["train_biotic_stress_acc"] = []
        record["val_biotic_stress_acc"] = []
        record["train_severity_acc"] = []
        record["val_severity_acc"] = []

        best_fs = 0.0

        for epoch in range(self.opt.epochs):
            # Training
            train_metrics = self.train(
                train_loader, model, criterion_train, optimizer, self.opt.data_augmentation
            )
            self.print_info(
                data_type="TRAIN", metrics=train_metrics, epoch=epoch, epochs=self.opt.epochs
            )

            # Validation
            val_metrics = self.validation(val_loader, model, criterion_val)
            self.print_info(
                data_type="VAL", metrics=val_metrics, epoch=epoch, epochs=self.opt.epochs
            )

            # Adjust learning rate
            optimizer = self.adjust_learning_rate(optimizer, epoch, self.opt)

            # Recording metrics
            record["train_loss"].append(train_metrics["loss"])
            record["train_biotic_stress_acc"].append(train_metrics["biotic_stress_acc"])
            record["train_severity_acc"].append(train_metrics["severity_acc"])

            record["val_loss"].append(val_metrics["loss"])
            record["val_biotic_stress_acc"].append(val_metrics["biotic_stress_acc"])
            record["val_severity_acc"].append(val_metrics["severity_acc"])

            # Record best model
            curr_fs = val_metrics["mean_fs"]
            if curr_fs > best_fs:
                best_fs = curr_fs

                # Saving model
                torch.save(model.state_dict(), get_file_path(self.opt, "net_weights.pth"))
                print("model saved")

            # Saving log
            with open(get_file_path(self.opt, "logs.json"), "w") as fp:
                json.dump(record, fp, indent=4, sort_keys=True)

    def run_test(self):
        # Dataset
        _, _, test_loader = data_loader(self.opt)

        # Loading model
        weights_path = get_file_path(self.opt, "net_weights.pth")
        model = cnn_model(self.opt.model, self.opt.pretrained, self.opt.num_classes, weights_path)

        # tell to pytorch that we are evaluating the model
        model.eval()

        y_pred_dis = np.empty(0)
        y_true_dis = np.empty(0)

        y_pred_sev = np.empty(0)
        y_true_sev = np.empty(0)

        with torch.no_grad():
            for images, labels_dis, labels_sev in test_loader:
                # Loading images on gpu
                if torch.cuda.is_available():
                    images, labels_dis, labels_sev = (
                        images.cuda(),
                        labels_dis.cuda(),
                        labels_sev.cuda(),
                    )

                # pass images through the network
                outputs_dis, outputs_sev = model(images)

                #### Compute metrics

                # Biotic stress
                pred = torch.max(outputs_dis.data, 1)[1]

                y_pred_dis = np.concatenate((y_pred_dis, pred.data.cpu().numpy()))
                y_true_dis = np.concatenate((y_true_dis, labels_dis.data.cpu().numpy()))

                # Severity
                pred = torch.max(outputs_sev.data, 1)[1]

                y_pred_sev = np.concatenate((y_pred_sev, pred.data.cpu().numpy()))
                y_true_sev = np.concatenate((y_true_sev, labels_sev.data.cpu().numpy()))

        # Biotic stress
        write_results(
            y_true=y_true_dis,
            y_pred=y_pred_dis,
            cm_target_names=["Healthy", "Leaf miner", "Rust", "Phoma", "Cercospora"],
            results_path=self.opt.results_path,
            task_name="biotic_stress",
            experiment_name=self.opt.experiment_name,
        )

        # Severity
        write_results(
            y_true=y_true_sev,
            y_pred=y_pred_sev,
            cm_target_names=["Healthy", "Very low", "Low", "High", "Very high"],
            results_path=self.opt.results_path,
            task_name="severity",
            experiment_name=self.opt.experiment_name,
        )

        return y_true_dis, y_pred_dis, y_true_sev, y_pred_sev

    def get_n_params(self):
        weights_path = get_file_path(self.opt, "net_weights.pth")
        model = cnn_model(self.opt.model, self.opt.pretrained, (5, 5), weights_path)
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


class SingleTaskClassifier(ModelTraining):
    """Single task classifier."""

    def __init__(self, options, images_dir):
        self.opt = options
        self.opt.num_classes = 5
        self.opt.images_dir = images_dir
        create_results_folder(self.opt.results_path, self.opt.experiment_name)

    def train(self, train_loader, model, criterion, optimizer, data_augmentation=None):
        # tell to pytorch that we are training the model
        model.train()

        metrics = {"loss": 0.0, "acc": 0.0}
        correct = 0
        total = 0

        pbar = tqdm(train_loader)
        for images, labels in pbar:
            # Loading images on gpu
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            if data_augmentation == "bc+":
                images, labels_a, _ = between_class(images, labels)
                labels = torch.max(labels_a, 1)[1]

            elif data_augmentation == "mixup":
                images, labels_a, labels_b, lam = mixup_data(images, labels)

            # Clear gradients parameters
            model.zero_grad()

            # pass images through the network
            outputs = model(images)

            if data_augmentation == "bc+":
                loss = criterion(torch.softmax(outputs, dim=1).log(), labels_a)

            elif data_augmentation == "mixup":
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            else:
                loss = criterion(outputs, labels)

            # Getting gradients
            loss.backward()

            # Clipping gradient
            clip_grad_norm_(model.parameters(), 5)

            # Updating parameters
            optimizer.step()

            # Compute metrics
            ## Loss
            metrics["loss"] += loss.data.cpu() * len(images)

            ## Accuracy
            pred = torch.max(outputs.data, 1)[1]
            y_pred = pred.cpu().int()

            if data_augmentation == "mixup":
                correct += accuracy_mixup(
                    y_pred, labels_a.cpu().int(), labels_b.cpu().int(), lam
                ) * len(images)
            else:
                correct += accuracy_score(labels.cpu().int(), y_pred) * len(images)

            total += labels.size(0)
            metrics["acc"] = 100.0 * float(correct) / total

            # Update progress bar
            pbar.set_description("[ACC: %.2f]" % metrics["acc"])

        metrics["loss"] = float(metrics["loss"] / len(train_loader.dataset))

        return metrics

    def validation(self, val_loader, model, criterion):
        # tell to pytorch that we are evaluating the model
        model.eval()

        metrics = {"loss": 0.0, "acc": 0.0, "fs": 0.0}
        correct_acc = 0
        correct_fs = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader)
            for images, labels in pbar:
                # Loading images on gpu
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()

                # pass images through the network
                outputs = model(images)

                # calculate loss
                loss = criterion(outputs, labels)

                # Compute metrics
                ## Loss
                metrics["loss"] += loss.data.cpu() * len(images)

                ## Accuracy
                pred = torch.max(outputs.data, 1)[1]
                y_pred = pred.cpu().int()
                y_true = labels.cpu().int()

                correct_acc += accuracy_score(y_true, y_pred) * len(images)
                correct_fs += f1_score(y_true, y_pred, average="macro") * len(images)

                total += labels.size(0)

                metrics["acc"] = 100.0 * float(correct_acc) / total
                metrics["fs"] = 100.0 * float(correct_fs) / total

                # Update progress bar
                pbar.set_description("[ACC: %.2f]" % metrics["acc"])

        metrics["loss"] = float(metrics["loss"] / len(val_loader.dataset))

        return metrics

    def print_info(self, **kwargs):
        data_type = kwargs.get("data_type")
        metrics = kwargs.get("metrics")
        epoch = kwargs.get("epoch")
        epochs = kwargs.get("epochs")

        print(
            "\r[Epoch:%3d/%3d][%s][LOSS: %4.2f][ACC: %5.2f]"
            % (epoch + 1, epochs, data_type, metrics["loss"], metrics["acc"])
        )

    def run_training(self):
        print(clf_label(self.opt) + " - training: " + self.opt.experiment_name)

        # Data
        train_loader, val_loader, _ = data_loader(self.opt)

        # Model
        model = cnn_model(self.opt.model, self.opt.pretrained, self.opt.num_classes)

        # Criterion
        criterion_train = (
            nn.CrossEntropyLoss()
            if self.opt.data_augmentation != "bc+"
            else torch.nn.KLDivLoss(reduction="batchmean")
        )
        criterion_val = nn.CrossEntropyLoss()

        # Optimizer
        if self.opt.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.9, weight_decay=self.opt.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, weight_decay=self.opt.weight_decay
            )

        record = {}
        record["model"] = self.opt.model
        record["batch_size"] = self.opt.batch_size
        record["weight_decay"] = self.opt.weight_decay
        record["optimizer"] = self.opt.optimizer
        record["pretrained"] = self.opt.pretrained
        record["data_augmentation"] = self.opt.data_augmentation
        record["epochs"] = self.opt.epochs
        record["train_loss"] = []
        record["val_loss"] = []
        record["train_acc"] = []
        record["val_acc"] = []

        best_fs = 0.0

        for epoch in range(self.opt.epochs):
            # Training
            train_metrics = self.train(
                train_loader, model, criterion_train, optimizer, self.opt.data_augmentation
            )
            self.print_info(
                data_type="TRAIN", metrics=train_metrics, epoch=epoch, epochs=self.opt.epochs
            )

            # Validation
            val_metrics = self.validation(val_loader, model, criterion_val)
            self.print_info(
                data_type="VAL", metrics=val_metrics, epoch=epoch, epochs=self.opt.epochs
            )

            # Adjust learning rate
            optimizer = self.adjust_learning_rate(optimizer, epoch, self.opt)

            # Recording metrics
            record["train_loss"].append(train_metrics["loss"])
            record["train_acc"].append(train_metrics["acc"])

            record["val_loss"].append(val_metrics["loss"])
            record["val_acc"].append(val_metrics["acc"])

            # Record best model
            curr_fs = val_metrics["fs"]
            if (curr_fs > best_fs) and epoch >= 3:
                best_fs = curr_fs

                # Saving model
                torch.save(
                    model.state_dict(),
                    get_file_path(self.opt, "net_weights.pth"),
                )
                print("model saved")

            # Saving log
            with open(get_file_path(self.opt, "logs.json"), "w") as fp:
                json.dump(record, fp, indent=4, sort_keys=True)

    def run_test(self):
        # Dataset
        _, _, test_loader = data_loader(self.opt)

        # Loading model
        weights_path = get_file_path(self.opt, "net_weights.pth")
        model = cnn_model(self.opt.model, self.opt.pretrained, self.opt.num_classes, weights_path)

        # tell to pytorch that we are evaluating the model
        model.eval()

        y_pred = np.empty(0)
        y_true = np.empty(0)

        with torch.no_grad():
            for images, labels in test_loader:
                # Loading images on gpu
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()

                # pass images through the network
                outputs = model(images)

                # Compute metrics
                pred = torch.max(outputs.data, 1)[1]
                y_pred = np.concatenate((y_pred, pred.data.cpu().numpy()))
                y_true = np.concatenate((y_true, labels.data.cpu().numpy()))

        # Biotic stress labels
        if self.opt.model_task == Tasks.SEVERITY:
            labels = ["Healthy", "Very low", "Low", "High", "Very high"]
            task_name = "severity"
        else:
            labels = ["Healhty", "Leaf miner", "Rust", "Phoma", "Cercospora"]
            task_name = "biotic_stress"

        write_results(
            y_true=y_true,
            y_pred=y_pred,
            cm_target_names=labels,
            results_path=self.opt.results_path,
            task_name=task_name,
            experiment_name=self.opt.experiment_name,
        )

        return y_true, y_pred

    def get_n_params(self):
        weights_path = get_file_path(self.opt, "net_weights.pth")
        model = cnn_model(self.opt.model, self.opt.pretrained, 5, weights_path)
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
