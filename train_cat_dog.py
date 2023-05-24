import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import models

from network.AlexNet import AlexNet
from network.ResNet import ResNet
from utils.CatDogDataset import CatDogDataset
from utils.EarlyStopping import EarlyStopping

label_dict = dict(
    cats='Chat',
    dogs='Chien'
)


def train(net, batch_size=64, learning_rate=1e-4, num_epochs=50, patience=10, delta=0, path='checkpoint.pt'):
    train_dataset = CatDogDataset(split="training_set", label_dict=label_dict)
    val_dataset = CatDogDataset(split="test_set", label_dict=label_dict)

    training_generator = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_generator = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []

    early_stopping = EarlyStopping(patience=patience, delta=delta, path=path, verbose=True)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for i in range(num_epochs):
        net.train()
        for j, sample in enumerate(training_generator):
            x, y = sample
            y = y.to(torch.long).cuda()
            optimizer.zero_grad()
            out = net(x.cuda())
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            print('\r Epoch', i, 'Step', j, ':', str(loss.data.cpu().numpy()), end="")

        net.eval()
        accuracy = []
        for j, sample in enumerate(test_generator):
            x, y = sample
            y = y.to(torch.long).cuda()
            out = net(x.cuda())
            best = np.argmax(out.data.cpu().numpy(), axis=-1)
            accuracy.extend(list(best == y.data.cpu().numpy()))
            loss = loss_func(out, y)
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_losses = []
        valid_losses = []
        print('\n Accuracy is ', str(np.mean(accuracy) * 100), " Validation loss is ", str(valid_loss))
        early_stopping(valid_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("done")
    return avg_train_losses, avg_valid_losses


def plot(title="Model", tr_losses=None, val_losses=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(list(range(1, len(tr_losses) + 1)), tr_losses, label='train')
    plt.plot(list(range(1, len(val_losses) + 1)), val_losses, label='val')
    idx = val_losses.index(min(val_losses)) + 1
    plt.axvline(idx, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Trainer for the Cat or Dog app.")

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Destination folder to save the model after training ends.",
        default="Custom",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Number of epochs max for training",
        default=1,
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Name of the model to train",
        default="Custom",
    )

    args = parser.parse_args()
    if Path(args.path).is_file():
        print(f"Model {args.path} already exists do you want to overwrite ?")
        y = input('Type "Yes" or "No": ')
        if y != "Yes":
            print("Aborting.")
            sys.exit()
    model_name = args.model
    if model_name == "alexnet":
        net = AlexNet(num_classes=len(label_dict.keys()))
        lr = 1e-4
    elif model_name == "resnet":
        net = ResNet(pretrained=False, num_classes=2).get_model()
        lr = 1e-3
    elif model_name == "alexnet-pretrained":
        net = models.alexnet(pretrained=True)
        net.classifier[6] = nn.Linear(in_features=net.classifier[6].in_features, out_features=len(label_dict.keys()))
        lr = 1e-4
    elif model_name == "resnet-pretrained":
        net = ResNet(pretrained=True, num_classes=2).get_model()
        lr = 1e-3
    else:
        print("Please select a valid model")
        net = None

    if net is not None:
        net = net.cuda()
        avg_train_losses, avg_valid_losses = train(net, learning_rate=lr, num_epochs=args.epochs, path=args.path)
        plot(model_name, avg_train_losses, avg_valid_losses)
    else:
        exit(-1)
