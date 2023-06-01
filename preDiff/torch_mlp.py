import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch


def draw_plot(train_data, test_data, train_label, test_label, title, xlabel, ylabel, epoch):
    x = range(0, epoch)
    plt.plot(x, train_data, color="blue", label=train_label, linewidth=2)
    plt.plot(x, test_data, color="orange", label=test_label, linewidth=2)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel=xlabel, fontsize=15)
    plt.ylabel(ylabel=ylabel, fontsize=15)


def draw_plots(epoch, train_loss_list, train_f1_acc_list, train_auc_acc_list, train_acc_list,
              val_loss_list, val_f1_acc_list, val_auc_acc_list, val_acc_list):
    plt.figure(figsize=(11, 28))

    plt.subplot(4, 1, 1)
    draw_plot(train_loss_list, val_loss_list, "train_loss Line", "val_loss Line", "Loss_curve", "Epochs", "Loss", epoch)
    plt.legend()
    plt.subplot(4, 1, 2)
    draw_plot(train_auc_acc_list, val_auc_acc_list, "train_auc Line", "val_auc Line", "AUC_curve", "Epochs", "Accuracy", epoch)
    plt.legend()
    plt.subplot(4, 1, 3)
    draw_plot(train_f1_acc_list, val_f1_acc_list, "train_f1 Line", "val_f1 Line", "F1_curve", "Epochs", "Accuracy", epoch)
    plt.legend()
    plt.subplot(4, 1, 4)
    draw_plot(train_acc_list, val_acc_list, "train_acc Line", "val_acc Line", "Acc_curve", "Epochs", "Accuracy", epoch)
    plt.legend()
    plt.savefig("Loss&acc.jpg")


def accuracy(y, pred):
    result = 0
    lenth = len(y)
    for i in range(lenth):
        if y[i] == pred[i]:
            result += 1
    return result / lenth


def str2lst(i):
    return list(map(float, i.split(" ")))


def preprocess(dataPath, batch_size, device):
    data = pd.read_csv(dataPath)
    data_x = data["embed"].tolist()
    data_y = torch.as_tensor(data["difficulty"]).long().to(device)
    data_x = [str2lst(str(i)) for i in data_x]
    data_x = torch.tensor(data_x).to(device)
    data_dataset = TensorDataset(data_x, data_y)
    data = DataLoader(data_dataset, batch_size=batch_size)
    return data


class MLP(nn.Module):
    def __init__(self, layers, modelDir, device):
        super(MLP, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-chinese",
                                                       cache_dir=modelDir)

        self.layers = nn.Sequential().to(device)
        for i in range(len(layers)-2):
            self.layers.add_module("Linear%d" % (i), nn.Linear(layers[i], layers[i+1]).to(device))
            self.layers.add_module("ReLU%d" % (i), nn.ReLU(inplace=True).to(device))
        self.layers.add_module("Linear%d" % (len(layers)-2), nn.Linear(layers[-2], layers[-1]).to(device))
        self.device = device

    def forward(self, x):
        return self.layers(x)


def train(train, test, layers, epoch,
          modelDir="../model_save",
          loss_and_acc_curve=True,
          device=torch.device("cpu")):
    mlp = MLP(layers, modelDir, device)
    optimizer = optim.SGD(params=mlp.parameters(), lr=1e-2)
    loss_func = nn.CrossEntropyLoss().to(device)
    train_loss_list = []
    train_f1_acc_list = []
    train_auc_acc_list = []
    train_acc_list = []
    val_loss_list = []
    val_f1_acc_list = []
    val_auc_acc_list = []
    val_acc_list = []

    for i_epoch in range(epoch):
        best_acc = 0
        total_loss = 0
        total_val_loss = 0

        epoch_size_trn = len(train)
        epoch_size_val = len(test)

        sm = nn.Softmax(dim=1).to(device)

        train_target = None
        val_target = None
        train_f1 = None
        train_auc = None
        val_f1 = None
        val_auc = None

        train_pbar = tqdm(train)
        for x, y in train_pbar:
            optimizer.zero_grad()

            pred = mlp(x)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            argpred = torch.argmax(pred, dim=1)
            softpred = sm(pred)

            if train_target is None:
                train_target = y
            else:
                train_target = torch.cat((train_target, y), dim=0)

            if train_f1 is None:
                train_f1 = argpred
            else:
                train_f1 = torch.cat((train_f1, argpred), dim=0)

            if train_auc is None:
                train_auc = softpred
            else:
                train_auc = torch.cat((train_auc, softpred), dim=0)
            train_pbar.set_description("[Train][Epoch: %d]" % i_epoch)

        train_loss = total_loss / epoch_size_trn
        train_f1_acc = f1_score(train_target.detach().cpu().numpy(), train_f1.detach().cpu().numpy(), average='micro')
        train_auc_acc = roc_auc_score(train_target.detach().cpu().numpy(), train_auc.detach().cpu().numpy(), multi_class='ovr')
        train_acc = accuracy(train_target.detach().cpu().numpy(), train_f1.detach().cpu().numpy())

        test_pbar = tqdm(test)
        for x, y in test_pbar:
            optimizer.zero_grad()

            pred = mlp(x)
            loss = loss_func(pred, y)

            total_val_loss += loss.item()

            argpred = torch.argmax(pred, dim=1)
            softpred = sm(pred) # 加 softmax 给 auc

            if val_target is None:
                val_target = y
            else:
                val_target = torch.cat((val_target, y), dim=0)

            if val_f1 is None:
                val_f1 = argpred
            else:
                val_f1 = torch.cat((val_f1, argpred), dim=0)

            if val_auc is None:
                val_auc = softpred
            else:
                val_auc = torch.cat((val_auc, softpred), dim=0)
            test_pbar.set_description("[Test][Epoch: %d]" % i_epoch)

        val_loss = total_val_loss / epoch_size_val
        val_f1_acc = f1_score(val_target.detach().cpu().numpy(), val_f1.detach().cpu().numpy(), average='micro')
        val_auc_acc = roc_auc_score(val_target.detach().cpu().numpy(), val_auc.detach().cpu().numpy(), multi_class='ovr')
        val_acc = accuracy(val_target.detach().cpu().numpy(), val_f1.detach().cpu().numpy())

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(mlp.state_dict(), "weight/mlp_weight-epoch%d.pt" % (i_epoch))

        train_loss_list.append(train_loss)
        train_f1_acc_list.append(train_f1_acc)
        train_auc_acc_list.append(train_auc_acc)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_f1_acc_list.append(val_f1_acc)
        val_auc_acc_list.append(val_auc_acc)
        val_acc_list.append(val_acc)

    if loss_and_acc_curve:
        draw_plots(epoch, train_loss_list, train_f1_acc_list, train_auc_acc_list, train_acc_list,
                  val_loss_list, val_f1_acc_list, val_auc_acc_list, val_acc_list)


if __name__ == "__main__":
    epoch = 300
    device = torch.device("cuda")
    batch_size = 256
    curve = True

    testSet = preprocess("../data/test.csv", batch_size, device)
    trainSet = preprocess("../data/train.csv", batch_size, device)

    train(trainSet, testSet, [768, 64, 4], epoch, device=device, loss_and_acc_curve=curve)
