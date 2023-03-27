import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import argparse
from utils.categorical_var import getOrdinal
from preprocessing.toke import tokenize_data
import pathlib

# todo: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=100)
    parser.add_argument('--num-epochs-train', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


class Datamaker(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class Classification(nn.Module):
    def __init__(self, inputs_size):
        super(Classification, self).__init__()
        self.layer_1 = nn.Linear(inputs_size, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x




def accuracy(y_pred, y_test):
    y_pred_tag = torch.max(y_pred, 1).indices

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]

    return acc


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load data with according genres
    # ['Fantasy', 'Paranormal', 'Magic', 'Supernatural', 'Adventure']
    df_Fantasy = pd.read_csv("data/df_genres/df_Fantasy.csv")
    df_Paranormal = pd.read_csv("data/df_genres/df_Paranormal.csv")
    df_Magic = pd.read_csv("data/df_genres/df_Magic.csv")
    df_Supernatural = pd.read_csv("data/df_genres/df_Supernatural.csv")
    df_Adventure = pd.read_csv("data/df_genres/df_Adventure.csv")
    df = pd.concat([df_Fantasy, df_Paranormal, df_Magic, df_Supernatural, df_Adventure])
    df.drop_duplicates(inplace=True)
    df_tokenized = tokenize_data(df)

    df_ordinals = getOrdinal(df_tokenized, "rating_value")

    X = df_ordinals.iloc[:, 0:-1]
    y = df_ordinals.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=args.seed)

    train_data = Datamaker(torch.tensor(X_train.values).to(torch.float32),
                           torch.tensor(y_train.values).to(torch.float32))

    val_data = Datamaker(torch.tensor(X_val.values).to(torch.float32),
                         torch.tensor(y_val.values).to(torch.float32))

    test_data = Datamaker(torch.tensor(X_test.values).to(torch.float32),
                          torch.tensor(y_test.values).to(torch.float32))

    train_loader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.bs)
    test_loader = DataLoader(dataset=test_data, batch_size=args.bs)

    num_features = len(X_train.columns)
    model = Classification(num_features)
    model.to(device)
    model_saved_name = "lstm_model_lr_" + str(args.lr) + "_bs" + str(args.bs) + "_ep_" + str(args.num_epochs_train)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for e in range(args.num_epochs_train):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch, len(X_batch))
            y_batch = y_batch.to(int)

            loss = criterion(y_pred, y_batch)
            acc = accuracy(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

        print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    pathlib.Path('export_models/').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 'export_models/' + model_saved_name + '.pt')

    # Todo: put this in a new file when you saved the model
    # Todo: load model

    y_pred_list = []
    acc_list = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_batch = y_batch.to(int)

            acc = accuracy(y_test_pred, y_batch)

            y_pred_tag = torch.max(y_test_pred, 1).indices
            y_pred_list.append(y_pred_tag.cpu().numpy())
            acc_list.append(acc.item())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_list = [item for sublist in y_pred_list for item in sublist]
    confusion_matrix(y_test, y_pred_list)
    acc = np.array(acc_list).mean()
    print("val accuracy is:" + str(acc))
