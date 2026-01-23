import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np

from project_utils import set_seed


class StrokeDataset(Dataset):
    def __init__(self, X_set, y_set):
        self.X_set = torch.from_numpy(X_set.values.astype(np.float32))
        self.y_set = torch.from_numpy(y_set.values.astype(np.float32))

    def __len__(self):
        return len(self.X_set)

    def __getitem__(self, index):
        x = self.X_set[index]
        y = self.y_set[index]
        return x, y


class DeepModelClassifier:
    def __init__(
        self,
        epochs,
        batch_size,
        in_features,
        hidden_layer,
        hidden_layer2,
        dropout,
        focal_alpha,
        seed,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.in_features = in_features
        self.hidden_layer = hidden_layer
        self.hidden_layer2 = hidden_layer2
        self.dropout = dropout
        self.focal_alpha = focal_alpha
        self.seed = seed

    def fit(self, X_train, y_train):
        set_seed(self.seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainset = StrokeDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, shuffle=True, batch_size=self.batch_size, num_workers=4
        )
        self.net = Net(
            self.in_features, self.hidden_layer, self.hidden_layer2, self.dropout
        ).to(device)
        criterion = FocalLoss(alpha=self.focal_alpha)
        optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        for epoch in tqdm(range(self.epochs)):
            running_loss = 0.0
            self.net.train()

            for data in train_loader:
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                predictions = self.net(inputs)
                loss = criterion(predictions, labels.view(-1, 1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

    def predict_proba(self, X_test, y_test):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.eval()
        testset = StrokeDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            testset, shuffle=False, batch_size=self.batch_size, num_workers=4
        )

        predictions = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.net(inputs)
                predictions.append(torch.sigmoid(outputs).cpu().numpy())

        return np.concatenate(predictions, axis=0).ravel()


class Net(nn.Module):
    def __init__(self, in_features, hidden_layer, hidden_layer2, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_layer)
        self.fc2 = nn.Linear(in_features=hidden_layer, out_features=hidden_layer2)
        self.fc3 = nn.Linear(in_features=hidden_layer2, out_features=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
class FocalLoss(nn.Module):
    def __init__(
        self, alpha=0.75, gamma=2.0
    ):  # alpha is weight for positive cases, gamma is how to focus on hard patients (that model gets wrong)
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean()
