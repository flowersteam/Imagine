import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import pickle
import os

cuda = torch.cuda.is_available()

n_points = 500000
batch_size = 64
lr = 0.005
n_objs = 2
n_epochs = 15

# balanced train set
x_train = np.random.uniform(0, 1, [n_points, n_objs])
y_train = (x_train > 0.5).astype(np.int).max(axis=1)
x_train_pos = x_train[y_train == 1][:n_points // 2]
y_train_pos = y_train[y_train == 1][:n_points // 2]
x_train_neg = np.random.uniform(0, 0.5, [n_points // 2, n_objs])
y_train_neg = np.array([0] * (n_points // 2))
y_train = np.concatenate([y_train_pos, y_train_neg])
x_train = np.concatenate([x_train_pos, x_train_neg], axis=0)
assert (y_train == 0).sum() == n_points // 2

# balanced valid set
x_valid = np.random.uniform(0, 1, [n_points, n_objs])
y_valid = (x_valid > 0.5).astype(np.int).max(axis=1)
x_valid_pos = x_valid[y_valid == 1][:n_points // 2]
y_valid_pos = y_valid[y_valid == 1][:n_points // 2]
x_valid_neg = np.random.uniform(0, 0.5, [n_points // 2, n_objs])
y_valid_neg = np.array([0] * (n_points // 2))
y_valid = np.concatenate([y_valid_pos, y_valid_neg])
x_valid = np.concatenate([x_valid_pos, x_valid_neg], axis=0)
assert (y_valid == 0).sum() == n_points // 2


class OrNet(nn.Module):
    def __init__(self, n_obj):
        super(OrNet, self).__init__()
        self.fc1 = nn.Linear(n_obj, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = 0.98 * x + 0.01
        return x


class MyAccumulatedAccuracyMetric():
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, targets):
        self.correct += ((outputs > 0.5).int() == targets.int()).sum()
        self.total += outputs.shape[0]
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Rel Dist Accuracy'


class OrDataset(Dataset):
    def __init__(self, x, y, split):
        self.split = split
        assert x.shape[0] == y.shape[0]

        inds = np.arange(x.shape[0])
        np.random.shuffle(inds)
        self.x = x[inds].astype(np.float)
        self.y = y[inds].astype(np.int)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (x, y) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs = model(x.float())

        loss = loss_fn(outputs, y.reshape(-1, 1).float())
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, y.reshape(-1, 1).float())

        if (batch_idx + 1) % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def valid_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (x, y) in enumerate(val_loader):

            outputs = model(x.float())
            loss = loss_fn(outputs, y.reshape(-1, 1).float())
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, y.reshape(-1, 1).float())

    return val_loss, metrics


if __name__ == "__main__":
    train_set = OrDataset(x_train, y_train, split='train')
    valid_set = OrDataset(x_valid, y_valid, split='valid')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, **kwargs)

    model = OrNet(n_objs)

    loss_fn = F.binary_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    best_accuracy = 0
    for epoch in range(n_epochs):

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, 200,
                                          [MyAccumulatedAccuracyMetric()])

        scheduler.step()
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        train_accuracy = metrics[0].value()

        val_loss, metrics = valid_epoch(valid_loader, model, loss_fn, cuda, [MyAccumulatedAccuracyMetric()])
        val_loss /= len(valid_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        valid_accuracy = metrics[0].value()
        if metrics[0].value() > best_accuracy:
            best_accuracy = valid_accuracy
            is_best = True
        else:
            is_best = False

        print(message)

    or_params = dict(fc1_weight=model.fc1.weight.data.numpy(),
                     fc1_bias=model.fc1.bias.data.numpy(),
                     fc2_weight=model.fc2.weight.data.numpy(),
                     fc2_bias=model.fc2.bias.data.numpy(),
                     fc3_weight=model.fc3.weight.data.numpy(),
                     fc3_bias=model.fc3.bias.data.numpy(),
                     description='fc1:[n_obj, 16]->tanh->fc2:[16, 16]->tanh->fc3:[16, 1]-> sigmoid',
                     layers=[[n_objs, 16], [16, 16], [16, 1]],
                     activations=['tanh', 'tanh', 'sigmoid']
                     )

    with open(os.path.dirname(os.path.realpath(__file__)) + '/or_params_{}objs.pk'.format(n_objs), 'wb') as f:
        pickle.dump(or_params, f)
    stop = 1
