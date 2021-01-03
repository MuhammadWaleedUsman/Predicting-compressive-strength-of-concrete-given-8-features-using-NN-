
import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from pprint import pprint

data = np.loadtxt("TrainData.csv")
labels = np.loadtxt("TrainLabels.csv")
print(data.shape)
print(labels.shape)
data = torch.from_numpy(data)
labels = torch.from_numpy(labels)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


# define network
class Net(nn.Module):
    def __init__(self, input_features = 8):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_features, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu_(x)
        x = self.fc2(x)
        x = torch.relu_(x)
        x = self.fc3(x)
        x = torch.relu_(x)
        return x


if __name__ == '__main__':
    # Initialize Network
    input_features = 8
    output = 1
    learning_rate = 0.01

    # using skorch implement the GridsearchCV
    net = NeuralNetRegressor(
        Net,
        max_epochs=10000,
        lr=0.1,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )

    # Training
    Loss = []
    params = {
        'lr': [0.01, 0.001, 0.0001, 0.1, 0.02, 0.002, 0.0002, 0.2, 0.05, 0.005, 0.0005, 0.5],
        'max_epochs': [1000, 3000, 8000],

    }
    gs = GridSearchCV(net, params, refit=False, cv=5, scoring='neg_root_mean_squared_error', verbose=1)

    gs.fit(data, labels)
    print(gs.best_score_, gs.best_params_)
    pprint(gs.cv_results_)