import torch
from torch import nn
import numpy as np


# this is one way to define a network
class Net(nn.Module):
    def __init__(self, d):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu_(x)
        x = self.fc2(x)
        x = torch.relu_(x)
        x = self.fc3(x)
        x = torch.relu_(x)
        return x


# loading test data
testData = np.loadtxt("TestData.csv")

# loading model
model = torch.load('myModel.ckpt')

# saving predictions in csv
predictions = []
for rows in testData:
    from torch import Tensor
    row = Tensor(rows)
    prediction = model(row)
    print(float(prediction.data))
    predictions.append(float(prediction.data))


# saving predictions in the csv file
np.savetxt("i192140_Predictions.csv", predictions)

