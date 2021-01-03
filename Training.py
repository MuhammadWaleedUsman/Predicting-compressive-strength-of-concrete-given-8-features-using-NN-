# -*- coding: utf-8 -*-
# Code by Muhammad Waleed Usman
# i192140

# Libraries
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import cross_val_score

# Root mean square Error
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


# define network
class Net(nn.Module):
    def __init__(self, input):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input, 64)
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

# Data load
def dataload():
    data = np.loadtxt("TrainData.csv")
    labels = np.loadtxt("TrainLabels.csv")
    print(data.shape)
    print(labels.shape)
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)

    return data, labels


# Training
def Training(input, output, optimize, lrate, epochs):
    net = Net(input)
    if optimize is 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lrate)
        loss_func = RMSELoss()
        # Training
        Loss = []
        print("Starting Training")

        for epoch in range(epochs):
            print("Epoch " + str(epoch) + " of " + str(epochs))
            optimizer.zero_grad()  # Required step
            outputs = net(data.float())  # feed forward
            outputs = torch.squeeze(outputs, output)
            # loss = criterion(outputs, labels.float())  # Compute loss/error
            loss = loss_func(outputs, labels.float())  # must be (1. nn output, 2. target)
            optimizer.zero_grad()  # clear gradients for next train
            Loss.append(float(loss.data))

            loss.backward()  # Error Backpropagation
            # Freezing the weights
            optimizer.step()  # Weight update

        for param in net.parameters():
            param.requires_grad = False

        return Loss, loss, net

# Main
if __name__ == '__main__':
    # loading dats
    print("Data is loading!")
    data, labels = dataload()

    # Convert data into suitable data type
    data = Variable(torch.DoubleTensor(data), requires_grad=True)
    labels = Variable(torch.DoubleTensor(labels), requires_grad=False)

    # Hyperparameter
    input_features = 8
    output = 1
    optimize = 'Adam'
    lrate = 0.001
    epochs = 5000

    # Defining Network and training
    LossArray, lossMSE, model = Training(input_features, output, optimize, lrate, epochs)
    import matplotlib.pyplot as plt
    print(lossMSE.mean())
    plt.plot(LossArray)
    plt.grid()
    plt.show()

    # Saving model
    torch.save(model, 'myModelwaleed.ckpt')




