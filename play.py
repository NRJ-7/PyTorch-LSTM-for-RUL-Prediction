import torch
import torch.nn as nn
from torch.autograd import Variable
from loading_data import *
from model import *
from model_2 import *
from visualize import *
import numpy as np
from main import *
import plotly.express as px
import plotly.io as pio
from myfuncs import *
# pio.templates.default = "none"
pio.templates.default = "plotly_dark"

N_HIDDEN = 96  # NUMBER OF HIDDEN STATES
N_LAYER = 4  # NUMBER OF LSTM LAYERS
N_EPOCH = 150  # NUM OF EPOCHS
RUL_UPPER_BOUND = 135  # UPPER BOUND OF RUL
LR = 0.01  # LEARNING RATE


class LSTM_3(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size, hidden_size, num_layers, seq_length=1):
        super(LSTM_3, self).__init__()
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            # batch_first=True,
                            dropout=0.1)
        print("batch_first=False")
        self.fc_1 = nn.Linear(hidden_size, 16)  # fully connected 1
        self.fc_2 = nn.Linear(16, 8)  # fully connected 2
        self.fc = nn.Linear(8, 1)  # fully connected last layer

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: input features
        :return: prediction results
        """
        h_0 = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))  # internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        hn_o = torch.Tensor(hn.detach().numpy()[-1, :, :])
        hn_o = hn_o.view(-1, self.hidden_size)
        hn_1 = torch.Tensor(hn.detach().numpy()[1, :, :])
        hn_1 = hn_1.view(-1, self.hidden_size)

        out = self.relu(self.fc_1(self.relu(hn_o + hn_1)))
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc(out)
        return out


# model, criterion, optimizer, group_train, group_test, y_test, result, rmse = run()
# fetch basic information from data sets
group_train, group_test, y_test = load_FD001(norm_type='zscore', rul_ub=RUL_UPPER_BOUND)
y_test_array = y_test.to_numpy()  # convert to numpy array
num_train, num_test = len(group_train.size()), len(group_test.size())
input_size = group_train.get_group(1).shape[1] - 3  # number of features

# LSTM model initialization
model = LSTM_3(input_size, N_HIDDEN, N_LAYER)  # our lstm class
criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

result, rmse = train(model, num_train, num_test, group_train, group_test, criterion, optimizer, y_test_array)
# result, rmse = testing_function(model, num_test, group_test, y_test)
visualize(result, y_test, num_test, rmse)