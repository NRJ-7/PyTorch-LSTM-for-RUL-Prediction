import torch
import torch.nn as nn
from torch.autograd import Variable


class RUL_LSTM_PAPER(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size, lstm_layer_sizes=[32, 64], ff_layer_sizes=[8, 8], seq_length=1):
        super(RUL_LSTM_PAPER, self).__init__()
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.1)
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
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        hn_o = torch.Tensor(hn.detach().numpy()[-1, :, :])
        hn_o = hn_o.view(-1, self.hidden_size)
        # hn_1 = torch.Tensor(hn.detach().numpy()[1, :, :])
        # hn_1 = hn_1.view(-1, self.hidden_size)

        out = self.relu(self.fc_1(self.relu(hn_o)))
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
    
    import torch
import torch.nn as nn


class RUL_LSTM(nn.Module):
    """LSTM architecture with 2 LSTM layers and feed-forward layers"""

    def __init__(self, input_size, lstm_layer_sizes=[32, 64], ff_layer_sizes=[8, 8], seq_length=1):
        super(RUL_LSTM, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length

        # LSTM layers
        self.lstm_1 = nn.LSTM(input_size=input_size, hidden_size=lstm_layer_sizes[0], batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=lstm_layer_sizes[0], hidden_size=lstm_layer_sizes[1], batch_first=True)

        # Fully connected layers
        self.fc_1 = nn.Linear(lstm_layer_sizes[1], ff_layer_sizes[0])
        self.fc_2 = nn.Linear(ff_layer_sizes[0], ff_layer_sizes[1])
        self.fc_out = nn.Linear(ff_layer_sizes[1], 1)  # Single-dimensional output layer

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass of the model
        :param x: Input tensor of shape (batch_size, seq_length, input_size)
        :return: Output tensor of shape (batch_size, 1)
        """
        # First LSTM layer
        output, (hn_1, cn_1) = self.lstm_1(x)

        # Second LSTM layer
        output, (hn_2, cn_2) = self.lstm_2(output)

        # Use the hidden state of the last LSTM layer
        hn_2 = hn_2[-1, :, :]  # Shape: (batch_size, lstm_layer_sizes[1])

        # Fully connected layers
        out = self.relu(self.fc_1(hn_2))
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc_out(out)  # Final output layer

        return out
