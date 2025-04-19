"""
LSTM architecture for RUL prediction
author: Neeraj Sohani
date: 18-04-2025
description:
    This module defines an LSTM model for predicting Remaining Useful Life (RUL) of machinery.
    The model consists of a stack of LSTM layers followed by fully connected layers.
    The LSTM layers are defined in a list, allowing for flexible architecture.
    The model can also be initialized with learnable initial states for the LSTM layers.
    The model can handle variable-length sequences using packed sequences.
"""

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class RUL_Model(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size, lstm_hidden_sizes, lstm_layer_sizes, dropout=0.1):
        super(RUL_Model, self).__init__()
        
        assert len(lstm_hidden_sizes) == len(lstm_layer_sizes), "hidden_sizes and layer_sizes must be the same length"
        
        self.input_size = input_size  # input size
        self.lstm_hidden_sizes = lstm_hidden_sizes  # list of hidden sizes
        self.lstm_layer_sizes = lstm_layer_sizes  # number of lstm layers for each hidden size
        self.lstm_stack_size = sum(lstm_layer_sizes)  # total number of lstm layers

        self.lstm_list = self.create_lstm_stack()
        self.fc_1 = nn.Linear(self.lstm_hidden_sizes[-1], 8)  # fully connected 1
        self.fc_2 = nn.Linear(8, 8)  # fully connected 2
        self.fc = nn.Linear(8, 1)  # fully connected last layer

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def create_lstm_stack(self):
        """Create LSTM stack based on hidden sizes and layer sizes"""
        lstm_list = nn.ModuleList()
        input_size = self.input_size
        for hidden_size, num_layers in zip(self.lstm_hidden_sizes, self.lstm_layer_sizes):
            lstm_list.append(nn.LSTM(input_size, 
                                     hidden_size, 
                                     num_layers, 
                                     batch_first=True))
            input_size = hidden_size
        return lstm_list

    def forward(self, x, lengths=None):
        """
        Forward pass through the LSTM stack and fully connected layers.
        :param x: input features
        :return: prediction results
        """
        input = pack_padded_sequence(x,
                                     lengths.cpu(), 
                                     batch_first=True, 
                                     enforce_sorted=False) if lengths is not None else x
        packed_lstm_outs = [input]
        for lstm in self.lstm_list:
            packed_lstm_outs.append(lstm(packed_lstm_outs[-1])[0])
            
        lstm_outs = [pad_packed_sequence(out,
                                         batch_first=True,
                                         padding_value=0.0,
                                         total_length=x.size(1))[0] for out in packed_lstm_outs] if lengths is not None else packed_lstm_outs
        
        out_1 = self.relu(self.fc_1(lstm_outs[-1]))
        out_2 = self.relu(self.fc_2(out_1))
        out_dp = self.dropout(out_2)
        out = self.fc(out_dp)
        
        return out #, (hn, cn), out_1, out_2, out_dp, output
    
    
class RUL_Model_LearnableInputs(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size, lstm_hidden_sizes, lstm_layer_sizes, dropout=0.1):
        super(RUL_Model_LearnableInputs, self).__init__()
        
        assert len(lstm_hidden_sizes) == len(lstm_layer_sizes), "hidden_sizes and layer_sizes must be the same length"
        
        self.input_size = input_size  # input size
        self.lstm_hidden_sizes = lstm_hidden_sizes  # list of hidden sizes
        self.lstm_layer_sizes = lstm_layer_sizes  # number of lstm layers for each hidden size
        self.lstm_stack_size = sum(lstm_layer_sizes)  # total number of lstm layers

        self.lstm_list = self.create_lstm_stack()
        self.initial_state_list = self.create_learnable_initial_states()
        self.fc_1 = nn.Linear(self.lstm_hidden_sizes[-1], 8)  # fully connected 1
        self.fc_2 = nn.Linear(8, 8)  # fully connected 2
        self.fc = nn.Linear(8, 1)  # fully connected last layer

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def create_lstm_stack(self):
        """Create LSTM stack based on hidden sizes and layer sizes"""
        lstm_list = nn.ModuleList()
        input_size = self.input_size
        for hidden_size, num_layers in zip(self.lstm_hidden_sizes, self.lstm_layer_sizes):
            lstm_list.append(nn.LSTM(input_size, 
                                     hidden_size, 
                                     num_layers, 
                                     batch_first=True))
            input_size = hidden_size
        return lstm_list
    
    def create_learnable_initial_states(self):
        initial_states = nn.ParameterList()
        for hidden_size, num_layers in zip(self.lstm_hidden_sizes, self.lstm_layer_sizes):
            h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
            c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
            initial_states.append((h0, c0))
        return initial_states
        

    def forward(self, x, lengths=None):
        """
        Forward pass through the LSTM stack and fully connected layers.
        :param x: input features
        :return: prediction results
        """
        batch_size = x.size(0)
        input = pack_padded_sequence(x,
                                     lengths.cpu(), 
                                     batch_first=True, 
                                     enforce_sorted=False) if lengths is not None else x
        packed_lstm_outs = [input]
        for lstm, (h0, c0) in zip(self.lstm_list, self.initial_state_list):
            h0_batch = h0.expand(-1, batch_size, -1).contiguous()
            c0_batch = c0.expand(-1, batch_size, -1).contiguous()
            packed_lstm_outs.append(lstm(packed_lstm_outs[-1], (h0_batch, c0_batch))[0])
            
        lstm_outs = [pad_packed_sequence(out,
                                         batch_first=True,
                                         padding_value=0.0,
                                         total_length=x.size(1))[0] for out in packed_lstm_outs] if lengths is not None else packed_lstm_outs
        
        out_1 = self.relu(self.fc_1(lstm_outs[-1]))
        out_2 = self.relu(self.fc_2(out_1))
        out_dp = self.dropout(out_2)
        out = self.fc(out_dp)
        
        return out #, (hn, cn), out_1, out_2, out_dp, output