"""
LSTM architecture for RUL prediction
author: Neeraj Sohani
date: 18-04-2025 (Note: Corrected date logic if needed)
description:
    This module defines an LSTM model for predicting Remaining Useful Life (RUL) of machinery.
    The model consists of a stack of LSTM layers followed by fully connected layers.
    The LSTM layers are defined in a list, allowing for flexible architecture.
    The model can also be initialized with learnable initial states for the LSTM layers.
    The model can handle variable-length sequences using packed sequences.
    Code adapted for automatic GPU (MPS/CUDA) usage if available.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class RUL_Model(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size, lstm_hidden_sizes, lstm_layer_sizes, lstm_dropout_rate=0, output_dropout_rate=0):
        super(RUL_Model, self).__init__()

        assert len(lstm_hidden_sizes) == len(lstm_layer_sizes), "hidden_sizes and layer_sizes must be the same length"

        self.input_size = input_size  # input size
        self.lstm_hidden_sizes = lstm_hidden_sizes  # list of hidden sizes
        self.lstm_layer_sizes = lstm_layer_sizes  # number of lstm layers for each hidden size
        self.lstm_dropout_rate = lstm_dropout_rate
        self.output_dropout_rate = output_dropout_rate
        self.lstm_stack_size = sum(lstm_layer_sizes)  # total number of lstm layers
        self.num_lstm_sub_stacks = len(self.lstm_hidden_sizes)
        self.perform_lstm_dropout = (lstm_dropout_rate > 0)
        self.lstm_dropout_layers = nn.ModuleList() # Dropout layers between lstm sub-stacks

        self.lstm_stack = self.create_lstm_stack()
        # Using Sequential for the fully connected part can be slightly cleaner
        self.fc_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_sizes[-1], 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout(self.output_dropout_rate) if self.output_dropout_rate > 0 else nn.Identity(),
            nn.Linear(8, 1)
        )

    def create_lstm_stack(self):
        """Create LSTM stack based on hidden sizes and layer sizes"""
        lstm_list = nn.ModuleList()
        input_size = self.input_size # Renamed variable for clarity
        for i, (hidden_size, num_layers) in enumerate(zip(self.lstm_hidden_sizes, self.lstm_layer_sizes)):
            lstm_list.append(nn.LSTM(input_size,
                                     hidden_size,
                                     num_layers,
                                     batch_first=True,
                                     # Apply LSTM's internal dropout if num_layers > 1
                                     dropout=self.lstm_dropout_rate if num_layers > 1 else 0))
            input_size = hidden_size # Input to next LSTM is output of current one
            # Apply dropout between different LSTM sub-stacks (different hidden sizes)
            if self.perform_lstm_dropout and (i < self.num_lstm_sub_stacks - 1):
                self.lstm_dropout_layers.append(nn.Dropout(self.lstm_dropout_rate))

        return lstm_list

    def forward(self, input_seq, lengths=None):
        """
        Forward pass through the LSTM stack and fully connected layers.
        Assumes 'input_seq' and the model instance have already been moved to the target device.
        :param input_seq: input features (batch_size, seq_len, input_size) or (seq_len, input_size)
        :param lengths: tensor of sequence lengths for packed sequence handling (must be on CPU)
        :return: prediction results
        """
        # Input should be on the correct device before calling forward
        # device = input_seq.device # Get device from input if needed internally

        # Handle non-batched input
        is_batched = input_seq.dim() == 3
        if not is_batched:
            input_seq = input_seq.unsqueeze(0)

        # Handle potential padding
        is_padded_sequence = (lengths is not None)

        current_input = input_seq
        lstm_output = None # To store the output of the final LSTM layer

        for i, lstm_layer in enumerate(self.lstm_stack):
            if not is_padded_sequence:
                # No packing needed
                lstm_output, _ = lstm_layer(current_input) # Use default (zero) initial states
            else:
                # Pack sequence
                packed_input = pack_padded_sequence(current_input,
                                                    lengths.cpu(), # lengths must be on CPU
                                                    batch_first=True,
                                                    enforce_sorted=False)
                # Pass packed sequence through LSTM
                packed_output, _ = lstm_layer(packed_input) # Use default (zero) initial states
                # Unpack sequence
                lstm_output, _ = pad_packed_sequence(packed_output,
                                                     batch_first=True,
                                                     padding_value=0.0,
                                                     # Use original max length if needed, otherwise calculates automatically
                                                     total_length=input_seq.size(1) if is_batched else 1
                                                     )
            # Apply dropout between LSTM sub-stacks if applicable
            if self.perform_lstm_dropout and (i < self.num_lstm_sub_stacks - 1):
                current_input = self.lstm_dropout_layers[i](lstm_output)
            else:
                current_input = lstm_output # Output of current becomes input for next

        # Pass through fully connected layers
        out = self.fc_layers(lstm_output)

        # Squeeze output if input was not batched
        if not is_batched:
            out = out.squeeze(0)

        return out


class RUL_Model_LearnableStates(nn.Module):
    """LSTM architecture with learnable initial states"""

    def __init__(self, input_size, lstm_hidden_sizes, lstm_layer_sizes, lstm_dropout_rate=0, output_dropout_rate=0, state_init=False):
        super(RUL_Model_LearnableStates, self).__init__()

        assert len(lstm_hidden_sizes) == len(lstm_layer_sizes), "hidden_sizes and layer_sizes must be the same length"

        self.input_size = input_size
        self.lstm_hidden_sizes = lstm_hidden_sizes
        self.lstm_layer_sizes = lstm_layer_sizes
        self.lstm_dropout_rate = lstm_dropout_rate
        self.output_dropout_rate = output_dropout_rate
        self.state_init = state_init
        self.lstm_stack_size = sum(lstm_layer_sizes)
        self.num_lstm_sub_stacks = len(self.lstm_hidden_sizes)
        self.perform_lstm_dropout = (lstm_dropout_rate > 0)
        self.lstm_dropout_layers = nn.ModuleList()
        

        self.lstm_stack = self.create_lstm_stack()

        # Initialize learnable initial states (will be moved to device with model.to(device))
        self.initial_hidden_state_list = nn.ParameterList()
        self.initial_cell_state_list = nn.ParameterList()
        self.fill_learnable_initial_states()

        self.fc_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_sizes[-1], 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout(self.output_dropout_rate) if self.output_dropout_rate > 0 else nn.Identity(),
            nn.Linear(8, 1)
        )

    def create_lstm_stack(self):
        """Create LSTM stack based on hidden sizes and layer sizes"""
        lstm_list = nn.ModuleList()
        input_size = self.input_size
        for i, (hidden_size, num_layers) in enumerate(zip(self.lstm_hidden_sizes, self.lstm_layer_sizes)):
            lstm_list.append(nn.LSTM(input_size,
                                     hidden_size,
                                     num_layers,
                                     batch_first=True,
                                     dropout=self.lstm_dropout_rate if num_layers > 1 else 0))
            input_size = hidden_size
            if self.perform_lstm_dropout and (i < self.num_lstm_sub_stacks - 1):
                self.lstm_dropout_layers.append(nn.Dropout(self.lstm_dropout_rate))

        return lstm_list

    def fill_learnable_initial_states(self):
        """Initialize learnable h0, c0 for each LSTM sub-stack"""
        for hidden_size, num_layers in zip(self.lstm_hidden_sizes, self.lstm_layer_sizes):
            # Initialize parameters (size: num_layers, 1, hidden_size)
            # The '1' dimension will be expanded to batch_size in forward
            h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
            c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
            if self.state_init:
                nn.init.xavier_uniform_(h0) # Example initialization
                nn.init.xavier_uniform_(c0) # Example initialization
            self.initial_hidden_state_list.append(h0)
            self.initial_cell_state_list.append(c0)


    def forward(self, input_seq, lengths=None):
        """
        Forward pass using learnable initial states.
        Assumes 'input_seq' and the model instance have already been moved to the target device.
        :param input_seq: input features (batch_size, seq_len, input_size) or (seq_len, input_size)
        :param lengths: tensor of sequence lengths for packed sequence handling (must be on CPU)
        :return: prediction results
        """
        # Parameters (h0, c0) and input should be on the correct device already

        is_batched = input_seq.dim() == 3
        if not is_batched:
            input_seq = input_seq.unsqueeze(0)
        batch_size = input_seq.size(0)

        is_padded_sequence = (lengths is not None)

        current_input = input_seq
        lstm_output = None

        # Iterate through LSTM layers and their corresponding initial states
        for i, (lstm_layer, h0, c0) in enumerate(zip(self.lstm_stack, self.initial_hidden_state_list, self.initial_cell_state_list)):
            # Expand learnable states to match batch size.
            # .contiguous() is important after expand for performance/compatibility.
            # Parameters h0/c0 are already on the correct device via model.to(device)
            h0_batch = h0.expand(-1, batch_size, -1).contiguous()
            c0_batch = c0.expand(-1, batch_size, -1).contiguous()
            # --- *** BUG FIX: Corrected h0/c0 assignment below *** ---
            # h0_batch = h0.expand(-1, batch_size, -1).contiguous() # Corrected
            # c0_batch = c0.expand(-1, batch_size, -1).contiguous() # Corrected

            initial_state = (h0_batch, c0_batch)

            if not is_padded_sequence:
                lstm_output, _ = lstm_layer(current_input, initial_state)
            else:
                packed_input = pack_padded_sequence(current_input,
                                                    lengths.cpu(),
                                                    batch_first=True,
                                                    enforce_sorted=False)
                # Pass packed sequence and initial state through LSTM
                packed_output, _ = lstm_layer(packed_input, initial_state)
                # Unpack sequence
                lstm_output, _ = pad_packed_sequence(packed_output,
                                                     batch_first=True,
                                                     padding_value=0.0,
                                                     total_length=input_seq.size(1))

            # Apply dropout between LSTM sub-stacks if applicable
            if self.perform_lstm_dropout and (i < self.num_lstm_sub_stacks - 1):
                current_input = self.lstm_dropout_layers[i](lstm_output)
            else:
                current_input = lstm_output

        # Pass through fully connected layers
        out = self.fc_layers(lstm_output)

        if not is_batched:
            out = out.squeeze(0)

        return out