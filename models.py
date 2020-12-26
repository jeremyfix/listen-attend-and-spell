#!/usr/bin/env python3

# Standard imports
# External imports
import torch
import torch.nn as nn


class Listener(nn.Module):

    def __init__(self, n_mels):
        super(Listener, self).__init__()
        # Pyramidal Bidirectionnal LSTM
        self.num_hidden = 256
        self.l1 = nn.LSTM(n_mels,
                          self.num_hidden,
                          bidirectional=True)
        self.l2 = nn.LSTM(2 * 2 * self.num_hidden,
                          self.num_hidden,
                          bidirectional=True)
        self.l3 = nn.LSTM(2 * 2 * self.num_hidden,
                          self.num_hidden,
                          bidirectional=True)

    def forward(self, inputs1):
        """
        Args:
            inputs : (batch, num_mels, time)
        """
        # Get the device to create tensors on the same
        device = inputs1.device

        # Transpose the inputs to be (time, batch, num_mels)
        inputs1 = inputs1.permute(2, 0, 1)

        seq_len, batch_size, num_mels = inputs1.shape

        # The initial hidden and cell states
        h0 = torch.zeros(2, batch_size, self.num_hidden,
                         device=device,
                         requires_grad=False)
        c0 = torch.zeros(2, batch_size, self.num_hidden,
                         device=device,
                         requires_grad=False)

        # out1 is (seq_len, batch_size, 2*num_hidden)
        out1, _ = self.l1(inputs1, (h0, c0))

        # Only keep an even number of time slices
        if seq_len % 2 != 0:
            out1 = out1[:-1, ...]
            seq_len = seq_len - 1

        # Concatenate two consecutive time steps for feeding in the next
        # layer
        # inputs2 is (seq_len//2, batch_size, 2*num_hidden)
        inputs2 = out1.transpose(0, 1)
        inputs2 = inputs2.reshape(batch_size, seq_len//2, 4*self.num_hidden)
        inputs2 = inputs2.transpose(0, 1)
        seq_len = seq_len//2
        # Forward pass through L2
        out2, _ = self.l2(inputs2, (h0, c0))

        if seq_len % 2 != 0:
            out2 = out2[:-1, ...]
            seq_len = seq_len - 1
        # Concatenate two consecutive time steps for feeding in the next
        # layer
        # inputs2 is (seq_len//2, batch_size, 2*num_hidden)
        inputs3 = out2.transpose(0, 1)
        inputs3 = inputs3.reshape(batch_size, seq_len//2, 4*self.num_hidden)
        inputs3 = inputs3.transpose(0, 1)
        seq_len = seq_len//2
        # Forward pass through L2
        out3, _ = self.l3(inputs3, (h0, c0))

        # out3 is (seq_len3, batch_size, 2*num_hidden)
        return out3


class Model(nn.Module):

    def __init__(self, n_mels: int):
        """
        Args:
            n_mels: int  The number of input mel scales
        """
        super(Model, self).__init__()
        self.listener = Listener(n_mels)

    def forward(self, inputs):
        self.listener(inputs)
