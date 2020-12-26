#!/usr/bin/env python3

# Standard imports
# External imports
import torch
import torch.nn as nn


class Listener(nn.Module):

    def __init__(self,
                 n_mels: int,
                 num_hidden: int):
        """
        Args:
            n_mels (int) : number of input mel scales
            num_hidden (int): number of LSTM cells per layer and per direction
        """
        super(Listener, self).__init__()
        self.num_hidden = num_hidden
        # Pyramidal Bidirectionnal LSTM
        self.l1 = nn.LSTM(n_mels,
                          self.num_hidden,
                          bidirectional=True,
                          batch_first=True)
        self.l2 = nn.LSTM(2 * 2 * self.num_hidden,
                          self.num_hidden,
                          bidirectional=True,
                          batch_first=True)
        self.l3 = nn.LSTM(2 * 2 * self.num_hidden,
                          self.num_hidden,
                          bidirectional=True,
                          batch_first=True)

    def forward(self, inputs1):
        """
        Args:
            inputs : (batch, num_mels, time)
        """
        # Get the device to create tensors on the same
        device = inputs1.device

        # Transpose the inputs to be (batch, time, num_mels)
        inputs1 = inputs1.permute(0, 2, 1)
        batch_size, seq_len, num_mels = inputs1.shape

        # The initial hidden and cell states
        h0 = torch.zeros(2, batch_size, self.num_hidden,
                         device=device,
                         requires_grad=False)
        c0 = torch.zeros(2, batch_size, self.num_hidden,
                         device=device,
                         requires_grad=False)

        # out1 is (batch_size, seq_len, 2*num_hidden)
        out1, _ = self.l1(inputs1, (h0, c0))

        # Only keep an even number of time slices
        if seq_len % 2 != 0:
            out1 = out1[:, :-1, :]
            seq_len = seq_len - 1

        # Concatenate two consecutive time steps for feeding in the next
        # layer
        # inputs2 is (batch_size, seq_len//2, 2*num_hidden)
        inputs2 = out1.reshape(batch_size, seq_len//2, 4*self.num_hidden)
        seq_len = seq_len//2
        # Forward pass through L2
        out2, _ = self.l2(inputs2, (h0, c0))

        if seq_len % 2 != 0:
            out2 = out2[:, :-1, :]
            seq_len = seq_len - 1
        # Concatenate two consecutive time steps for feeding in the next
        # layer
        # inputs2 is (batch_size, seq_len//2, 2*num_hidden)
        inputs3 = out2.reshape(batch_size, seq_len//2, 4*self.num_hidden)
        seq_len = seq_len//2
        # Forward pass through L2
        out3, _ = self.l3(inputs3, (h0, c0))

        # out3 is (batch_size, seq_len3, 2*num_hidden)
        #TODO: question, should we again concatenate 2 successive time steps ?
        # the paper says the time scale is downscaled by 2**3
        # but its Fig1 seems only to imply a downsclae by 2**2 except if 
        # we also concatenate 2 successive time step from the output of 
        # the last layer ?
        return out3


class AttendAndSpell(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 num_inputs: int,
                 num_hidden: int):
        """
        Args:
            vocab_size: the number of output characters
            num_inputs: the number of inputs from the previous layer
            num_hidden (int) : The number of LSTM cells per layer
        """
        super(AttendAndSpell, self).__init__()
        self.num_inputs = num_inputs
        self.vocab_size = vocab_size

        # The first LSTM layer
        self.L1 = nn.LSTM(num_inputs,
                          num_hidden,
                          bidirectional=False)
        # 
        #TODO WIP !
        # see also https://arxiv.org/pdf/1506.07503.pdf
        #          https://www.aclweb.org/anthology/D15-1166.pdf
        #          https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch/blob/master/src/asr.py


    def forward(self, inputs, gt_outputs):
        """
        inputs: (seq_len, batch_size, num_features)
        gt_outputs : (batch_size, seq_len)
        outputs : (seq_len, batch_size, vocab_size)
        """

        # We must manually perform the loop in time
        # since we must be computing the attention mask at every
        # time step
        seq_len, batch_size, num_features = inputs.shape




class Model(nn.Module):

    def __init__(self,
                 n_mels: int,
                 vocab_size: int,
                 num_hidden_listen: int,
                 num_hidden_spell: int):
        """
        Args:
            n_mels (int)  The number of input mel scales
            vocab_size (int) The size of the vocabulary
            num_hidden_listen(int): The number of LSTM cells per layer, per
                                    direction for the listen module
            num_hidden_spell(int): The number of LSTM cells per layer for
                                   the spell module
        """
        super(Model, self).__init__()
        self.listener = Listener(n_mels, num_hidden_listen)
        self.attend_and_spell = AttendAndSpell(vocab_size,
                                               2*num_hidden_listen,
                                               num_hidden_spell)

    def forward(self, inputs, gt_outputs):
        """
        inputs: (batch_size, num_mels, time)
        gt_outputs: (batch_size, time)
        """
        out_listen = self.listener(inputs)
        out_attend_and_spell = self.attend_and_spell(out_listen, gt_outputs)
