#!/usr/bin/env python3

# Standard imports
# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PackedCELoss(nn.Module):

    def __init__(self):
        super(PackedCELoss, self).__init__()
        self.base_loss = nn.CrossEntropyLoss()

    def forward(self, packed_predictions, packed_targets):
        """
        Args:

        """
        # The packed_targets virtualy "contain"
        #  <sos> c1 c2 c3 c4 .... <eos>
        # The predictions are expected to predict
        # c1 c2 c3 ... <eos>

        # Therefore, the targets to predict are in the slice
        #  targets[i, 1:li]
        # Which are to be compared with the probabilities in
        #  predictions[i, :li-1, ...]
        # predictions, lens_predictions = pad_packed_sequence(packed_predictions,
        #                                                     batch_first=True)
        targets, lens_targets = pad_packed_sequence(packed_targets,
                                                    batch_first=True)
        # Remove the <sos> from the targets
        targets = targets[:, 1:]
        lens_targets -= 1
        # Repack it
        packed_targets = pack_padded_sequence(targets,
                                              lengths=lens_targets,
                                              enforce_sorted=False,
                                              batch_first=True)

        return self.base_loss(packed_predictions.data, packed_targets.data)


class EncoderListener(nn.Module):

    def __init__(self,
                 n_mels: int,
                 num_hidden: int):
        """
        Args:
            n_mels (int) : number of input mel scales
            num_hidden (int): number of LSTM cells per layer and per direction
        """
        super(EncoderListener, self).__init__()
        self.num_hidden = num_hidden
        self.batch_first = True
        # Pyramidal Bidirectionnal LSTM
        self.l1 = nn.LSTM(n_mels,
                          self.num_hidden,
                          bidirectional=True,
                          batch_first=self.batch_first)
        self.l2 = nn.LSTM(2 * 2 * self.num_hidden,
                          self.num_hidden,
                          bidirectional=True,
                          batch_first=self.batch_first)
        self.l3 = nn.LSTM(2 * 2 * self.num_hidden,
                          self.num_hidden,
                          bidirectional=True,
                          batch_first=self.batch_first)

    def downscale_pack(self, packed_sequence):
        """
        Concatenate the features of two consecutive time steps
        Args:
            packed_sequence (PackedSequence) batch_first
        """
        unpacked, lens = pad_packed_sequence(packed_sequence,
                                             batch_first=self.batch_first)
        # unpackaged is (batch_size, seq_len, num_features)
        if self.batch_first:
            batch_size, seq_len, num_features = unpacked.shape

            if seq_len % 2 != 0:
                unpacked = unpacked[:, :-1, :]
                lens[lens == seq_len] = seq_len - 1
        else:
            raise NotImplementedError("Cannot process non batch first tensors")
        lens = lens // 2
        assert(self.batch_first)
        unpacked = unpacked.reshape(batch_size, seq_len//2, 2*num_features)
        # repack the sequences
        return pack_padded_sequence(unpacked,
                                   lengths=lens,
                                   batch_first=self.batch_first)

    def forward(self, inputs1):
        """
        Args:
            inputs1 : PackedSequence (batch, time, n_mels)
        """
        # Note, using packed sequence is computationnally less
        # expensive but it also makes handling of bidirectionnaly LSTM
        # easier. If we were to use the unpacked sequences, the LSTM for the
        # backward direction would first eat the pad tokens

        # Forward pass through L1
        # out1 is (batch_size, seq_len, 2*num_hidden)
        packedout1, _ = self.l1(inputs1)

        # Forward pass through L2
        inputs2 = self.downscale_pack(packedout1)
        packedout2, _ = self.l2(inputs2)

        # Forward pass through L2
        inputs3 = self.downscale_pack(packedout2)
        packedout3, _ = self.l3(inputs3)

        # out3 is (batch_size, seq_len3, 2*num_hidden)
        # the paper says the time scale is downscaled by 2**3
        # but its Fig1 seems only to imply a downsclae by 2**2 except if
        # we also concatenate 2 successive time step from the output of
        # the last layer which is what we do below
        return self.downscale_pack(packedout3)


class Decoder(nn.Module):
    """
    Simple decoder with the Seq2Seq like architecture
    see [Sutskever et al(2014)]
    """

    def __init__(self,
                 vocab_size: int,
                 num_inputs: int,
                 dim_embed: int,
                 num_hidden: int):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_inputs = num_inputs
        self.dim_embed = dim_embed
        self.num_hidden = num_hidden

        # An embedding layer for processing the grounth truth characters
        # Note: can be initialized from one-hot
        self.embed = nn.Embedding(vocab_size, dim_embed)

        # A linear layer for projecting the encoder features to the 
        # initial hidden state of the LSTM
        self.encoder_to_hidden = nn.Linear(num_inputs, self.num_hidden)

        # The decoder RNN
        self.rnn = nn.LSTM(dim_embed,
                           self.num_hidden,
                           batch_first=True)

        # The linear linear before the softmax
        self.charlin = nn.Linear(self.num_hidden, self.vocab_size)

    def forward(self, packed_features, packed_outputs):
        """
        encoder_features: (seq_len, batch_size, num_encoder_features)
        gt_outputs : (batch_size, seq_len)
        outputs : (seq_len, batch_size, vocab_size)
        """

        unpacked_targets, lens_targets = pad_packed_sequence(packed_outputs,
                                                            batch_first=True)
        # Remove the <eos>
        lens_targets -= 1

        # Forward propagate through the embedding layer
        # embeddings is (batch_size, Ty, dim_embedding)
        embeddings = self.embed(unpacked_targets)

        # Pack the result
        packed_embedded = pack_padded_sequence(embeddings,
                                               lengths=lens_targets,
                                               enforce_sorted=False,
                                               batch_first=True
                                              )

        # Use the last encoder_features to compute the initial hidden
        # state of the decoder
        unpacked_features, lens_features = pad_packed_sequence(packed_features,
                                                              batch_first=True)
        # unpacked_features is (batch_size, seq_len, num_features)
        encoder_features = torch.stack([unpacked_features[i, ti-1, :] for i, ti in enumerate(lens_features)], dim=0)

        h0 = self.encoder_to_hidden(encoder_features).unsqueeze(dim=0)
        c0 = torch.zeros_like(h0)
        packedout_rnn, _ = self.rnn(packed_embedded, (h0, c0))

        unpacked_out, lens_out = pad_packed_sequence(packedout_rnn,
                                                     batch_first=True)

        # Compute the logits over the vocabulary
        # outchar is (batch_size, seq_len, vocab_size)
        outchar = self.charlin(unpacked_out)

        return pack_padded_sequence(outchar,
                                    batch_first=True,
                                    enforce_sorted=False,
                                    lengths=lens_targets)

class AttendAndSpell(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 num_inputs: int,
                 dim_embed: int,
                 num_hidden: int):
        """
        Args:
            vocab_size: the number of output characters
            num_inputs: the number of inputs from the previous layer
            dim_embed (int) : the size of the input embedding
            num_hidden (int) : The number of LSTM cells per layer
        """
        super(AttendAndSpell, self).__init__()
        self.num_inputs = num_inputs
        self.vocab_size = vocab_size

        # An embedding layer for processing the grounth truth characters
        # Note: can be initialized from one-hot
        self.embed = nn.Embedding(vocab_size, dim_embed)

        # The first LSTM layer
        self.L1 = nn.LSTM(num_inputs,
                          num_hidden,
                          bidirectional=False)
        # 
        #TODO WIP !
        # see also https://arxiv.org/pdf/1506.07503.pdf
        #          https://www.aclweb.org/anthology/D15-1166.pdf
        #          https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch/blob/master/src/asr.py


    def forward(self, encoder_features, gt_outputs):
        """
        encoder_features: (seq_len, batch_size, num_features)
        gt_outputs : (batch_size, seq_len)
        outputs : (seq_len, batch_size, vocab_size)
        """

        # We must manually perform the loop in time
        # since we must be computing the attention mask at every
        # time step
        seq_len, batch_size, num_features = inputs.shape

        # During training, we use the ground truth labels for input
        # forcing
        char_embeddings = self.embed(gt_outputs)

        #TODO


class Model(nn.Module):

    def __init__(self,
                 n_mels: int,
                 vocab_size: int,
                 num_hidden_listen: int,
                 dim_embed: int,
                 num_hidden_spell: int):
        """
        Args:
            n_mels (int)  The number of input mel scales
            vocab_size (int) The size of the vocabulary
            num_hidden_listen(int): The number of LSTM cells per layer, per
                                    direction for the listen module
            dim_embed(int): The size of the embedding for the Spell module
            num_hidden_spell(int): The number of LSTM cells per layer for
                                   the spell module
        """
        super(Model, self).__init__()
        self.listener = EncoderListener(n_mels, num_hidden_listen)
        self.decoder = Decoder(vocab_size,
                               4*num_hidden_listen,
                               dim_embed,
                               num_hidden_spell)
        # self.attend_and_spell = AttendAndSpell(vocab_size,
        #                                        2*num_hidden_listen,
        #                                        dim_embed,
        #                                        num_hidden_spell)

    def forward(self, inputs, gt_outputs=None):
        """
        inputs: (batch_size, num_mels, time)
        gt_outputs: (batch_size, time)
        """

        # We assume that
        #  forward(inputs, gt_outputs) is the propagation during learning
        #  forward(inputs)             is the propagation during inference
        if gt_outputs is not None:
            # Propagation for learning
            # or for evaluating the probability of the ground truth
            # transcriptions
            out_listen = self.listener(inputs)
            out_decoder = self.decoder(out_listen, gt_outputs)
            # out_attend_and_spell = self.attend_and_spell(out_listen, gt_outputs)
            return out_decoder
        else:
            # Inference
            pass
