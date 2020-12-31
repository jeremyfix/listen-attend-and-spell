#!/usr/bin/env python3

# Standard imports
# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
# Local imports
import data

class Encoder(nn.Module):

    def __init__(self,
                 n_mels: int,
                 num_hidden: int) -> None:
        """
        Args:
            n_mels (int) : number of input mel scales
            num_hidden (int): number of LSTM cells per layer and per direction
        """
        super(Encoder, self).__init__()
        self.num_hidden = num_hidden
        self.batch_first = True
        self.l1 = nn.LSTM(n_mels,
                          self.num_hidden,
                          bidirectional=False,
                          batch_first=self.batch_first)

    def forward(self,
                inputs: PackedSequence) -> PackedSequence:
        """
        Args:
            inputs1 : PackedSequence (batch, time, n_mels)
        Returns:
            packed_features : PackedSequence (batch, time//2**3, 2*num_hidden)
        """
        # Forward pass through the LSTM layer
        # cn is (1, batch_size, num_hidden)
        _, (_, cn) = self.l1(inputs)

        _, batch_size, _ = cn.shape

        return cn.transpose(0, 1).reshape(batch_size, -1)


class Decoder(nn.Module):
    """
    Simple decoder with the Seq2Seq like architecture
    see [Sutskever et al(2014)]
    """

    def __init__(self,
                 charmap: data.CharMap,
                 num_inputs: int,
                 dim_embed: int,
                 num_hidden: int,
                 teacher_forcing: bool) -> None:
        super(Decoder, self).__init__()
        self.charmap = charmap
        self.vocab_size = self.charmap.vocab_size
        self.num_inputs = num_inputs
        self.dim_embed = dim_embed
        self.num_hidden = num_hidden
        self.teacher_forcing = teacher_forcing

        # An embedding layer for processing the grounth truth characters
        # Note: can be initialized from one-hot
        self.embed = nn.Embedding(self.vocab_size, dim_embed)

        # A linear layer for projecting the encoder features to the
        # initial hidden state of the LSTM
        self.encoder_to_cell = nn.Linear(num_inputs, self.num_hidden)

        # The decoder RNN
        self.rnn = nn.LSTM(dim_embed,
                           self.num_hidden,
                           batch_first=True)

        # The linear linear before the softmax
        self.charlin = nn.Linear(self.num_hidden, self.vocab_size)

    def set_forcing(self, forcing):
        self.teacher_forcing = forcing

    def forward(self,
                encoder_features: torch.Tensor,
                packed_gt_outputs: PackedSequence) -> PackedSequence:
        """
        encoder_features: (batch_size, num_encoder_features)
        packed_gt_outputs : (batch_size, seq_len)
        packed_outputs : (batch_size, seq_len, vocab_size)
        """

        batch_size, _ = encoder_features.shape
        device = encoder_features.device

        # c0 is (1, batch_size, num_hidden)
        c0 = self.encoder_to_cell(encoder_features).unsqueeze(dim=0)
        # h0 is (1, batch_size, num_hidden)
        h0 = torch.zeros_like(c0)

        if self.teacher_forcing:
            print("with teacher forcing")
            # We proceed with teacher forcing
            # this might help training but also makes differences
            # between training and inference

            unpacked_targets, lens_targets = pad_packed_sequence(packed_gt_outputs,
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
                                                   batch_first=True)

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
        else:
            print("withOUT teacher forcing")
            # Without teacher forcing, we manually unroll the loop
            # to provide as input the character with the highest
            # probability

            # We start all (for every sample in the batch) the decoders
            # with the sos token
            soschar_token = self.charmap.encode(self.charmap.soschar)
            input_chars = torch.LongTensor([soschar_token] * batch_size).to(device)
            hn_1, cn_1 = h0, c0

            # Unpack the targets to get
            # 1- the maxlength number of steps during which to iterate
            # 2- to extract the probabilities at the right time step
            #    for every sample in the minibatch
            unpacked_targets, lens_targets = pad_packed_sequence(packed_gt_outputs,
                                                                 batch_first=True)

            max_length = lens_targets.max().item()
            outchar = None
            for ti in range(max_length-1):
                # Compute the input character embeddings
                embeddings = self.embed(input_chars)

                # Forward propagate one step through the RNN
                out_rnn, (hn, cn) = self.rnn(embeddings,
                                             (hn_1, cn_1))
                out_rnn = out_rnn.squeeze()

                # Compute the probability distribution over the characters
                # outchar_n is (batch_size, vocab_size)
                outchar_n = self.charlin(out_rnn)
                if outchar is None:
                    outchar = outchar_n.unsqueeze(dim=0)
                else:
                    outchar = torch.vstack([outchar,
                                            outchar_n.unsqueeze(dim=0)])

                # Loop
                # 1- update the hidden states of the previous step
                hn_1, cn_1 = hn, cn
                # 2- update the input chars and their embeddings
                input_chars = outchar_n.argmax(dim=1).unsqueeze(dim=1)


            # At the end of the loop, outchar is (seq_len, batch_size, vocab_size)
            # we permute it to be (batch_size, seq_len, vocab_size)
            outchar = outchar.permute(1, 0, 2)

            # Compute the logits over the vocabulary
            # outchar is (batch_size, seq_len, vocab_size)
            return pack_padded_sequence(outchar,
                                        batch_first=True,
                                        enforce_sorted=False,
                                        lengths=lens_targets-1)


    def decode(self,
               beamwidth: int,
               maxlength: int,
               encoder_features: PackedSequence,
               charmap: data.CharMap,
               gt_transcript: PackedSequence = None) -> PackedSequence:
        """
        Note: cannot handle more than one sample at a time

        Performs beam search decode of the provided spectrogram
        Args:
            beamwidth(int): The number of alternatives to consider
            maxlength(int): The maximal length of the sequence if no <eos> is
            predicted
            encoder_features(torch.Tensor): The output of the encoder
                                           (batch, num_features)
            charmap
            gt_transcripts
        Returns:
            packed_outputs(PackedSequence): The most likely decoded sequences
                                            (batch, maxlength)
        """

        if gt_transcript is not None:
            raise NotImplementedError("Evaluating the probability of a transcript is not yet implemented")

        batch_size, _ = encoder_features.shape
        if batch_size != 1:
            raise NotImplementedError("Cannot handle batch size larger than 1")

        # c0 is (1, batch_size, num_hidden)
        c0 = self.encoder_to_cell(encoder_features).unsqueeze(dim=0)
        # h0 is (1, batch_size, num_hidden)
        h0 = torch.zeros_like(c0) #self.encoder_to_hidden(encoder_features).unsqueeze(dim=0)

        # We now need to iterate manually over the time steps to
        # perform the decoding since we must be feeding in the characters
        # we decoded

        # Collection holding :
        # - the possible alternatives,
        # - their log probabilities
        # - the hidden and cell states they had
        # We need all these to go on expanding the tree for decoding
        sequences = [[0.0, charmap.encode(charmap.soschar), (h0, c0)]]

        for ti in range(maxlength):
            # Forward propagate through the LSTM for every alternative
            # and compute the possible expansions for every path
            expansions = []
            hidden_states = []
            for its, (prob, seq, (hn_1, cn_1)) in enumerate(sequences):

                # Compute the embeddings of the input chars
                input_char = torch.LongTensor([[seq[-1]]]).to(encoder_features.device)
                embeddings = self.embed(input_char)
                packed_embedded = pack_padded_sequence(embeddings,
                                                       lengths=[1]*batch_size,
                                                       batch_first=True)

                packedout_rnn, (hn, cn) = self.rnn(packed_embedded,
                                                   (hn_1, cn_1))
                unpacked_out, _ = pad_packed_sequence(packedout_rnn,
                                                      batch_first=True)
                outchar = self.charlin(unpacked_out).squeeze()

                # Compute the log probabilities of the next characters
                # charlogprobs is (vocab_size, )
                charlogprobs = F.log_softmax(outchar, dim=0).squeeze()

                # Store all the possible expansions
                for ci, lpc in enumerate(charlogprobs):
                    expansions.append((its, ci, prob+lpc.item()))
                hidden_states.append((hn, cn))

            # Sort the expansions by their conditional probabilities
            # first is better
            expansions = sorted(expansions, key=lambda itcp: itcp[2],
                                reverse=True)
            expansions_to_keep = expansions[:beamwidth]

            # And keep only the most likely
            # by updating "sequences"
            newsequences = []
            for its, ci, newlogprob in expansions_to_keep:
                newprob = newlogprob
                newseq = sequences[its][1] + [ci]
                newhidden = hidden_states[its]
                newsequences.append([newprob, newseq, newhidden])
            sequences = newsequences
        # Return the candidates with their logprobabilities
        return [ (p, charmap.decode(s)) for p, s, _ in sequences ]



class EncoderListener(nn.Module):

    def __init__(self,
                 n_mels: int,
                 num_hidden: int) -> None:
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

    def downscale_pack(self,
                       packed_sequence: PackedSequence) -> PackedSequence:
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

    def forward(self,
                inputs1: PackedSequence) -> PackedSequence:
        """
        Args:
            inputs1 : PackedSequence (batch, time, n_mels)
        Returns:
            packed_features : PackedSequence (batch, time//2**3, 2*num_hidden)
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

class Seq2Seq(nn.Module):

    def __init__(self,
                 n_mels: int,
                 charmap,
                 num_hidden_encoder: int,
                 dim_embed: int,
                 num_hidden_decoder: int,
                 teacher_forcing: bool) -> None:
        """
        Args:
            n_mels (int)  The number of input mel scales
            charmap
            num_hidden_encoder(int): The number of LSTM cells for the encoder
            dim_embed(int): The size of the embedding for the Spell module
            num_hidden_deocder(int): The number of LSTM cells for
                                     the decoder
            teacher_forcing(bool) : whether to use teacher forcing
        """
        super(Seq2Seq, self).__init__()
        self.charmap = charmap
        self.encoder = Encoder(n_mels,
                               num_hidden_encoder)
        self.decoder = Decoder(charmap,
                               num_hidden_encoder,
                               dim_embed,
                               num_hidden_decoder,
                               teacher_forcing)

    def set_forcing(self, forcing):
        self.decoder.set_forcing(forcing)

    def forward(self,
                inputs: PackedSequence,
                gt_outputs: PackedSequence) -> PackedSequence:
        """
        Args:
            inputs(PackedSequence): (batch_size, num_mels, time)
            gt_outputs(PackedSequence): (batch_size, time)
        """

        out_encoder = self.encoder(inputs)
        out_decoder = self.decoder(out_encoder, gt_outputs)
        return out_decoder

    def decode(self,
               beamwidth: int,
               maxlength: int,
               inputs: PackedSequence,
               transcript: PackedSequence=None) -> PackedSequence:
        """
        Performs beam search decode of the provided spectrogram
        Args:
            beamwidth(int): The number of alternatives to consider
            maxlength(int): The maximal length of the sequence if no <eos> is
                             predicted
            inputs(torch.Tensor): The input spectrograms
            charmap
            transcript (PackedSequence): optional, used for teacher forcing
                                         decoding (for debug)
        Returns:
            out_decoder(torch.Tensor): The decoded sequences
        """
        with torch.no_grad():
            # Forward propagated through the encoder
            out_encoder = self.encoder(inputs)
            # Performing beam search on the decoder
            packed_out_decoder = self.decoder.decode(beamwidth,
                                                     maxlength,
                                                     out_encoder,
                                                     self.charmap,
                                                     transcript)
            return packed_out_decoder


class Model(nn.Module):

    def __init__(self,
                 n_mels: int,
                 charmap,
                 num_hidden_listen: int,
                 dim_embed: int,
                 num_hidden_spell: int,
                 teacher_forcing: bool) -> None:
        """
        Args:
            n_mels (int)  The number of input mel scales
            charmap
            num_hidden_listen(int): The number of LSTM cells per layer, per
                                    direction for the listen module
            dim_embed(int): The size of the embedding for the Spell module
            num_hidden_spell(int): The number of LSTM cells per layer for
                                   the spell module
            teacher_forcing(bool) : whether to use teacher forcing
        """
        super(Model, self).__init__()
        self.charmap = charmap
        self.listener = EncoderListener(n_mels, num_hidden_listen)
        self.decoder = Decoder(charmap,
                               4*num_hidden_listen,
                               dim_embed,
                               num_hidden_spell,
                               teacher_forcing)
        # self.attend_and_spell = AttendAndSpell(vocab_size,
        #                                        2*num_hidden_listen,
        #                                        dim_embed,
        #                                        num_hidden_spell)

    def set_forcing(self, forcing):
        self.decoder.set_forcing(forcing)

    def forward(self,
                inputs: torch.Tensor,
                gt_outputs: torch.Tensor) -> PackedSequence:
        """
        Args:
            inputs(torch.Tensor): (batch_size, num_mels, time)
            gt_outputs(torch.Tensor): (batch_size, time)
        """

        # Propagation for learning
        # or for evaluating the probability of the ground truth
        # transcriptions
        out_listen = self.listener(inputs)
        out_decoder = self.decoder(out_listen, gt_outputs)
        # out_attend_and_spell = self.attend_and_spell(out_listen, gt_outputs)
        return out_decoder

    def decode(self,
               beamwidth: int,
               maxlength: int,
               inputs: PackedSequence,
               transcript: PackedSequence=None) -> PackedSequence:
        """
        Performs beam search decode of the provided spectrogram
        Args:
            beamwidth(int): The number of alternatives to consider
            maxlength(int): The maximal length of the sequence if no <eos> is
                             predicted
            inputs(torch.Tensor): The input spectrograms
            charmap
            transcript (PackedSequence): optional, used for teacher forcing
                                         decoding (for debug)
        Returns:
            out_decoder(torch.Tensor): The decoded sequences
        """
        with torch.no_grad():
            # Forward propagated through the encoder
            packed_out_listen = self.listener(inputs)
            # Performing beam search on the decoder
            packed_out_decoder = self.decoder.decode(beamwidth,
                                                     maxlength,
                                                     packed_out_listen,
                                                     self.charmap,
                                                     transcript)
            return packed_out_decoder
