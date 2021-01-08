#!/usr/bin/env python3

# Standard imports
import collections
import math
from typing import List, Tuple
import tqdm
# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
# Local imports
import data

class CTCModel(nn.Module):

    def __init__(self,
                 charmap: data.CharMap,
                 n_mels: int,
                 num_hidden: int,
                 num_layers: int,
                 cell_type: str):
        """
        Args:
            charmap (data.Charmap) : the character/int map
            n_mels (int) : number of input mel scales
            num_hidden (int): number of LSTM cells per layer and per direction
            num_layers (int) : number of stacked RNN layers
            cell_type(str) either "GRU" or "LSTM"
        """
        super(CTCModel, self).__init__()
        self.batch_first = True
        self.charmap = charmap
        self.n_mels = n_mels
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=1,
        #               out_channels=32,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        # )

        if cell_type not in ["GRU", "LSTM"]:
            raise NotImplementedError(f"Unrecognized cell type {cell_type}")

        cell_builder = getattr(nn, cell_type)
        self.rnn = cell_builder(n_mels,
                                self.num_hidden,
                                num_layers=num_layers,
                                batch_first=self.batch_first,
                                bidirectional=True)
        self.charlin = nn.Sequential(
            nn.Linear(2*self.num_hidden , charmap.vocab_size + 1)  # add the blank
            # nn.Linear(2*self.num_hidden, 128), nn.ReLU(),
            # nn.Linear(128, 128), nn.ReLU(),
            # nn.Linear(128 , charmap.vocab_size + 1)  # add the blank
        )

    def forward(self,
                inputs: PackedSequence) -> PackedSequence:

        # # # Go through the CNN
        # unpacked_inputs, lens_inputs = pad_packed_sequence(inputs,
        #                                                    batch_first=True)
        # # unpacked_inputs is (batch, seq_len, num_mels)
        # # unsqueeze to make input channel_size = 1
        # out_cnn = self.cnn(unpacked_inputs.unsqueeze(dim=1))
        # # out_cnn is (B, C, seq_len, num_mels)
        # # make it (B, seq_len, num_features=C*num_mels)
        # batch_size = unpacked_inputs.shape[0]
        # seq_len = unpacked_inputs.shape[1]
        # out_cnn = out_cnn.transpose(1, 2).reshape(batch_size, seq_len, -1)

        # # pack the cnn output, given there is no downsampling in the cnn
        # # the lenghts are the same
        # inputs = pack_padded_sequence(out_cnn, lengths=lens_inputs,
        #                               batch_first=True)

        # Go through the RNN
        packed_outrnn, _ = self.rnn(inputs)  # batch, seq, num_hidden

        unpacked_outrnn, lens_outrnn = pad_packed_sequence(packed_outrnn,
                                                           batch_first=True)

        # Go through the next char predictor
        out_lin = self.charlin(unpacked_outrnn)

        # Repack the result
        outputs = pack_padded_sequence(out_lin,
                                       lengths=lens_outrnn,
                                       batch_first=True)
        return outputs

    def decode(self,
               inputs: PackedSequence) -> List[Tuple[float, str]]:
        """
        Greedy decoder.

        Args:
            inputs (PackedSequence) : the input spectrogram

        Returns:
            list of pairs of the negative log likelihood of the sequence
                          with the corresponding sequence
        """
        with torch.no_grad():
            outputs = self.forward(inputs)  # packed batch, seq, num_char
            unpacked_outputs, lens_outputs = pad_packed_sequence(outputs,
                                                                 batch_first=True)
            batch_size, seq_len, num_char = unpacked_outputs.shape
            if batch_size != 1:
                raise NotImplementedError("Can decode only one batch at a time")

            unpacked_outputs = unpacked_outputs.squeeze(dim=0)  # seq, vocab_size
            outputs = unpacked_outputs.log_softmax(dim=1)
            top_values, top_indices = outputs.topk(k=1, dim=1)

            prob = -top_values.sum().item()
            seq = [ci for ci in top_indices if ci != self.charmap.vocab_size]

            # Remove the repetitions
            if len(seq) != 0:
                last_char = seq[-1]
                seq = [c1 for c1, c2 in zip(seq[:-1], seq[1:]) if c1 != c2]
                seq.append(last_char)

            # Decode the list of integers
            seq = self.charmap.decode(seq)

            # Drop out every blank
            return [(prob, seq)]

    def beam_decode(self,
                    inputs: PackedSequence,
                    beam_size: int,
                    blank_id: int):
        """
        Performs inference for the given output probabilities.
        Assuming a single sample is given.
        Adapted from :
            https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0

        Arguments:
            inputs (PackedSequence) : the input spectrogram
            beam_size (int): Size of the beam to use during inference.
            blank (int): Index of the CTC blank label.
        Returns the output label sequence and the corresponding negative
        log-likelihood estimated by the decoder.
        """
        NEG_INF = -float("inf")
        def make_new_beam():
            fn = lambda : (NEG_INF, NEG_INF)
            return collections.defaultdict(fn)

        def logsumexp(*args):
            """
            Stable log sum exp.
            """
            if all(a == NEG_INF for a in args):
                return NEG_INF
            a_max = max(args)
            lsp = math.log(sum(math.exp(a - a_max) for a in args))
            return a_max + lsp

        with torch.no_grad():
            outputs = self.forward(inputs)  # packed batch, seq, num_char
            unpacked_outputs, lens_outputs = pad_packed_sequence(outputs,
                                                                 batch_first=True)
            batch_size, T, S = unpacked_outputs.shape
            if batch_size != 1:
                raise NotImplementedError("Can decode only one batch at a time")

            unpacked_outputs = unpacked_outputs.squeeze(dim=0)  # seq, vocab_size
            probs = unpacked_outputs.log_softmax(dim=1)

            # Elements in the beam are (prefix, (p_blank, p_no_blank))
            # Initialize the beam with the empty sequence, a probability of
            # 1 for ending in blank and zero for ending in non-blank
            # (in log space).
            beam = [(tuple(), (0.0, NEG_INF))]

            for t in tqdm.tqdm(range(T)):  # Loop over time

                # A default dictionary to store the next step candidates.
                next_beam = make_new_beam()

                for s in range(S): # Loop over vocab
                    p = probs[t, s]

                    # The variables p_b and p_nb are respectively the
                    # probabilities for the prefix given that it ends in a
                    # blank and does not end in a blank at this time step.
                    for prefix, (p_b, p_nb) in beam:  # Loop over beam

                        # If we propose a blank the prefix doesn't change.
                        # Only the probability of ending in blank gets updated.
                        if s == blank_id:
                            n_p_b, n_p_nb = next_beam[prefix]
                            n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                            next_beam[prefix] = (n_p_b, n_p_nb)
                            continue

                        # Extend the prefix by the new character s and add it to
                        # the beam. Only the probability of not ending in blank
                        # gets updated.
                        end_t = prefix[-1] if prefix else None
                        n_prefix = prefix + (s,)
                        n_p_b, n_p_nb = next_beam[n_prefix]
                        if s != end_t:
                            n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                        else:
                            # We don't include the previous probability of not ending
                            # in blank (p_nb) if s is repeated at the end. The CTC
                            # algorithm merges characters not separated by a blank.
                            n_p_nb = logsumexp(n_p_nb, p_b + p)

                        # *NB* this would be a good place to include an LM score.
                        next_beam[n_prefix] = (n_p_b, n_p_nb)

                        # If s is repeated at the end we also update the unchanged
                        # prefix. This is the merging case.
                        if s == end_t:
                            n_p_b, n_p_nb = next_beam[prefix]
                            n_p_nb = logsumexp(n_p_nb, p_nb + p)
                            next_beam[prefix] = (n_p_b, n_p_nb)

                # Sort and trim the beam before moving on to the
                # next time-step.
                beam = sorted(next_beam.items(),
                              key=lambda x: logsumexp(*x[1]),
                              reverse=True)
                beam = beam[:beam_size]

            best = beam[0]

            # Decode the list of integers
            seq = self.charmap.decode(best[0])

            return [(-logsumexp(*best[1]), seq)]


class Encoder(nn.Module):

    def __init__(self,
                 n_mels: int,
                 num_hidden: int,
                 num_layers: int) -> None:
        """
        Args:
            n_mels (int) : number of input mel scales
            num_hidden (int): number of LSTM cells per layer and per direction
            num_layers (int) : number of stacked RNN layers
        """
        super(Encoder, self).__init__()
        self.num_hidden = num_hidden
        self.batch_first = True
        self.num_layers = num_layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_mels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            *([nn.Conv1d(32, 32, 3, 1, 1), nn.ReLU()]*3)
        )
        self.l1 = nn.GRU(32,
                         self.num_hidden,
                         num_layers=num_layers,
                         batch_first=self.batch_first)
        # self.l1 = nn.LSTM(n_mels,
        #                   self.num_hidden,
        #                   bidirectional=False,
        #                   batch_first=self.batch_first)

    def forward(self,
                inputs: PackedSequence) -> PackedSequence:
        """
        Args:
            inputs1 : PackedSequence (batch, time, n_mels)
        Returns:
            packed_features : PackedSequence (batch, time//2**3, 2*num_hidden)
        """
        # # Forward pass through the LSTM layer
        # # cn is (1, batch_size, num_hidden)
        # outputs, (hn, cn) = self.l1(inputs)
        # # _, batch_size, _ = cn.shape
        # # return cn.transpose(0, 1).reshape(batch_size, -1)
        # return (hn, cn)

        # Unpack the input : (batch, time, n_mels)

        unpacked_inputs, lens_inputs = pad_packed_sequence(inputs,
                                                           batch_first=True)
        # Transpose time and n_mels dimensions to be (batch, chan_in, time)
        unpacked_inputs = unpacked_inputs.transpose(1, 2)
        out_cnn = self.cnn(unpacked_inputs)  # batch, 32, seq_len
        out_cnn = out_cnn.transpose(1, 2)  # batch, seq_len, 32
        inputs = pack_padded_sequence(out_cnn,
                                      lengths=lens_inputs,
                                      batch_first=True)

        # hn is (1, batch_size, num_hidden)
        _, hn = self.l1(inputs)
        return hn


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
                 num_layers: int,
                 teacher_forcing: bool) -> None:
        super(Decoder, self).__init__()
        self.charmap = charmap
        self.vocab_size = self.charmap.vocab_size
        self.num_inputs = num_inputs
        self.dim_embed = dim_embed
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.teacher_forcing = teacher_forcing

        # An embedding layer for processing the grounth truth characters
        # Note: can be initialized from one-hot
        self.embed = nn.Sequential(
            nn.Embedding(self.vocab_size, dim_embed),
            nn.ReLU()
        )

        # A linear layer for projecting the encoder features to the
        # initial hidden state of the LSTM
        # self.encoder_to_cell = nn.Linear(num_inputs, self.num_hidden)

        # The decoder RNN
        # self.rnn = nn.LSTM(dim_embed,
        #                    self.num_hidden,
        #                    batch_first=True)
        self.rnn = nn.GRU(dim_embed,
                          self.num_hidden,
                          num_layers=self.num_layers,
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

        # batch_size, _ = encoder_features.shape
        # device = encoder_features.device

        # # c0 is (1, batch_size, num_hidden)
        # c0 = self.encoder_to_cell(encoder_features).unsqueeze(dim=0)
        # # h0 is (1, batch_size, num_hidden)
        # h0 = torch.zeros_like(c0)

        h0 = encoder_features
        device = h0.device
        batch_size = h0.shape[1]


        if self.teacher_forcing:
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

            packedout_rnn, _ = self.rnn(packed_embedded, h0)

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
            # Without teacher forcing, we manually unroll the loop
            # to provide as input the character with the highest
            # probability

            # We start all (for every sample in the batch) the decoders
            # with the sos token
            soschar_token = self.charmap.encode(self.charmap.soschar)
            # input_chars is (batch_size, 1) for seq_len=1
            input_chars = torch.LongTensor([soschar_token] * batch_size).to(device)
            hn_1 = h0

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
                # embeddings is (batch_size, 1, dim_embed)
                embeddings = self.embed(input_chars)

                # Forward propagate one step through the RNN
                # out_rnn is (batch_size, 1, n_hidden)
                out_rnn, hn = self.rnn(embeddings, hn_1)
                out_rnn = out_rnn.squeeze()

                # Compute the probability distribution over the characters
                # outchar_n is (batch_size, vocab_size)
                outchar_n = self.charlin(out_rnn)
                if outchar is None:
                    outchar = outchar_n.unsqueeze(dim=0)
                else:
                    outchar = torch.cat([outchar,
                                         outchar_n.unsqueeze(dim=0)],
                                        dim=0)

                # Loop
                # 1- update the input chars
                input_chars = outchar_n.argmax(dim=1).unsqueeze(dim=1).detach()
                # 2- update the hidden states of the previous step
                hn_1 = hn

            # At the end of the loop, outchar is (seq_len, batch_size, vocab_size)
            # we permute it to be (batch_size, seq_len, vocab_size)
            outchar = outchar.permute(1, 0, 2)

            # outchar is (batch_size, seq_len, vocab_size)
            # it contains one less character than the targets
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

        # batch_size, _ = encoder_features.shape

        # # c0 is (1, batch_size, num_hidden)
        # c0 = self.encoder_to_cell(encoder_features).unsqueeze(dim=0)
        # # h0 is (1, batch_size, num_hidden)
        # h0 = torch.zeros_like(c0) #self.encoder_to_hidden(encoder_features).unsqueeze(dim=0)

        h0 = encoder_features
        device = h0.device
        batch_size = h0.shape[1]

        if batch_size != 1:
            raise NotImplementedError("Cannot handle batch size larger than 1")

        # We now need to iterate manually over the time steps to
        # perform the decoding since we must be feeding in the characters
        # we decoded

        # Collection holding :
        # - the possible alternatives,
        # - their log probabilities
        # - the hidden and cell states they had
        # We need all these to go on expanding the tree for decoding
        sequences = [[0.0, charmap.encode(charmap.soschar), h0]]

        for ti in range(maxlength):
            # Forward propagate through the LSTM for every alternative
            # and compute the possible expansions for every path
            expansions = []
            for its, (prob, seq, hn_1) in enumerate(sequences):

                # Compute the embeddings of the input chars
                input_char = torch.LongTensor([[seq[-1]]]).to(device)
                embeddings = self.embed(input_char)
                packed_embedded = pack_padded_sequence(embeddings,
                                                       lengths=[1]*batch_size,
                                                       batch_first=True)

                packedout_rnn, hn = self.rnn(packed_embedded, hn_1)
                unpacked_out, _ = pad_packed_sequence(packedout_rnn,
                                                      batch_first=True)
                outchar = self.charlin(unpacked_out).squeeze()

                # Compute the log probabilities of the next characters
                # charlogprobs is (vocab_size, )
                charlogprobs = F.log_softmax(outchar, dim=0).squeeze()

                # Store all the possible expansions
                for ci, lpc in enumerate(charlogprobs):
                    expansions.append((its, ci, prob+lpc.item(), hn))

            # Sort the expansions by their conditional probabilities
            # first is better
            expansions = sorted(expansions, key=lambda itcp: itcp[2],
                                reverse=True)
            expansions_to_keep = expansions[:beamwidth]

            # And keep only the most likely
            # by updating "sequences"
            newsequences = []
            for its, ci, newlogprob, newhidden in expansions_to_keep:
                newprob = newlogprob
                newseq = sequences[its][1] + [ci]
                newsequences.append([newprob, newseq, newhidden])
            sequences = newsequences
        # Return the candidates with their logprobabilities
        return [(p, charmap.decode(s)) for p, s, _ in sequences]


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
                 num_layers_encoder: int,
                 dim_embed: int,
                 num_hidden_decoder: int,
                 num_layers_decoder: int,
                 teacher_forcing: bool) -> None:
        """
        Args:
            n_mels (int)  The number of input mel scales
            charmap
            num_hidden_encoder(int): The number of LSTM cells for the encoder
            num_layers_encoder(int): The number of RNN layers for the encoder
            dim_embed(int): The size of the embedding for the Spell module
            num_hidden_deocder(int): The number of LSTM cells for
                                     the decoder
            num_layers_decoder(int) : THe number of RNN layers for the decoder
            teacher_forcing(bool) : whether to use teacher forcing
        """
        super(Seq2Seq, self).__init__()
        self.charmap = charmap
        self.encoder = Encoder(n_mels,
                               num_hidden_encoder,
                               num_layers_encoder)
        self.decoder = Decoder(charmap,
                               num_hidden_encoder,
                               dim_embed,
                               num_hidden_decoder,
                               num_layers_decoder,
                               teacher_forcing)

    def set_forcing(self, forcing):
        self.decoder.set_forcing(forcing)

    def forward(self,
                inputs: PackedSequence,
                gt_outputs: PackedSequence) -> PackedSequence:
        """
        Args:
            inputs(PackedSequence): (batch_size, time, n_mels)
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
            inputs(torch.Tensor): (batch_size, time, n_mels)
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


def ex_ctc():
    # The size of our minibatches
    batch_size = 32
    # The size of our vocabulary (including the blank character)
    vocab_size = 44
    # The class id for the blank token
    blank_id = 43

    max_spectro_length = 50
    min_transcript_length = 10
    max_transcript_length = 30

    # Compute a dummy vector of probabilities over the vocabulary (including the blank)
    # log_probs is here batch_first, i.e. (Batch, Tx, vocab_size)
    log_probs = torch.randn(batch_size, max_spectro_length, vocab_size).log_softmax(dim=1)
    spectro_lengths = torch.randint(low=max_transcript_length,
                                    high=max_spectro_length,
                                    size=(batch_size, ))

    # Compute some dummy transcripts
    # targets is here (batch_size, Ty)
    targets = torch.randint(low=0, high=vocab_size+1,  # include the blank character
                            size=(batch_size, max_transcript_length))
    target_lengths = torch.randint(low=min_transcript_length,
                                   high=max_transcript_length,
                                   size=(batch_size, ))
    loss = torch.nn.CTCLoss(blank=blank_id)

    # The log_probs must be given as (Tx, Batch, vocab_size)
    vloss = loss(log_probs.transpose(0, 1),
                 targets,
                 spectro_lengths,
                 target_lengths)

    print(f"Our dummy loss equals {vloss}")

def ex_pack():
    import random
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

    batch_size = 10
    n_mels = 80
    max_length = 512
    # Given a list of batch_size tensors of variable sizes of shape (Ti, n_mels)
    tensors = [torch.randn(random.randint(1, max_length), n_mels)for i in range(batch_size)]

    # To be packed, the tensors need to be sorted by
    # decreasing length
    tensors = sorted(tensors,
                     key=lambda tensor: tensor.shape[0],
                     reverse=True)
    lengths = [t.shape[0] for t in tensors]

    # We start by padding the sequences to the max_length
    tensors = pad_sequence(tensors, batch_first=True)
    # tensors is (batch_size, T, n_mels)
    # note T is equal to the maximal length of the sequences

    # We can pack the sequence
    packed_data = pack_padded_sequence(tensors, lengths=lengths,
                                      batch_first=True)

    # Latter, we can unpack the sequence
    unpacked_data, lens_data = pad_packed_sequence(packed_data,
                                                  batch_first=True)


if __name__ == '__main__':
    ex_ctc()
    ex_pack()
