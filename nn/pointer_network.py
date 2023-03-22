# MIT License
#
# Copyright (c) 2017 Patrick E.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# modified from https://github.com/pemami4911/neural-combinatorial-rl-pytorch

import math

import numpy as np
import torch
import torch.nn as nn
from torch import count_nonzero
from torch.autograd import Variable
from torch.distributions import Categorical

from utils import config, tools


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, input_dim, hidden_dim, device=config.device):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.device = device
        self.enc_init_state = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = Variable(torch.zeros(hidden_dim), requires_grad=False).to(self.device)

        # enc_init_hx.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))

        enc_init_cx = Variable(torch.zeros(hidden_dim), requires_grad=False).to(self.device)

        # enc_init_cx = nn.Parameter(enc_init_cx)
        # enc_init_cx.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))
        return (enc_init_hx, enc_init_cx)


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10, device=config.device):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        v = torch.FloatTensor(dim).to(device)
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 max_length,
                 tanh_exploration,
                 use_tanh,
                 n_glimpses=0,
                 device=config.device):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.max_length = max_length
        self.device = device

        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration, device=self.device)
        self.glimpse = Attention(hidden_dim, use_tanh=False, device=self.device)
        self.sm = nn.Softmax(-1)

    def apply_mask_to_logits(self, logits, mask):
        maskk = mask.clone()
        logits[maskk] = -np.inf
        return logits, maskk

    def forward(self, decoder_input, hidden, context, job_types, skip_mask):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim]. 
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        def recurrence(x, hidden, logit_mask):

            hx, cx = hidden  # batch_size x hidden_dim

            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # batch_size x hidden_dim

            g_l = hy
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(g_l, context)
                logits, logit_mask = self.apply_mask_to_logits(logits, logit_mask)
                # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] = 
                # [batch_size x h_dim x 1]
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(g_l, context)

            logits, logit_mask = self.apply_mask_to_logits(logits, logit_mask)
            return hy, cy, logits, logit_mask

        mask = job_types[~skip_mask] == -1

        hx, cx, index_logits, mask = recurrence(decoder_input, hidden, mask)
        hidden = (hx, cx)
        return index_logits, hidden


class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq 
    model"""

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 max_decoding_len,
                 n_glimpses=0,
                 tanh_exploration=10,
                 use_tanh=True,
                 device=config.device):
        super(PointerNetwork, self).__init__()

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
            device)

        self.decoder = Decoder(
            embedding_dim,
            hidden_dim,
            max_length=max_decoding_len,
            tanh_exploration=tanh_exploration,
            use_tanh=use_tanh,
            n_glimpses=n_glimpses,
            device=device)

        # Trainable initial hidden states
        dec_in_0 = torch.FloatTensor(embedding_dim).to(device)

        self.decoder_in_0 = nn.Parameter(dec_in_0)
        self.decoder_in_0.data.uniform_(-(1. / math.sqrt(embedding_dim)),
                                        1. / math.sqrt(embedding_dim))

    def forward(self, obs, job_types, skip_mask, state=None):
        """
            Propagate inputs through the network
            inputs: [sourceL x batch_size x embedding_dim]
        """

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        size = count_nonzero(~skip_mask)
        encoder_hx = encoder_hx.unsqueeze(0).repeat(size, 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(size, 1).unsqueeze(0)

        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(obs[:, ~skip_mask, :], (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(size, 1)

        index_logits, dec_hidden_t = self.decoder(decoder_input, dec_init_state, enc_h, job_types, skip_mask)
        return index_logits
