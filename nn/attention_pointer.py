import math

import torch
import torch.nn as nn

from utils import config


class PointerNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(PointerNetwork, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pointer = Attention(embedding_dim)

    def forward(self, obs, job_types, skip_mask, state=None):
        output = self.transformer_encoder(obs[~skip_mask, :, :].permute(1, 0, 2))
        query = torch.mean(output, 0)
        _, idxs_logits = self.pointer(output, query)
        mask = job_types[~skip_mask] == -1
        idxs_logits[mask] = -torch.inf
        return idxs_logits


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10, device=config.device):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim, bias=False)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1, bias=False)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()
        self.V = nn.Parameter(torch.rand(dim, dtype=torch.float)-0.5)

    def forward(self, ref, query):
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
        v_view = self.V.unsqueeze(0).expand(
            expanded_q.size(0), len(self.V)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits
