from typing import Any, Union, Dict, Sequence, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from tianshou.data import Batch
from tianshou.utils.net.common import MLP
from torch import nn
from torch.distributions import Categorical
from torch_geometric.nn import to_hetero

from nn import pointer_network, attention_pointer
from utils import config, tools


class Actors(nn.Module):
    def __init__(
            self,
            preprocess_net: nn.Module,
            pointer_type="lstm",
            no_pointer=False,
            no_GNN=False,
            is_toy=False,
    ) -> None:
        super().__init__()
        if config.device == torch.device("cpu") or not config.distributed_HAN:
            self.preprocess = preprocess_net
        else:
            self.preprocess = torch_geometric.nn.DataParallel(preprocess_net)
        self.actors = torch.nn.ModuleList()
        self.no_pointer = no_pointer
        self.no_GNN = no_GNN
        self.is_toy = is_toy
        for _ in range(config.agent_num):
            self.actors.append(
                Actor(config.one_discrete_action_space, config.hidden_dim,
                      pointer_type=pointer_type,
                      device=config.device, no_pointer=self.no_pointer, is_toy=self.is_toy))

    def forward(
            self,
            obs,
            state: Any = None,
            info: Dict[str, Any] = {},
    ):
        if self.no_GNN:
            logits = torch.from_numpy(obs.inp).to(config.device)
            logits = self.preprocess(logits.reshape(-1, config.stacking_dim)).reshape(
                logits.shape[:-1] + (config.hidden_dim,))
        else:
            if config.device == torch.device("cpu") or not config.distributed_HAN:
                inp = torch_geometric.data.Batch.from_data_list(obs.inp.squeeze(-1).tolist()).to(config.device)
                logits, hidden = self.preprocess(inp, state)
            else:
                logits, hidden = self.preprocess(obs.inp.squeeze(-1).tolist())
            logits = tools.refactor_d_state(obs.shape[0], logits["devices"], obs.connections_index, obs.device_num)

        tmp_obs = Batch(job_types=obs.job_types, resource=obs.resource)
        if self.is_toy:
            tmp_obs.first_job = obs.first_job
        batches = []
        for idx, actor in enumerate(self.actors):
            res, state = actor(tmp_obs[:, idx], logits[:, idx], state)
            batches.append(res)
        return Batch.stack(batches, 1), state


class Actor(nn.Module):
    """Simple actor network.

    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
            self,
            action_shape: Sequence[int],
            preprocess_net_output_dim,
            hidden_sizes: Sequence[int] = (),
            pointer_type="lstm",
            device: Union[str, int, torch.device] = "cpu",
            no_pointer=False,
            is_toy=False,
    ) -> None:
        super().__init__()
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        self.no_pointer = no_pointer
        self.is_toy = is_toy
        input_dim = preprocess_net_output_dim
        if pointer_type == "lstm":
            self.pointer = pointer_network.PointerNetwork(config.hidden_dim, config.hidden_dim, 1)
        elif pointer_type == "attn":
            self.pointer = nn.DataParallel(attention_pointer.PointerNetwork(config.hidden_dim))
        else:
            self.pointer = None
        self.last = nn.DataParallel(nn.Sequential(nn.Linear(input_dim, input_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(input_dim, self.output_dim)), dim=0)

    def forward(
            self,
            obs,
            logits,
            state: Any = None,
            info: Dict[str, Any] = {},
    ):
        r"""Mapping: s -> Q(s, \*)."""
        skip_mask = (obs.job_types == -1).all(-1)
        if self.is_toy:
            index_logits = torch.full((np.count_nonzero(skip_mask == False), config.pending_num), -torch.inf).to(
                config.device)
            index_logits[
                range(index_logits.shape[0]), torch.from_numpy(obs.first_job[~skip_mask]).to(config.device)] = 0
        else:
            logits = logits.reshape((obs.shape[0], config.pending_num, -1))
            # pointer
            index_logits = self.pointer(logits,
                                        torch.from_numpy(obs.job_types).to(config.device),
                                        torch.from_numpy(skip_mask).to(config.device))

            # mask
            if self.no_pointer:
                index_logits[index_logits != -torch.inf] = 0

            logits = logits.permute(1, 0, 2).reshape((-1, config.hidden_dim))
        allocation_logits = self.last(logits)
        allocation_logits = allocation_logits.reshape((obs.shape[0], config.pending_num, -1))[
            ~torch.from_numpy(skip_mask).to(config.device)].reshape((-1, config.one_discrete_action_space))
        resource = np.expand_dims(obs.resource[~skip_mask], 1).repeat(config.pending_num, 1)
        resource = resource.reshape(-1, *resource.shape[2:])
        mask = tools.toy_mask_one_logits(obs.job_types[~skip_mask].flatten(), resource)
        mask_tensor = torch.from_numpy(mask).to(config.device)
        allocation_logits[~mask_tensor] = -torch.inf

        allocation_logits = allocation_logits.reshape((-1, config.pending_num, config.one_discrete_action_space))

        all_index_logits = torch.zeros((skip_mask.shape[0], config.pending_num)).to(config.device)
        all_allocation_logits = torch.zeros(
            (skip_mask.shape[0], config.pending_num, config.one_discrete_action_space)).to(config.device)
        all_index_logits[~torch.from_numpy(skip_mask).to(config.device)] = index_logits
        all_allocation_logits[~torch.from_numpy(skip_mask).to(config.device)] = allocation_logits

        return Batch(all_index_logits=all_index_logits, all_allocation_logits=all_allocation_logits), state


class Critics(nn.Module):
    """Simple critic network. Will create an actor operated in discrete \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int last_size: the output dimension of Critic network. Default to 1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
            self,
            preprocess_net: nn.Module,
            hidden_sizes: Sequence[int] = (),
            last_size: int = 1,
            preprocess_net_output_dim: Optional[int] = None,
            device: Union[str, int, torch.device] = "cpu",
            no_pointer=False,
            no_GNN=False,
            is_toy=False,
    ) -> None:
        super().__init__()
        self.device = device
        if config.device == torch.device("cpu") or not config.distributed_HAN:
            self.preprocess = preprocess_net
        else:
            self.preprocess = torch_geometric.nn.DataParallel(preprocess_net)
        self.output_dim = last_size
        self.no_pointer = no_pointer
        self.no_GNN = no_GNN
        self.is_toy = is_toy
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim) * config.pending_num
        self.critics = torch.nn.ModuleList()
        for _ in range(config.agent_num):
            self.critics.append(nn.DataParallel(nn.Sequential(nn.Linear(input_dim, input_dim),
                                                              nn.ReLU(),
                                                              nn.Linear(input_dim, last_size),
                                                              nn.ReLU(), )))

    def forward(
            self, obs, **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        if self.no_GNN:
            logits = torch.from_numpy(obs.inp).to(config.device)
            logits = self.preprocess(logits.reshape(-1, config.stacking_dim)).reshape(
                logits.shape[:-1] + (config.hidden_dim,))
        else:
            if config.device == torch.device("cpu") or not config.distributed_HAN:
                inp = torch_geometric.data.Batch.from_data_list(obs.inp.squeeze(-1).tolist()).to(config.device)
                logits, _ = self.preprocess(inp, state=kwargs.get("state", None))
            else:
                logits, _ = self.preprocess(obs.inp.squeeze(-1).tolist())
            # server_state = torch.reshape(logits["servers"], (obs.shape[0], config.agent_num, -1))
            logits = tools.refactor_d_state(obs.shape[0], logits["devices"], obs.connections_index, obs.device_num)
        batches = []
        for idx, critic in enumerate(self.critics):
            res = critic(logits[:, idx].flatten(start_dim=1))
            batches.append(res)
        return torch.cat(batches, 1)


class GlobalCritics(nn.Module):
    def __init__(
            self,
            preprocess_net: nn.Module,
            hidden_sizes: Sequence[int] = (),
            last_size: int = 1,
            preprocess_net_output_dim: Optional[int] = None,
            device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        if config.device == torch.device("cpu") or not config.distributed_HAN:
            self.preprocess = preprocess_net
        else:
            self.preprocess = torch_geometric.nn.DataParallel(preprocess_net)
        self.output_dim = last_size
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim) * config.pending_num * config.agent_num
        self.critic = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.ReLU(),
                                    nn.Linear(input_dim, last_size),
                                    nn.ReLU(), )

    def forward(
            self, obs, **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        if config.device == torch.device("cpu") or not config.distributed_HAN:
            inp = torch_geometric.data.Batch.from_data_list(obs.inp.squeeze(-1).tolist()).to(config.device)
            logits, _ = self.preprocess(inp, state=kwargs.get("state", None))
        else:
            logits, _ = self.preprocess(obs.inp.squeeze(-1).tolist())
        logits = tools.refactor_d_state(obs.shape[0], logits["devices"], obs.connections_index, obs.device_num)
        return self.critic(logits.flatten(start_dim=1))


class HAN(torch.nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]] = config.input_dim,
                 out_channels=config.hidden_dim,
                 hidden_channels=config.hidden_dim,
                 num_layers=config.num_layers, heads=config.gnn_heads):
        super().__init__()
        self.output_dim = out_channels
        self.convs = torch.nn.ModuleList()
        self.input = torch_geometric.nn.HANConv(in_channels, hidden_channels, heads=heads,
                                                metadata=config.metadata)
        for _ in range(num_layers):
            conv = torch_geometric.nn.HANConv(hidden_channels, hidden_channels, heads=heads,
                                              metadata=config.metadata)
            self.convs.append(conv)
        self.output = torch_geometric.nn.HANConv(hidden_channels, out_channels, heads=heads,
                                                 metadata=config.metadata)

    def forward(self, inp, state: Any = None):
        x_dict, edge_index_dict = inp.x_dict, inp.edge_index_dict
        if x_dict[config.node_types[2]].shape[0] == 0:
            x_dict.pop(config.node_types[2])
            edge_index_dict.pop(config.edge_types[2])
            edge_index_dict.pop(config.edge_types[3])
        x_dict = self.input(x_dict, edge_index_dict)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        x_dict = self.output(x_dict, edge_index_dict)
        return x_dict, state


class DataParalleledHAN(torch.nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]] = config.input_dim,
                 out_channels=config.hidden_dim,
                 hidden_channels=config.hidden_dim,
                 num_layers=config.num_layers, heads=config.gnn_heads):
        super().__init__()
        self.output_dim = out_channels
        self.input = torch_geometric.nn.HANConv(in_channels, hidden_channels, heads=heads,
                                                metadata=config.metadata)
        self.conv1 = torch_geometric.nn.HANConv(hidden_channels, hidden_channels, heads=heads,
                                                metadata=config.metadata)
        self.conv2 = torch_geometric.nn.HANConv(hidden_channels, hidden_channels, heads=heads,
                                                metadata=config.metadata)
        self.conv3 = torch_geometric.nn.HANConv(hidden_channels, hidden_channels, heads=heads,
                                                metadata=config.metadata)
        self.conv4 = torch_geometric.nn.HANConv(hidden_channels, hidden_channels, heads=heads,
                                                metadata=config.metadata)
        self.conv5 = torch_geometric.nn.HANConv(hidden_channels, hidden_channels, heads=heads,
                                                metadata=config.metadata)
        self.conv6 = torch_geometric.nn.HANConv(hidden_channels, hidden_channels, heads=heads,
                                                metadata=config.metadata)
        self.output = torch_geometric.nn.HANConv(hidden_channels, out_channels, heads=heads,
                                                 metadata=config.metadata)

    def forward(self, inp, state: Any = None):
        x_dict, edge_index_dict = inp.x_dict, inp.edge_index_dict
        if x_dict[config.node_types[2]].shape[0] == 0:
            x_dict.pop(config.node_types[2])
            edge_index_dict.pop(config.edge_types[2])
            edge_index_dict.pop(config.edge_types[3])
        x_dict = self.input(x_dict, edge_index_dict)
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = self.conv4(x_dict, edge_index_dict)
        x_dict = self.conv5(x_dict, edge_index_dict)
        x_dict = self.conv6(x_dict, edge_index_dict)
        x_dict = self.output(x_dict, edge_index_dict)
        return x_dict, state


class GNN(torch.nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]] = config.input_dim,
                 out_channels: int = config.hidden_dim,
                 hidden_channels=config.hidden_dim):
        super().__init__()
        self.output_dim = out_channels
        self.server_lin = torch_geometric.nn.Linear(in_channels["servers"], hidden_channels)
        self.device_lin = torch_geometric.nn.Linear(in_channels["devices"], hidden_channels)
        self.sage = SAGE(out_channels, hidden_channels)
        self.sage = to_hetero(self.sage, config.metadata, aggr='mean')

    def forward(self, inp, state: Any = None):
        inp["servers"].x = self.server_lin(inp["servers"].x)
        inp["devices"].x = self.device_lin(inp["devices"].x)
        x_dict = self.sage(inp.x_dict, inp.edge_index_dict)
        return x_dict, state


class SAGE(torch.nn.Module):
    def __init__(self, out_channels: int = config.hidden_dim,
                 hidden_channels=config.hidden_dim,
                 num_layers=config.num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.input = torch_geometric.nn.SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        for _ in range(num_layers):
            conv = torch_geometric.nn.SAGEConv(hidden_channels, hidden_channels)
            self.convs.append(conv)
        self.output = torch_geometric.nn.SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.input(x, edge_index).relu()
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = self.output(x, edge_index).relu()
        return x


class ToyModel(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 device="cpu"):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x, state: Any = None):
        return self.model(x)
