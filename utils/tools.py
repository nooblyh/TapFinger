from random import sample

import numpy as np
import torch
from numba import jit
from tianshou.data import Batch
from torch.distributions import Categorical

from utils import config


def get_actions(raw_actions):
    actions = np.zeros((raw_actions.shape[0],) + config.discrete_action_dimension, dtype=np.float64)
    actions.reshape((raw_actions.shape[0], -1))[range(raw_actions.shape[0]), raw_actions] = 1
    res = np.nonzero(actions)
    return np.stack(res[1:]).transpose()


def toy_mask_one_logits(job_types, resource):
    mask = np.full((resource.shape[0], config.one_discrete_action_space), True)
    for j_t in config.JobType:
        j_mask = job_types == j_t
        if not j_mask.any():
            continue
        # change view to mask
        tmp_mask = mask[j_mask].reshape((-1,) + config.discrete_action_dimension)
        tmp_mask = get_middle_mask(tmp_mask, config.job_min_requirement[j_t])
        # change view to original
        mask[j_mask] = tmp_mask.reshape((-1, config.one_discrete_action_space))
    mask[:, 0] = True

    # change view to mask
    tmp_mask = mask.reshape((-1,) + config.discrete_action_dimension)

    # get bound
    bound = np.repeat((np.asarray(config.discrete_action_dimension) - 1)[np.newaxis, ...], mask.shape[0], 0)
    remained = resource - bound
    bound[remained < 0] = resource[remained < 0]

    tmp_mask = get_final_mask(tmp_mask, bound, config.resource_dim)
    mask = tmp_mask.reshape((-1, config.one_discrete_action_space))
    return mask


def get_middle_mask(tmp_mask, job_min_requirement):
    for i_resource, min_requirement in enumerate(job_min_requirement):
        if i_resource + 1 == len(job_min_requirement):
            tmp_mask[..., :min_requirement] = False
        else:
            tmp_mask = np.swapaxes(tmp_mask, -1, i_resource + 1)
            tmp_mask[..., :min_requirement] = False
            tmp_mask = np.swapaxes(tmp_mask, i_resource + 1, -1)
    return tmp_mask


def get_final_mask(tmp_mask, bound, resource_dim):
    for i_resource in range(resource_dim):
        if i_resource + 1 == resource_dim:
            for i_tmp in range(tmp_mask.shape[0]):
                tmp_mask[i_tmp, ..., bound[i_tmp, i_resource] + 1:] = False
        else:
            tmp_mask = np.swapaxes(tmp_mask, -1, i_resource + 1)
            for i_tmp in range(tmp_mask.shape[0]):
                tmp_mask[i_tmp, ..., bound[i_tmp, i_resource] + 1:] = False
            tmp_mask = np.swapaxes(tmp_mask, i_resource + 1, -1)
    return tmp_mask


def get_act_from_logits(obs, logits, deterministic=False):
    skip_mask = torch.from_numpy((obs.job_types == -1).all(-1)).to(config.device)
    connections_index = torch.from_numpy(obs.connections_index).to(config.device)
    job_types = torch.from_numpy(obs.job_types).to(config.device)
    index_dist = Categorical(logits=logits.all_index_logits)
    if deterministic:
        index_act = logits.all_index_logits.argmax(-1)
    else:
        index_dist = Categorical(logits=logits.all_index_logits)
        index_act = index_dist.sample()
    connections_index = torch.gather(connections_index, -1, index_act.unsqueeze(-1)).squeeze(-1).clone()
    connections_index[skip_mask] = -1
    score = torch.gather(index_dist.probs, -1, index_act.unsqueeze(-1)).squeeze(-1).clone()
    score[skip_mask] = -1
    job_types = torch.gather(job_types, -1, index_act.unsqueeze(-1)).squeeze(-1).clone()
    job_types[skip_mask] = -1

    conflict_mask = resolve_conflict_bayes(connections_index, score, skip_mask,
                                           torch.from_numpy(obs.resource).to(config.device), job_types)

    tmp_shape = logits.all_allocation_logits.shape
    tmp_all_allocation_logits = logits.all_allocation_logits.flatten(end_dim=1)
    tmp_all_allocation_logits = tmp_all_allocation_logits[
        range(tmp_all_allocation_logits.shape[0]), index_act.flatten()].reshape(
        (tmp_shape[0], tmp_shape[1], tmp_shape[-1]))
    skip_mask |= (tmp_all_allocation_logits[..., 1:] == -torch.inf).all(-1)

    allocation_dist = Categorical(logits=tmp_all_allocation_logits)
    allocation_act = allocation_dist.sample()

    all_allocation_act = torch.full((skip_mask.shape[0], skip_mask.shape[1]), -1).to(config.device)
    all_allocation_act[~skip_mask] = allocation_act[~skip_mask]
    all_index_act = torch.full((skip_mask.shape[0], skip_mask.shape[1]), -1).to(config.device)
    all_index_act[~skip_mask] = index_act[~skip_mask]

    acts = Batch(allocation_act=all_allocation_act, index_act=all_index_act, skip_mask=skip_mask,
                 conflict_mask=conflict_mask)
    return acts


def get_log_prob(res, minibatch):
    skip_mask = minibatch.act.skip_mask.bool()
    conflict_mask = minibatch.act.conflict_mask.bool()
    log_prob = torch.zeros((minibatch.act.shape[0], minibatch.act.shape[1])).to(config.device)
    entropy = torch.zeros((minibatch.act.shape[0], minibatch.act.shape[1])).to(config.device)
    index_dist = Categorical(logits=res.logits.all_index_logits[~skip_mask])
    log_prob[~skip_mask] = index_dist.log_prob(minibatch.act.index_act[~skip_mask])
    entropy[~skip_mask] = index_dist.entropy()
    tmp_all_allocation_logits = res.logits.all_allocation_logits.flatten(end_dim=1)
    allocation_dist = Categorical(
        logits=tmp_all_allocation_logits[range(tmp_all_allocation_logits.shape[0]), minibatch.act.index_act.flatten()][
            ~(skip_mask.flatten() | conflict_mask.flatten())])

    log_prob[~(skip_mask | conflict_mask)] += allocation_dist.log_prob(
        minibatch.act.allocation_act[~(skip_mask | conflict_mask)])
    entropy[~(skip_mask | conflict_mask)] += allocation_dist.entropy()
    return log_prob, entropy


@jit(forceobj=True)
def random_server_connection(np_random):
    connect_server_num = np_random.choice(a=[i + 1 for i in range(config.agent_num)],
                                          size=1,
                                          p=[1 / config.agent_num for _ in
                                             range(config.agent_num)]).item()
    connect_server = sample([i for i in range(config.agent_num)], connect_server_num)
    connect_server.sort()
    return connect_server


def refactor_d_state(num_of_obs: int, obs_states_devices, connection, device_num):
    d_states = torch.zeros([num_of_obs, config.agent_num, config.pending_num, config.hidden_dim]).to(
        config.device)
    for n in range(num_of_obs):
        device_index = device_num[:n].sum()
        all_d_state = obs_states_devices[device_index:device_index + device_num[n]]
        for index in range(config.agent_num):
            d_states[n][index] = all_d_state[connection[n][index]]
    return d_states


# @jit(forceobj=True)
# def resolve_conflict(global_index, score, skip_mask, resource, task_types):
#     conflict = torch.full((global_index.shape[0], config.agent_num), False).to(config.device)
#     for obs_idx in range(global_index.shape[0]):
#         unique_global_index, inverse_index, counts = torch.unique(global_index[obs_idx], sorted=False,
#                                                                   return_inverse=True,
#                                                                   return_counts=True)
#         if (counts > 1).any():
#             conflict = counts > 1
#             for idx in range(conflict.shape[0]):
#                 if conflict[idx] and unique_global_index[idx] != -1:
#                     competitors = inverse_index == idx
#                     competitors_index = torch.nonzero(competitors).squeeze(-1)
#                     winner = np.random.choice(torch.nonzero(
#                         score[obs_idx, competitors] == score[obs_idx, competitors].max()).squeeze(-1).cpu().numpy())
#                     competitors[competitors_index[winner]] = False
#                     conflict[obs_idx] |= competitors
#                     # skip_mask[obs_idx] |= competitors
#     return conflict


# @jit(forceobj=True)
def resolve_conflict_bayes(global_index, score, skip_mask, resource, task_types):
    conflict_mask = torch.full((global_index.shape[0], config.agent_num), False).to(config.device)
    for obs_idx in range(global_index.shape[0]):
        unique_global_index, inverse_index, counts = torch.unique(global_index[obs_idx], sorted=False,
                                                                  return_inverse=True,
                                                                  return_counts=True)
        if (counts > 1).any():
            conflict = counts > 1
            for idx in range(conflict.shape[0]):
                if conflict[idx] and unique_global_index[idx] != -1:
                    competitors = inverse_index == idx
                    competitors_index = torch.nonzero(competitors).squeeze(-1)
                    task_type = task_types[obs_idx, competitors_index[0]].item()

                    p_server, _ = torch.min(
                        resource[obs_idx, competitors][:, :config.requirement_nonzero_num] / torch.from_numpy(
                            config.job_min_requirement[task_type][:config.requirement_nonzero_num].copy()).to(config.device),
                        -1)

                    p_server = torch.nn.functional.softmax(p_server, dim=-1)
                    reversed_score = score[obs_idx, competitors] * p_server / (
                            score[obs_idx, competitors] * p_server).sum()

                    winner = np.random.choice(
                        torch.nonzero(reversed_score == reversed_score.max()).squeeze(-1).cpu().numpy())
                    competitors[competitors_index[winner]] = False
                    conflict_mask[obs_idx] |= competitors
                    # skip_mask[obs_idx] |= competitors
    return conflict_mask
