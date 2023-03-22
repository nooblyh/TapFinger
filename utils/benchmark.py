import numpy as np
from scipy.optimize import curve_fit
from tianshou.data import Batch

from custom_gym.myenv.env_test_multi_agent import EnvTestMultiAgent
from utils import config
from utils.config import JobType
from utils.tools import toy_mask_one_logits


def random_select_random_allocate(obs):
    agent, index = np.nonzero(obs["job_types"] != -1)
    index_act = np.full((config.agent_num,), -1, dtype=np.int64)
    allocation_act = np.full((config.agent_num,), -1, dtype=np.int64)
    conflict_mask = np.full((config.agent_num,), False)
    resource = np.expand_dims(obs["resource"], 1).repeat(config.pending_num, 1)
    resource = resource.reshape(-1, *resource.shape[2:])
    mask = toy_mask_one_logits(obs["job_types"].flatten(), resource)
    mask = mask.reshape((config.agent_num, config.pending_num, -1))
    check_conflict = []

    for agent_idx in range(config.agent_num):
        if index[agent == agent_idx].shape[0] != 0:
            index_act[agent_idx] = np.random.choice(a=index[agent == agent_idx])
            if obs['connections_index'][agent_idx][index_act[agent_idx]] in check_conflict:
                conflict_mask[agent_idx] = True
                allocation_act[agent_idx] = -1
                continue
            else:
                check_conflict.append(obs['connections_index'][agent_idx][index_act[agent_idx]])
            (available_index,) = np.nonzero(mask[agent_idx, index_act[agent_idx]])
            allocation_act[agent_idx] = np.random.choice(a=available_index)
        else:
            index_act[agent_idx] = -1
            allocation_act[agent_idx] = -1

    return Batch(allocation_act=allocation_act, index_act=index_act, conflict_mask=conflict_mask)


def random_select_min_allocate(obs):
    agent, index = np.nonzero(obs["job_types"] != -1)
    index_act = np.full((config.agent_num,), -1, dtype=np.int64)
    allocation_act = np.full((config.agent_num,), -1, dtype=np.int64)
    conflict_mask = np.full((config.agent_num,), False)
    resource = np.expand_dims(obs["resource"], 1).repeat(config.pending_num, 1)
    resource = resource.reshape(-1, *resource.shape[2:])
    mask = toy_mask_one_logits(obs["job_types"].flatten(), resource)
    mask = mask.reshape((config.agent_num, config.pending_num,) + config.discrete_action_dimension)
    check_conflict = []

    for agent_idx in range(config.agent_num):
        if index[agent == agent_idx].shape[0] != 0:
            index_act[agent_idx] = np.random.choice(a=index[agent == agent_idx])
            if obs['connections_index'][agent_idx][index_act[agent_idx]] in check_conflict:
                conflict_mask[agent_idx] = True
                allocation_act[agent_idx] = -1
                continue
            else:
                check_conflict.append(obs['connections_index'][agent_idx][index_act[agent_idx]])

            min_resource_allocation = config.job_min_requirement[obs["job_types"][agent_idx][index_act[agent_idx]]]
            if mask[agent_idx, index_act[agent_idx]][tuple(min_resource_allocation)]:
                resource_onehot = np.zeros(config.discrete_action_dimension)
                resource_onehot[tuple(min_resource_allocation)] = 1
                allocation_act[agent_idx] = resource_onehot.flatten().nonzero()[0].item()
            else:
                allocation_act[agent_idx] = 0
        else:
            index_act[agent_idx] = -1
            allocation_act[agent_idx] = -1

    return Batch(allocation_act=allocation_act, index_act=index_act, conflict_mask=conflict_mask)


def optimus_sync_speed_fit_func(X, a, b, c, d, e):
    p, w = X
    return 1 / (a * 1 / w + b + c * w / p + d * w + e * p)


def optimus_sync_speed_curve_fitting(cpu_arr, gpu_arr, speed_arr):
    param_bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
    sigma = np.ones(len(cpu_arr))
    params = curve_fit(optimus_sync_speed_fit_func, (cpu_arr, gpu_arr), speed_arr,
                       sigma=np.array(sigma), absolute_sigma=False, bounds=param_bounds)
    return params[0]


def optimus(obs, params, hetero=False):
    if hetero:
        resource = config.hetero_resource_capacity.copy()
    else:
        resource = np.asarray([config.resource_capacity], dtype=np.int64).repeat(config.agent_num, 0)
    maximal = np.asarray(config.discrete_action_dimension) - 1
    index_act = np.full((config.agent_num,), -1, dtype=np.int64)
    allocation_act = np.full((config.agent_num,), -1, dtype=np.int64)
    duplicate = []
    for agent_idx in range(config.agent_num):
        running = obs["running"][agent_idx]
        r_a = None
        d_curr = None
        if hetero:
            shortage = np.min(config.hetero_resource_capacity[agent_idx] // (config.job_min_requirement[JobType.IMG]))
        else:
            shortage = np.min(np.asarray(config.resource_capacity) // (config.job_min_requirement[JobType.IMG]))
        if len(running) < shortage:
            tmp = obs["pending"][agent_idx][obs["pending"][agent_idx].nonzero()[0]]
            if tmp.shape[0] == 0:
                index_act[agent_idx] = -1
            else:
                for d_curr in tmp:
                    if d_curr.device_id in duplicate:
                        continue
                    else:
                        duplicate.append(d_curr.device_id)
                        index_act[agent_idx] = d_curr.drl_index[agent_idx]
                        r_a = config.job_min_requirement[d_curr.job_type].copy()
                        break
                if r_a is None:
                    index_act[agent_idx] = -1

        for r in running:
            obs["resource_allocation"][agent_idx][running[r].device_id] = config.job_min_requirement[
                running[r].job_type].copy()
            resource[agent_idx] -= obs["resource_allocation"][agent_idx][running[r].device_id]

        if r_a is not None:
            obs["resource_allocation"][agent_idx][d_curr.device_id] = r_a
            resource[agent_idx] -= r_a

        # allocation
        length = len(running) if r_a is None else len(running) + 1
        while (resource[agent_idx] >= 0).any():
            if not length:
                break
            q = np.zeros((length, 2))
            for l in range(length):
                if r_a is not None and l == length - 1:
                    device = d_curr
                else:
                    device = running[list(running.keys())[l]]
                estimate_speed = optimus_sync_speed_fit_func(
                    tuple(obs["resource_allocation"][agent_idx][device.device_id]), *params[device.job_type])
                base = (1 - device.progress) / estimate_speed
                for ll in range(2):
                    delta = np.asarray([0, 0])
                    if resource[agent_idx][ll] > 0:
                        delta[ll] = 1
                    if (obs["resource_allocation"][agent_idx][device.device_id] + delta > maximal).any():
                        q[l, ll] = 0
                    else:
                        estimate_speed = optimus_sync_speed_fit_func(
                            tuple(obs["resource_allocation"][agent_idx][device.device_id] + delta),
                            *params[device.job_type])
                        now = (1 - device.progress) / estimate_speed
                        q[l, ll] = (base - now) / obs["resource_allocation"][agent_idx][device.device_id][ll]
            if (q <= 0).all():
                break
            l, ll = np.unravel_index(np.argmax(q), q.shape)
            if r_a is not None and l == length - 1:
                device = d_curr
            else:
                device = running[list(running.keys())[l]]
            delta = np.asarray([0, 0])
            delta[ll] = 1
            obs["resource_allocation"][agent_idx][device.device_id] += delta
            resource[agent_idx] -= delta

        if r_a is not None:
            resource_onehot = np.zeros(config.discrete_action_dimension)
            resource_onehot[tuple(obs["resource_allocation"][agent_idx][d_curr.device_id])] = 1
            allocation_act[agent_idx] = resource_onehot.flatten().nonzero()[0].item()
            obs["resource_allocation"][agent_idx].pop(d_curr.device_id)
            resource[agent_idx] += r_a
    assert (resource >= 0).all()
    return {"allocation_act": allocation_act, "index_act": index_act, "resource_allocation": obs["resource_allocation"],
            "resource": resource}


def tiresias(obs, elastic=True, hetero=False):
    if hetero:
        resource = config.hetero_resource_capacity.copy()
    else:
        resource = np.asarray([config.resource_capacity], dtype=np.int64).repeat(config.agent_num, 0)
    maximal = np.asarray(config.discrete_action_dimension) - 1
    index_act = np.full((config.agent_num,), -1, dtype=np.int64)
    allocation_act = np.full((config.agent_num,), -1, dtype=np.int64)
    duplicate = []
    for agent_idx in range(config.agent_num):
        running = obs["running"][agent_idx]
        r_a = None
        d_curr = None
        if hetero:
            shortage = np.min(config.hetero_resource_capacity[agent_idx] // (config.job_min_requirement[JobType.IMG]))
        else:
            shortage = np.min(np.asarray(config.resource_capacity) // (config.job_min_requirement[JobType.IMG]))
        if len(running) < shortage:
            tmp = obs["pending"][agent_idx][obs["pending"][agent_idx].nonzero()[0]]
            if tmp.shape[0] == 0:
                index_act[agent_idx] = -1
            else:
                for d_curr in tmp:
                    if d_curr.device_id in duplicate:
                        continue
                    else:
                        duplicate.append(d_curr.device_id)
                        index_act[agent_idx] = d_curr.drl_index[agent_idx]
                        r_a = np.zeros((config.resource_dim,), dtype=np.int64)
                        break
                if r_a is None:
                    index_act[agent_idx] = -1

        for r in running:
            obs["resource_allocation"][agent_idx][running[r].device_id] = np.zeros((config.resource_dim,),
                                                                                   dtype=np.int64)

        if r_a is not None:
            obs["resource_allocation"][agent_idx][d_curr.device_id] = r_a

        # allocation
        length = len(running) if r_a is None else len(running) + 1
        las_now = np.zeros((length,))
        while (resource[agent_idx][config.gpu_dim] > 0).any():
            if not length:
                break
            las = np.zeros((length,))
            las += las_now
            start_time = np.zeros((length,))
            for l in range(length):
                if r_a is not None and l == length - 1:
                    device = d_curr
                else:
                    device = running[list(running.keys())[l]]
                if device.start_time == -1:
                    start_time[l] = obs["time"]
                else:
                    start_time[l] = device.start_time
                if elastic:
                    target = maximal[config.gpu_dim]
                else:
                    target = config.job_min_requirement[device.job_type][config.gpu_dim]
                if obs["resource_allocation"][agent_idx][device.device_id][config.gpu_dim] + 1 > target:
                    las[l] = np.inf
                else:
                    las[l] += device.attained_gpu_service
            if (las == np.inf).all():
                break
            res = list(zip(las, start_time,
                           list(running.keys()) + [d_curr.device_id] if r_a is not None else list(running.keys())))
            res.sort(key=lambda y: y[0])
            middle_index = len(res) // 2
            levels = [res[:middle_index], res[middle_index:]]

            for level_i in range(2):
                levels[level_i].sort(key=lambda y: y[1])
                device_id = None
                for r in levels[level_i]:
                    if r[0] != np.inf:
                        device_id = r[-1]
                        break
                if device_id is not None:
                    break

            if r_a is not None and device_id == d_curr.device_id:
                device = d_curr
            else:
                device = running[device_id]
            delta = np.zeros((config.resource_dim,), dtype=np.int64)
            delta[config.gpu_dim] = 1

            obs["resource_allocation"][agent_idx][device.device_id] += delta
            resource[agent_idx] -= delta
            if r_a is not None and device_id == d_curr.device_id:
                l = length - 1
            else:
                l = list(running.keys()).index(device_id)
            las_now[l] += 1

        if np.count_nonzero(las_now != 0) != 0:
            cpu_resource = min(resource[agent_idx][config.cpu_dim] // np.count_nonzero(las_now != 0),
                               maximal[config.cpu_dim])
            for device_id in obs["resource_allocation"][agent_idx]:
                if r_a is not None and device_id == d_curr.device_id:
                    idx = length - 1
                else:
                    idx = list(running.keys()).index(device_id)
                if las_now[idx] != 0:
                    obs["resource_allocation"][agent_idx][device_id][config.cpu_dim] += cpu_resource
                    resource[agent_idx][config.cpu_dim] -= cpu_resource

            while resource[agent_idx][config.cpu_dim] > 0:
                tmp = resource[agent_idx][config.cpu_dim]
                for device_id in obs["resource_allocation"][agent_idx]:
                    if r_a is not None and device_id == d_curr.device_id:
                        idx = length - 1
                    else:
                        idx = list(running.keys()).index(device_id)
                    if obs["resource_allocation"][agent_idx][device_id][config.cpu_dim] + 1 <= maximal[
                        config.cpu_dim] and las_now[idx] != 0:
                        obs["resource_allocation"][agent_idx][device_id][config.cpu_dim] += 1
                        resource[agent_idx][config.cpu_dim] -= 1
                    if resource[agent_idx][config.cpu_dim] == 0:
                        break
                if tmp == resource[agent_idx][config.cpu_dim]:
                    break

        if r_a is not None:
            resource_onehot = np.zeros(config.discrete_action_dimension)
            resource_onehot[tuple(obs["resource_allocation"][agent_idx][d_curr.device_id])] = 1
            allocation_act[agent_idx] = resource_onehot.flatten().nonzero()[0].item()
            r_a = obs["resource_allocation"][agent_idx].pop(d_curr.device_id)
            resource[agent_idx] += r_a
    assert (resource >= 0).all()
    return {"allocation_act": allocation_act, "index_act": index_act, "resource_allocation": obs["resource_allocation"],
            "resource": resource}
