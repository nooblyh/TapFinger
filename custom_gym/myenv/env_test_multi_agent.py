import uuid

import gym
import numpy as np
import torch
from gym.utils import seeding
from torch_geometric.data import HeteroData

from graph.node_graph import DeviceNode
from trace_util import convert_trace
from utils import config, tools
import copy

from utils.config import JobType


class EnvTestMultiAgent(gym.Env):

    def __init__(self, is_random, is_test, needs_print=False, is_inference=False, is_optimus=False, is_tiresias=False,
                 gnn_state=True, trace_fit=None, hetero=False):
        self.gnn_state = gnn_state
        self.action_space = None
        self.is_random = is_random
        self.is_test = is_test
        self.is_inference = is_inference
        self.is_optimus = is_optimus
        self.is_tiresias = is_tiresias
        self.needs_print = needs_print
        self.trace_fit = trace_fit
        self.devices = {}
        self.running = [{} for _ in range(config.agent_num)]
        self.pending = np.empty((config.agent_num, config.pending_num), dtype=object)
        self.queue = [[] for _ in range(config.agent_num)]
        self.hetero = hetero
        if self.hetero:
            self.resource = config.hetero_resource_capacity.copy()
        else:
            self.resource = np.asarray([config.resource_capacity], dtype=np.int64).repeat(config.agent_num, 0)
        self.resource_allocation = [{} for _ in range(config.agent_num)]
        self.last_resource_allocation = None
        self.count = 0
        self.time = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_n):
        job_types = np.zeros_like(action_n["allocation_act"])
        not_ready = False
        for agent_idx in range(config.agent_num):
            resource_allocation_act = action_n["allocation_act"][agent_idx]
            index_act = action_n["index_act"][agent_idx]
            if self.is_optimus or self.is_tiresias:
                self.resource_allocation = action_n["resource_allocation"]
                self.resource = action_n["resource"]

            if "conflict_mask" in action_n and action_n["conflict_mask"][agent_idx]:
                actions = "conflict"
                device = None
                job_types[agent_idx] = -1
            elif index_act == -1:
                actions = None
                device = None
                job_types[agent_idx] = -1
            else:
                actions = tools.get_actions(np.asarray([resource_allocation_act])).flatten().astype(np.int64)
                device = self.pending[agent_idx][index_act]
                job_types[agent_idx] = device.job_type
                assert device is not None

            if self.is_test and self.needs_print:
                print("agent {} step {}: act {}, job type {}, resource {}, queue length {}, running {}".format(
                    agent_idx, self.time, actions,
                    device.job_type if device is not None else "None",
                    self.resource[agent_idx],
                    len(self.queue[agent_idx]) + np.count_nonzero(self.pending[agent_idx] != None),
                    len(self.running[agent_idx]),
                ))

            not_ready |= self.apply_action(agent_idx, device, actions)

        assert (self.resource >= 0).all()
        # reward = -len(self.devices)
        # reward = np.full((config.agent_num,), -len(self.devices))
        reward = np.zeros((config.agent_num,))

        # real time walk
        if not not_ready:
            for agent_idx in range(config.agent_num):
                reward[agent_idx] = -len(self.running[agent_idx]) - len(self.queue[agent_idx]) - np.count_nonzero(
                    self.pending[agent_idx] != None)

            _, jct = self.world_walk()
            self.time += 1
            self.generate_new_tasks()
            self.load_tasks()

        end = False
        if not self.devices and self.count == config.episode_task_num:
            end = True
        if self.is_inference and not not_ready:
            if self.hetero:
                utility = config.hetero_resource_capacity.copy() - self.resource
            else:
                utility = np.asarray([config.resource_capacity], dtype=np.int64).repeat(config.agent_num,
                                                                                        0) - self.resource
            info = {"jct": jct, "task_accumulation": len(self.devices), "utility": utility.tolist(),
                    "allocation": action_n["allocation_act"], "job_types": job_types}
        else:
            info = {}

        if self.is_optimus or self.is_tiresias:
            obs = {"running": np.asarray(self.running), "pending": self.pending,
                   "resource_allocation": self.resource_allocation, "time": self.time}
        elif not self.gnn_state:
            obs = self.stack_state()
        else:
            obs = self.build_state()
        return obs, reward, end, info

    def reset(self):
        self.devices = {}
        self.running = [{} for _ in range(config.agent_num)]
        self.pending = np.empty((config.agent_num, config.pending_num), dtype=object)
        self.queue = [[] for _ in range(config.agent_num)]
        if self.hetero:
            self.resource = config.hetero_resource_capacity.copy()
        else:
            self.resource = np.asarray([config.resource_capacity], dtype=np.int64).repeat(config.agent_num, 0)
        self.resource_allocation = [{} for _ in range(config.agent_num)]
        self.count = 0
        self.time = 0
        self.generate_new_tasks()
        if self.is_optimus or self.is_tiresias:
            return {"running": np.asarray(self.running), "pending": self.pending,
                    "resource_allocation": self.resource_allocation, "time": self.time}
        elif not self.gnn_state:
            return self.stack_state()
        else:
            return self.build_state()

    def render(self, mode="human"):
        pass

    def apply_action(self, agent_idx, device, actions):
        if device is not None and (actions != 0).any():
            self.running[agent_idx][device.device_id] = device
            self.resource[agent_idx] -= actions
            self.resource_allocation[agent_idx][device.device_id] = actions
            self.pending[agent_idx][device.drl_index[agent_idx]] = None
            device.drl_index[agent_idx] = -1
            self.del_duplicate(device, agent_idx)
            device.start_time = self.time
            return True
        else:
            return False

    def del_duplicate(self, device, agent_idx):
        for tmp_agent_idx in range(config.agent_num):
            if tmp_agent_idx == agent_idx:
                continue
            if device.drl_index[tmp_agent_idx] != -1:
                self.pending[tmp_agent_idx][device.drl_index[tmp_agent_idx]] = None
                device.drl_index[tmp_agent_idx] = -1
            for i, tmp_device in enumerate(self.queue[tmp_agent_idx]):
                if tmp_device is not None and tmp_device.device_id == device.device_id:
                    self.queue[tmp_agent_idx].pop(i)
                    break

    def world_walk(self):
        for d in self.devices:
            self.devices[d].time += 1
        reward = np.zeros((config.agent_num,))
        jct = []
        for agent_idx in range(config.agent_num):
            ids = []
            for _, device_id in enumerate(self.running[agent_idx]):
                device = self.running[agent_idx][device_id]
                resource_allocation = self.resource_allocation[agent_idx][device_id]
                if self.trace_fit:
                    progress_delta = self.calculate_trace_progress(device.job_type, resource_allocation,
                                                                   device.progress)
                else:
                    progress_delta = self.calculate_progress(device.job_type, resource_allocation, device.progress)
                if self.last_resource_allocation is not None and device_id in self.last_resource_allocation[
                    agent_idx] and (
                        self.last_resource_allocation[agent_idx][device_id] != self.resource_allocation[agent_idx][
                    device_id]).any():
                    progress_delta = progress_delta * config.switch_weight[device.job_type]
                reward[agent_idx] += progress_delta
                device.progress += progress_delta
                device.running_time += 1
                if self.is_tiresias:
                    device.attained_gpu_service += resource_allocation[1]
                if device.progress >= 1:
                    ids.append(device_id)
                    jct.append(device.time)

            for id in ids:
                self.devices.pop(id)
                self.running[agent_idx].pop(id)
                resource_allocation = self.resource_allocation[agent_idx].pop(id)
                self.resource[agent_idx] += resource_allocation
        self.last_resource_allocation = copy.deepcopy(self.resource_allocation)
        return reward, jct

    def generate_new_tasks(self):
        num = self.np_random.poisson(config.arrival_lambda, 1).item()
        for _ in range(num):
            if self.count < config.episode_task_num:
                job_type = config.JobType(self.np_random.choice(a=[j_t for j_t in config.JobType],
                                                                size=1, p=config.job_probs).item())
                device = DeviceNode(device_id=str(uuid.uuid1()),
                                    job_type=job_type)
                device.arrive_time = self.time
                self.devices[device.device_id] = device
                chosen_servers = tools.random_server_connection(self.np_random)
                for agent_idx in chosen_servers:
                    waiting = True
                    occupancy = self.pending[agent_idx] != None
                    for index, occupied in enumerate(occupancy):
                        if not occupied:
                            self.pending[agent_idx][index] = device
                            device.drl_index[agent_idx] = index
                            waiting = False
                            break
                    if waiting:
                        self.queue[agent_idx].append(device)
                self.count += 1

    def calculate_progress(self, job_type, resource_allocation, progress):
        if ((resource_allocation - config.job_min_requirement[job_type]) < 0).any():
            return 0
        else:
            res = (config.resource_progress_weight[job_type][tuple(resource_allocation)]) * (np.sqrt(1.2 - progress)) + \
                  config.resource_progress_base[job_type] + (self.np_random.random() - 0.5) * config.progress_noise
            return max(0, res)

    def calculate_trace_progress(self, job_type, resource_allocation, progress):
        if ((resource_allocation - config.job_min_requirement[job_type]) < 0).any():
            return 0
        else:
            if job_type == JobType.AUD:
                res = self.trace_fit[config.task_name[job_type]][tuple(resource_allocation)]
            else:
                res = convert_trace.get_delta_progress(progress, *self.trace_fit[config.task_name[job_type]][
                    tuple(resource_allocation)]) + (self.np_random.random() - 0.5) * config.progress_noise
            return max(0, res)

    def load_tasks(self):
        for agent_idx in range(config.agent_num):
            while True:
                occupancy = self.pending[agent_idx] != None
                if occupancy.all() or not self.queue[agent_idx]:
                    break
                device = self.queue[agent_idx].pop(0)
                for occupancy_index, occupied in enumerate(occupancy):
                    if not occupied:
                        self.pending[agent_idx][occupancy_index] = device
                        device.drl_index[agent_idx] = occupancy_index
                        break

    def build_state(self):
        servers = np.zeros((config.agent_num, config.input_dim["servers"]))
        job_types = np.zeros((config.agent_num, config.pending_num), dtype=np.int32)
        connections_index = np.zeros((config.agent_num, config.pending_num), dtype=np.int64)

        # reset index
        for d in self.devices.values():
            d.gnn_index = -1
        device_index = 0
        running_index = 0
        devices = []
        running = []

        servers_to_devices = [[], []]
        servers_to_running = [[], []]

        for agent_idx in range(config.agent_num):
            # resource_onehot = np.zeros(config.discrete_action_dimension)
            # resource_onehot[tuple(self.resource[agent_idx])] = 1
            servers[agent_idx] = np.concatenate([self.resource[agent_idx], np.asarray([len(self.queue[agent_idx])])])

            for i, d in enumerate(self.pending[agent_idx]):
                if d is None:
                    job_types[agent_idx, i] = -1
                    connections_index[agent_idx, i] = device_index
                    servers_to_devices[0].append(agent_idx)
                    servers_to_devices[1].append(device_index)
                    devices.append(np.concatenate(
                        [np.zeros(config.resource_dim), np.ones(1), np.zeros(len(config.JobType))],
                        dtype=np.float32))
                    device_index += 1
                else:
                    if d.gnn_index == -1:
                        d.gnn_index = device_index
                        # resource_onehot = np.zeros(config.discrete_action_dimension)
                        # resource_onehot[tuple(config.job_min_requirement[d.job_type])] = 1
                        task_type_onehot = np.zeros(len(config.JobType))
                        task_type_onehot[d.job_type] = 1
                        devices.append(
                            np.concatenate([config.job_min_requirement[d.job_type], np.zeros(1), task_type_onehot],
                                           dtype=np.float32))
                        device_index += 1
                    job_types[agent_idx, i] = d.job_type
                    connections_index[agent_idx, i] = d.gnn_index
                    servers_to_devices[0].append(agent_idx)
                    servers_to_devices[1].append(d.gnn_index)

            for i, d in enumerate(self.running[agent_idx].values()):
                if d.gnn_index == -1:
                    d.gnn_index = running_index
                    # resource_onehot = np.zeros(config.discrete_action_dimension)
                    # resource_onehot[tuple(self.resource_allocation[agent_idx][d.device_id])] = 1
                    task_type_onehot = np.zeros(len(config.JobType))
                    task_type_onehot[d.job_type] = 1
                    running.append(
                        np.concatenate(
                            [self.resource_allocation[agent_idx][d.device_id], task_type_onehot,
                             np.asarray([d.running_time])],
                            dtype=np.float32))
                    running_index += 1
                servers_to_running[0].append(agent_idx)
                servers_to_running[1].append(d.gnn_index)

        if not running:
            running = torch.empty((0, config.input_dim["running"]), dtype=torch.float32)
        else:
            running = torch.from_numpy(np.asarray(running, dtype=np.float32))

        servers_to_devices = np.asarray(servers_to_devices, dtype=np.int32)
        servers_to_running = np.asarray(servers_to_running, dtype=np.int32)

        devices_to_servers = servers_to_devices[[1, 0], :].copy()
        running_to_servers = servers_to_running[[1, 0], :].copy()

        inp = HeteroData({config.node_types[0]: {'x': torch.from_numpy(servers.astype(np.float32))},
                          config.node_types[1]: {'x': torch.from_numpy(np.asarray(devices, dtype=np.float32))},
                          config.node_types[2]: {'x': running},
                          config.node_types[3]: {'x': torch.from_numpy(np.zeros((1, 1), dtype=np.float32))},
                          config.edge_types[0]: {
                              "edge_index": torch.from_numpy(servers_to_devices)},
                          config.edge_types[1]: {
                              "edge_index": torch.from_numpy(devices_to_servers)},
                          config.edge_types[2]: {
                              "edge_index": torch.from_numpy(servers_to_running)},
                          config.edge_types[3]: {
                              "edge_index": torch.from_numpy(running_to_servers)},
                          config.edge_types[4]: {
                              "edge_index": torch.from_numpy(config.servers_from_central.copy())},
                          config.edge_types[5]: {
                              "edge_index": torch.from_numpy(config.central_to_servers.copy())},
                          })

        obj_inp = np.empty((1,), dtype=object)
        obj_inp[0] = inp

        return {"inp": obj_inp, "job_types": job_types, "connections_index": connections_index,
                "resource": self.resource.copy(), "device_num": len(devices)}

    def stack_state(self):
        # reset index
        for d in self.devices.values():
            d.gnn_index = -1
        device_index = 0
        job_types = np.zeros((config.agent_num, config.pending_num), dtype=np.int32)
        connections_index = np.zeros((config.agent_num, config.pending_num), dtype=np.int64)
        states = []
        earliest = np.full((config.agent_num, 2), np.inf)
        for agent_idx in range(config.agent_num):
            # resource_onehot = np.zeros(config.discrete_action_dimension)
            # resource_onehot[tuple(self.resource[agent_idx])] = 1
            server = np.concatenate([self.resource[agent_idx], np.asarray([len(self.queue[agent_idx])])])
            devices = []
            for i, d in enumerate(self.pending[agent_idx]):
                if d is None:
                    job_types[agent_idx, i] = -1
                    connections_index[agent_idx, i] = device_index
                    device_index += 1
                    devices.append(np.concatenate(
                        [np.zeros(config.resource_dim), np.ones(1), np.zeros(len(config.JobType)), server],
                        dtype=np.float32))
                else:
                    if earliest[agent_idx, 1] > d.arrive_time:
                        earliest[agent_idx, 0] = i
                        earliest[agent_idx, 1] = d.arrive_time
                    if d.gnn_index == -1:
                        d.gnn_index = device_index
                        device_index += 1
                    job_types[agent_idx, i] = d.job_type
                    connections_index[agent_idx, i] = d.gnn_index
                    task_type_onehot = np.zeros(len(config.JobType))
                    task_type_onehot[d.job_type] = 1
                    devices.append(
                        np.concatenate([config.job_min_requirement[d.job_type], np.zeros(1), task_type_onehot, server],
                                       dtype=np.float32))
            if earliest[agent_idx, 0] == np.inf:
                earliest[agent_idx, 0] = 0
            states.append(devices)
        return {"inp": np.asarray(states), "job_types": job_types, "connections_index": connections_index,
                "resource": self.resource.copy(), "first_job": np.int64(earliest[:, 0])}
