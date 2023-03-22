import uuid
from typing import Dict, List

import gym
import numpy as np
import torch
from gym.utils import seeding
from torch_geometric.data import HeteroData

from graph.node_graph import DeviceNode
from utils import config, tools


class EnvTest1(gym.Env):

    def __init__(self, is_random, is_test, needs_print=False):
        self.is_random = is_random
        self.is_test = is_test
        self.needs_print = needs_print
        self.running: Dict[str] = {}
        self.pending = np.empty(config.pending_num, dtype=object)
        self.queue: List[DeviceNode] = []
        self.resource = np.asarray(config.discrete_action_dimension, dtype=np.int64) - 1
        self.resource_allocation: Dict[str] = {}
        self.count = 0
        self.time = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_n):
        resource_allocation_act = action_n.allocation_act
        index_act = action_n.index_act

        if index_act == -1:
            actions = None
            device = None
        else:
            actions = tools.get_actions(np.asarray([[resource_allocation_act]])).flatten().astype(np.int64)
            device = self.pending[index_act]

        if self.is_test and self.needs_print:
            print("step {}: act {}, job type {}, resource {}".format(self.time, actions,
                                                                     device.job_type if device is not None else "None",
                                                                     self.resource))

        self.apply_action(device, actions)
        pending_num = np.count_nonzero(self.pending != None)
        reward = - len(self.running) - pending_num - len(self.queue)
        self.world_walk()
        self.time += 1
        self.generate_new_tasks()
        self.load_tasks()

        end = False
        if not self.queue and not self.running and (self.pending == None).all() and self.count == config.episode_task_num:
            end = True
        obs = self.build_state()
        return obs, reward, end, {}

    def reset(self):
        self.running = {}
        self.pending = np.empty(config.pending_num, dtype=object)
        self.queue = []
        self.count = 0
        self.time = 0
        self.resource = np.asarray(config.discrete_action_dimension, dtype=np.int64) - 1
        self.resource_allocation = {}
        self.generate_new_tasks()
        return self.build_state()

    def render(self, mode="human"):
        pass

    def apply_action(self, device, actions):
        if device is not None and (actions != 0).any():
            self.running[device.device_id] = device
            self.resource -= actions
            self.resource_allocation[device.device_id] = actions
            self.pending[device.drl_index] = None
        return

    def world_walk(self):
        ids = []
        for _, device_id in enumerate(self.running):
            device = self.running[device_id]
            resource_allocation = self.resource_allocation[device_id]
            progress_delta = EnvTest1.calculate_progress(device.job_type, resource_allocation)
            device.progress += progress_delta
            if device.progress >= 1:
                ids.append(device_id)

        for id in ids:
            self.running.pop(id)
            resource_allocation = self.resource_allocation.pop(id)
            self.resource += resource_allocation
        return

    def generate_new_tasks(self):
        num = self.np_random.poisson(1, 1).item()
        for _ in range(num):
            if self.count < config.episode_task_num:
                job_type = config.JobType(self.np_random.choice(a=[j_t for j_t in config.JobType],
                                                                size=1, p=config.job_probs).item())
                device = DeviceNode(device_id=str(uuid.uuid1()),
                                    job_type=job_type)
                waiting = True
                occupancy = self.pending != None
                for index, occupied in enumerate(occupancy):
                    if not occupied:
                        self.pending[index] = device
                        device.drl_index = index
                        waiting = False
                        break
                if waiting:
                    self.queue.append(device)
                self.count += 1
        return

    @staticmethod
    def calculate_progress(job_type, resource_allocation):
        return (resource_allocation * config.resource_progress_weight[job_type] + config.resource_progress_base[
            job_type]).sum()

    def load_tasks(self):
        while True:
            occupancy = self.pending != None
            if occupancy.all() or not self.queue:
                break
            device = self.queue.pop(0)
            for occupancy_index, occupied in enumerate(occupancy):
                if not occupied:
                    self.pending[occupancy_index] = device
                    device.drl_index = occupancy_index
                    break

    def build_state(self):
        resource_onehot = np.zeros(config.discrete_action_dimension)
        resource_onehot[tuple(np.split(self.resource, self.resource.shape[0]))] = 1
        servers = np.asarray([resource_onehot.flatten()])
        devices = np.zeros((config.pending_num, config.input_dim["devices"]))
        job_types = np.zeros(config.pending_num, dtype=np.int32)
        for i, d in enumerate(self.pending):
            if d is None:
                devices[i] = np.concatenate(
                    [np.ones(1), np.zeros(len(config.JobType))], dtype=np.float32)
                job_types[i] = -1
            else:
                task_type_onehot = np.zeros(len(config.JobType))
                task_type_onehot[d.job_type] = 1
                devices[i] = np.concatenate([np.zeros(1), task_type_onehot], dtype=np.float32)
                job_types[i] = d.job_type

        servers_to_devices = np.stack(
            [np.full(config.pending_num, 0, dtype=np.int32), np.arange(config.pending_num, dtype=np.int32)])
        devices_to_servers = np.stack(
            [np.arange(config.pending_num, dtype=np.int32), np.full(config.pending_num, 0, dtype=np.int32)])

        running = []
        for i, d in enumerate(self.running.values()):
            resource_onehot = np.zeros(config.discrete_action_dimension)
            resource_allocation = self.resource_allocation[d.device_id]
            resource_onehot[tuple(np.split(resource_allocation, resource_allocation.shape[0]))] = 1
            task_type_onehot = np.zeros(len(config.JobType))
            task_type_onehot[d.job_type] = 1
            running.append(
                np.concatenate([task_type_onehot, resource_onehot.flatten(), np.asarray([d.progress])], dtype=np.float32))

        running, servers_to_running, running_to_servers = EnvTest1.get_node_inp(running, "running")

        queue = []
        for i, d in enumerate(self.queue):
            task_type_onehot = np.zeros(len(config.JobType))
            task_type_onehot[d.job_type] = 1
            queue.append(np.concatenate([task_type_onehot], dtype=np.float32))

        queue, servers_to_queue, queue_to_servers = EnvTest1.get_node_inp(queue, "queue")

        inp = HeteroData({config.node_types[0]: {'x': torch.from_numpy(servers.astype(np.float32))},
                          config.node_types[1]: {'x': torch.from_numpy(np.asarray(devices, dtype=np.float32))},
                          config.node_types[2]: {'x': running},
                          config.node_types[3]: {'x': queue},
                          config.edge_types[0]: {
                              "edge_index": torch.from_numpy(np.asarray(servers_to_devices, dtype=np.int32))},
                          config.edge_types[1]: {
                              "edge_index": torch.from_numpy(np.asarray(devices_to_servers, dtype=np.int32))},
                          config.edge_types[2]: {
                              "edge_index": torch.from_numpy(np.asarray(servers_to_running, dtype=np.int32))},
                          config.edge_types[3]: {
                              "edge_index": torch.from_numpy(np.asarray(running_to_servers, dtype=np.int32))},
                          config.edge_types[4]: {
                              "edge_index": torch.from_numpy(np.asarray(servers_to_queue, dtype=np.int32))},
                          config.edge_types[5]: {
                              "edge_index": torch.from_numpy(np.asarray(queue_to_servers, dtype=np.int32))},
                          })

        obj_inp = np.empty((1,), dtype=object)
        obj_inp[0] = inp

        return {"inp": obj_inp, "job_types": job_types, "resource": self.resource.copy()}

    @staticmethod
    def get_node_inp(node_list, node_type):
        servers_to_node = np.stack(
            [np.full(len(node_list), 0, dtype=np.int32), np.arange(len(node_list), dtype=np.int32)])
        node_to_servers = np.stack(
            [np.arange(len(node_list), dtype=np.int32), np.full(len(node_list), 0, dtype=np.int32)])

        if not node_list:
            node_tensor = torch.empty((0, config.input_dim[node_type]), dtype=torch.float32)
        else:
            node_tensor = torch.from_numpy(np.asarray(node_list, dtype=np.float32))

        return node_tensor, servers_to_node, node_to_servers
