import uuid
from typing import Dict, List

import gym
import numpy as np
from gym.utils import seeding

from graph.node_graph import DeviceNode
from utils import config, tools


class EnvTest2(gym.Env):

    def __init__(self, is_random, is_test, needs_print=False):
        self.device = None
        self.is_random = is_random
        self.is_test = is_test
        self.needs_print = needs_print
        self.running: Dict[str] = {}
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
        if action_n == 0:
            actions = None
        else:
            actions = tools.get_actions(np.asarray([[action_n]])).flatten().astype(np.int64)

        if self.is_test and self.needs_print:
            print("step {}: act {}, job type {}, resource {}".format(self.time, actions,
                                                                     self.device.job_type if self.device else "None",
                                                                     self.resource))

        self.apply_action(actions)
        reward = - len(self.running) - len(self.queue)
        if self.device is not None:
            reward -= 1
        self.world_walk()
        self.time += 1
        self.generate_new_tasks()

        end = False
        if not self.queue and not self.running and self.count == config.episode_task_num:
            end = True
        obs, job_type = self.build_state()
        return {"obs": obs, "resource": self.resource, "job_type": job_type}, reward, end, {"obj_reward": 0,
                                                                                            "completion_time": 0}

    def reset(self):
        self.device = None
        self.running = {}
        self.queue = []
        self.count = 0
        self.time = 0
        self.resource = np.asarray(config.discrete_action_dimension, dtype=np.int64) - 1
        self.resource_allocation = {}
        self.generate_new_tasks()
        obs, job_type = self.build_state()
        return {"obs": obs, "resource": self.resource, "job_type": job_type}

    def render(self, mode="human"):
        pass

    def build_state(self):
        resource_onehot = np.zeros(config.discrete_action_dimension)
        resource_onehot[self.resource] = 1
        if self.queue or self.device:
            if self.device is None:
                self.device = self.queue.pop(0)
            task_type_onehot = np.zeros(len(config.JobType))
            task_type_onehot[self.device.job_type] = 1
            obs = np.concatenate([np.zeros(1), task_type_onehot, resource_onehot],
                                 dtype=np.float32)
            return obs, self.device.job_type
        else:
            obs = np.concatenate(
                [np.ones(1), np.zeros(len(config.JobType)), resource_onehot],
                dtype=np.float32)
            return obs, -1

    def apply_action(self, actions):
        if actions is not None:
            self.running[self.device.device_id] = self.device
            self.resource -= actions
            self.resource_allocation[self.device.device_id] = actions
            self.device = None
        return

    def world_walk(self):
        ids = []
        for _, device_id in enumerate(self.running):
            device = self.running[device_id]
            resource_allocation = self.resource_allocation[device_id]
            progress_delta = EnvTest2.calculate_progress(device.job_type, resource_allocation)
            device.progress += progress_delta
            if device.progress >= 1:
                ids.append(device_id)

        for id in ids:
            self.running.pop(id)
            resource_allocation = self.resource_allocation.pop(id)
            self.resource += resource_allocation
        return

    def generate_new_tasks(self):
        num = self.np_random.poisson(2, 1).item()
        for _ in range(num):
            if self.count < config.episode_task_num:
                job_type = config.JobType(self.np_random.choice(a=[j_t for j_t in config.JobType],
                                                                size=1, p=config.job_probs).item())
                device = DeviceNode(device_id=str(uuid.uuid1()),
                                    job_type=job_type)
                self.queue.append(device)
                self.count += 1
        return

    @staticmethod
    def calculate_progress(job_type, resource_allocation):
        return (resource_allocation * config.resource_progress_weight[job_type] + config.resource_progress_base[
            job_type]).sum()
