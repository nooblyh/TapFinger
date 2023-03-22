import json
import os
import random
import sys

import gym
import numpy as np

import custom_gym.myenv
import torch
from tianshou.data import Batch

from nn.model import Actors, ToyModel
from nn.model import HAN
from trace import convert_trace
from utils import config, tools
from utils.benchmark import random_select_random_allocate, random_select_min_allocate, optimus, tiresias, \
    optimus_sync_speed_curve_fitting
from utils.plot import plot_jct, plot_task_accumulation

if __name__ == '__main__':
    RANDOM_SEED = 42
    # trace_fit = convert_trace.csv_to_dict()
    trace_fit = None
    hetero = False
    with torch.no_grad():
        model_names = ["Tiresias", "Optimus", "Random", "Min", "TapFinger"]
        # model_names = ["TapFinger", "Vanilla DRL"]
        # model_names = ["TapFinger", "No-pointer", "Optimus", "Min"]
        # model_names = ["Tiresias", "Optimus", "Random", "Min", "TapFinger", "No-HAN"]

        if "Optimus" in model_names:
            env = gym.make('ENVTEST-v3', is_random=False, is_test=True, needs_print=False, is_inference=True,
                           is_optimus=True, trace_fit=trace_fit, hetero=hetero)
            env.seed(RANDOM_SEED)
            cpu_arr = np.tile(np.arange(0, config.discrete_action_dimension[config.cpu_dim], dtype=float),
                              config.discrete_action_dimension[config.gpu_dim])
            cpu_arr = np.tile(cpu_arr, len(config.job_time)).reshape(len(config.job_time), -1)
            gpu_arr = np.arange(0, config.discrete_action_dimension[config.gpu_dim], dtype=float).repeat(
                config.discrete_action_dimension[config.cpu_dim])
            gpu_arr = np.tile(gpu_arr, len(config.job_time)).reshape(len(config.job_time), -1)

            speed_arr = np.zeros((len(config.job_time), config.discrete_action_dimension[config.cpu_dim],
                                  config.discrete_action_dimension[config.gpu_dim]), dtype=float)
            for j in config.JobType:
                if j not in config.resource_progress_weight:
                    continue
                for c in range(config.discrete_action_dimension[config.cpu_dim]):
                    for g in range(config.discrete_action_dimension[config.gpu_dim]):
                        r_a = np.asarray([0, 0])
                        r_a[config.cpu_dim], r_a[config.gpu_dim] = c, g
                        if trace_fit is None:
                            speed_arr[j, c, g] = env.calculate_progress(j, r_a, 0)
                        else:
                            speed_arr[j, c, g] = env.calculate_trace_progress(j, r_a, 0)

            cpu_arr[cpu_arr == 0] = 0.001
            gpu_arr[gpu_arr == 0] = 0.001
            params = np.zeros((len(config.JobType), 5), dtype=float)
            for j in config.JobType:
                params[j] = optimus_sync_speed_curve_fitting(cpu_arr[j], gpu_arr[j], speed_arr[j].flatten())

        test_name = "hetero_6_agent"
        for model_name in model_names:
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            is_tiresias = False
            is_optimus = False
            gnn_state = True
            if model_name == "Tiresias":
                is_tiresias = True
            elif model_name == "Optimus":
                is_tiresias = True
            elif model_name == "Vanilla DRL" or model_name == "No-HAN":
                gnn_state = False
            env = gym.make('ENVTEST-v3', is_random=False, is_test=True, needs_print=False, is_inference=True,
                           is_tiresias=is_tiresias, is_optimus=is_optimus, gnn_state=gnn_state, trace_fit=trace_fit,
                           hetero=hetero)
            env.seed(RANDOM_SEED)
            observation = env.reset()

            if model_name == "TapFinger":
                actor = Actors(HAN(), pointer_type="attn")
                actor.to(config.device)
                # trial_name = sys.argv[1]
                trial_name = "hetero_6_agent/model"
                log_path = os.path.join('img', trial_name)
                if config.device == torch.device("cpu"):
                    actor.load_state_dict(
                        torch.load(os.path.join(log_path, "best_actor.pth"), map_location=torch.device("cpu")))
                else:
                    actor.load_state_dict(torch.load(os.path.join(log_path, "best_actor.pth")))
                actor.eval()

            if model_name == "Vanilla DRL":
                actor = Actors(ToyModel(config.stacking_dim, config.hidden_dim), pointer_type="attn", is_toy=True,
                               no_GNN=True)
                actor.to(config.device)
                # trial_name = sys.argv[1]
                trial_name = "synthetic_toy/model"
                log_path = os.path.join('img', trial_name)
                if config.device == torch.device("cpu"):
                    actor.load_state_dict(
                        torch.load(os.path.join(log_path, "best_actor.pth"), map_location=torch.device("cpu")))
                else:
                    actor.load_state_dict(torch.load(os.path.join(log_path, "best_actor.pth")))
                actor.eval()

            if model_name == "No-pointer":
                actor = Actors(HAN(), pointer_type="attn", no_pointer=True)
                actor.to(config.device)
                # trial_name = sys.argv[1]
                trial_name = "no_pointer/model"
                log_path = os.path.join('img', trial_name)
                if config.device == torch.device("cpu"):
                    actor.load_state_dict(
                        torch.load(os.path.join(log_path, "best_actor.pth"), map_location=torch.device("cpu")))
                else:
                    actor.load_state_dict(torch.load(os.path.join(log_path, "best_actor.pth")))
                actor.eval()

            if model_name == "No-HAN":
                actor = Actors(ToyModel(config.stacking_dim, config.hidden_dim), pointer_type="attn", no_GNN=True)
                actor.to(config.device)
                # trial_name = sys.argv[1]
                trial_name = "no_gnn/model"
                log_path = os.path.join('img', trial_name)
                if config.device == torch.device("cpu"):
                    actor.load_state_dict(
                        torch.load(os.path.join(log_path, "best_actor.pth"), map_location=torch.device("cpu")))
                else:
                    actor.load_state_dict(torch.load(os.path.join(log_path, "best_actor.pth")))
                actor.eval()

            jct = []
            task_accumulation = []
            utility = []
            while True:
                if model_name == "Tiresias":
                    action = tiresias(observation, hetero=hetero)
                    observation, reward, done, info = env.step(action)
                elif model_name == "Optimus":
                    action = optimus(observation, params, hetero=hetero)
                    observation, reward, done, info = env.step(action)
                elif model_name == "Random":
                    action = random_select_random_allocate(observation)
                    observation, reward, done, info = env.step(action)
                elif model_name == "Min":
                    action = random_select_min_allocate(observation)
                    observation, reward, done, info = env.step(action)
                else:
                    obs = Batch.stack([observation])
                    logits, _ = actor(obs)
                    action = tools.get_act_from_logits(obs, logits)
                    action.to_numpy()
                    observation, reward, done, info = env.step(action[0])

                if info:
                    jct = jct + info["jct"]
                    task_accumulation.append(info["task_accumulation"])
                    utility.append(info["utility"])
                if done:
                    break
            env.close()
            from pathlib import Path

            jct_file_name = "img/{}/{}_jct.json".format(test_name, model_name)
            accumulation_file_name = "img/{}/{}_task_accumulation.json".format(test_name, model_name)
            utility_file_name = "img/{}/{}_utility.json".format(test_name, model_name)

            output_file = Path(jct_file_name)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            output_file = Path(accumulation_file_name)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            output_file = Path(utility_file_name)
            output_file.parent.mkdir(exist_ok=True, parents=True)

            json.dump(jct, open(jct_file_name, 'w+'))
            json.dump(task_accumulation, open(accumulation_file_name, 'w+'))
            json.dump(utility, open(utility_file_name, 'w+'))
