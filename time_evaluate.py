import os
import time

import gym
import numpy as np
import torch
import random

from matplotlib import pyplot as plt, cycler, ticker

import custom_gym.myenv
import statistics
from tianshou.data import Batch

from nn.model import Actors, HAN
from trace import convert_trace
from utils import config, tools


def log_time():
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    is_tiresias = False
    is_optimus = False
    gnn_state = True
    trace_fit = convert_trace.csv_to_dict()

    env = gym.make('ENVTEST-v3', is_random=False, is_test=True, needs_print=False, is_inference=True,
                   is_tiresias=is_tiresias, is_optimus=is_optimus, gnn_state=gnn_state, trace_fit=trace_fit)
    env.seed(RANDOM_SEED)
    observation = env.reset()

    actor = Actors(HAN(), pointer_type="attn")
    actor.to(config.device)
    # trial_name = sys.argv[1]
    trial_name = "trace_6_agent_2000/model"
    log_path = os.path.join('img', trial_name)
    if config.device == torch.device("cpu"):
        actor.load_state_dict(torch.load(os.path.join(log_path, "best_actor.pth"), map_location=torch.device("cpu")))
    else:
        actor.load_state_dict(torch.load(os.path.join(log_path, "best_actor.pth")))
    actor.eval()
    jct = []
    task_accumulation = []
    utility = []
    inference_time = []
    while True:
        obs = Batch.stack([observation])
        torch.save(obs, 'data/6_agent_obs.pt')
        start_time = time.time()
        logits, _ = actor(obs)
        action = tools.get_act_from_logits(obs, logits)
        inference_time.append(time.time() - start_time)
        torch.save(action, 'data/6_agent_act.pt')
        action.to_numpy()
        observation, reward, done, info = env.step(action[0])

        jct = jct + info["jct"]
        task_accumulation.append(info["task_accumulation"])
        utility.append(info["utility"])
        if done:
            break
    env.close()
    print(statistics.mean(inference_time))

def plot_time_evaluate():
    plt.style.use(['default'])
    prop_cycle = plt.rcParams['axes.prop_cycle']

    plt.rcParams['axes.facecolor'] = "whitesmoke"

    plt.style.use(['ieee'])
    plt.rcParams['axes.prop_cycle'] = prop_cycle

    fig1, ax1 = plt.subplots()

    x_ticks_labels = ["1000", "2000", "3000", "4000"]
    x = np.arange(4)
    infer_time = np.asarray([0.005344090468774342, 0.005344090468774342, 0.005344090468774342, 0.005344090468774342])
    communication_time = np.asarray([0.009549437284469604, 0.015254642486572265, 0.023978033542633057, 0.02506453585624695])
    idle = np.ones(1) - infer_time - communication_time
    ratio = (infer_time + communication_time)*100

    ax1.bar(x, communication_time, width=0.4, label="Communication Time")
    ax1.bar(x, infer_time, width=0.4, bottom=communication_time, label="Inference Time")
    ax1.bar(x, idle, width=0.4, bottom=communication_time+infer_time)

    for i in x:
        ax1.text(i, communication_time[i] + infer_time[i]+0.05, "{:.2f}%".format(ratio[i]), ha='center', va='center')

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_ticks_labels)
    ax1.set_ylabel("Overhead in Scheduling Interval")
    ax1.set_xlabel("Distance from Coordinator (km)")
    ax1.legend(loc='upper right')
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    plt.savefig('./overhead_ratio.pdf')
    return

if __name__ == "__main__":
    plot_time_evaluate()
    # import requests
    #
    # ip = "http://192.168.50.202:8888"
    # url = '{}/data/'.format(ip)
    # test = []
    # for _ in range(100):
    #     tmp = 0
    #     for name in ["obs.pt", "act.pt"]:
    #         tmp_url = url + name
    #         start_time = time.time()
    #         r = requests.get(tmp_url, allow_redirects=True)
    #         tmp += time.time() - start_time
    #         time.sleep(1)
    #     test.append(tmp)
    # print(statistics.mean(test))
