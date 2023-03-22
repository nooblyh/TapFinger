import json
import matplotlib
# matplotlib.use("QtAgg")
import numpy as np
from matplotlib import pyplot as plt
import statistics

from utils.plot import plot_avg_jct, plot_5_avg_jct


def different_workload():
    jcts = np.zeros((3, 3))
    model_names = ["Tiresias", "Optimus", "TapFinger"]
    xticks_name = ["2000", "4000", "8000"]

    for i, step in enumerate(xticks_name):
        data_folder = "./img/trace_6_agent_{}/".format(step)
        for j, model_name in enumerate(model_names):
            jct_file_name = data_folder + "{}_jct.json".format(model_name)
            current_jcts = json.load(open(jct_file_name))
            jcts[j, i] = statistics.mean(current_jcts)
            print("{}: {} tasks, JCT avg:{}".format(model_name, step, jcts[j, i]))
        # normalized
        # m = max(jcts[:, i])
        # for j, model_name in enumerate(model_names):
        #     jcts[j, i] /= m

    plot_avg_jct(jcts, model_names, xticks_name, data_folder)

def different_arrival_rate():
    jcts = np.zeros((5, 3))
    model_names = ["Tiresias", "Optimus", "TapFinger", "Random", "Min"]
    xticks_name = [r'$\lambda=2$', r'$\lambda=3$', r'$\lambda=4$']

    for i, step in enumerate(["synthetic_6_agent_ar=2", "synthetic_6_agent_ar=3", "synthetic_6_agent"]):
        data_folder = "./img/{}/".format(step)
        for j, model_name in enumerate(model_names):
            jct_file_name = data_folder + "{}_jct.json".format(model_name)
            current_jcts = json.load(open(jct_file_name))
            jcts[j, i] = statistics.mean(current_jcts)
            print("{}: {} tasks, JCT avg:{}".format(model_name, step, jcts[j, i]))

    plot_5_avg_jct(jcts, model_names, xticks_name, data_folder)

if __name__ == '__main__':
    different_workload()
