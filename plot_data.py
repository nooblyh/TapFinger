from datetime import datetime
import json
import time
import pandas as pd
import matplotlib
# matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
import statistics

from utils.plot import plot_jct, plot_task_accumulation, plot_cpu_utility, plot_gpu_utility, plot_jct_container

def get_data_container(model_names):
    jcts = []
    task_accumulations = []
    utilities = []
    for model_name in model_names:
        print(model_name)
        create_time_file_name = data_folder + "{}_create_time.json".format(model_name)
        accumulation_file_name = data_folder + "{}_task_accumulation.json".format(model_name)
        utility_file_name = data_folder + "{}_utility.json".format(model_name)
        exit_time_file_name = data_folder + "{}_start_exited.json".format(model_name)
        create_time = json.load(open(create_time_file_name))
        exit_time = json.load(open(exit_time_file_name))
        assert len(create_time) == len(exit_time)
        current_jcts = []
        for key in create_time:
            et = pd.to_datetime(exit_time["/"+key][1], format='%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
            current_jcts.append(et - create_time[key])
            # print(et - create_time[key])
        jcts.append(current_jcts)
        task_accumulations.append(json.load(open(accumulation_file_name)))
        utilities.append(json.load(open(utility_file_name)))
        tmp = sorted(current_jcts)
        print("{}: JCT median:{}, JCT avg:{}".format(model_name, statistics.median(tmp), statistics.mean(tmp)))
    return jcts, task_accumulations, utilities

def get_data(model_names):
    jcts = []
    task_accumulations = []
    utilities = []
    for model_name in model_names:
        jct_file_name = data_folder + "{}_jct.json".format(model_name)
        accumulation_file_name = data_folder + "{}_task_accumulation.json".format(model_name)
        utility_file_name = data_folder + "{}_utility.json".format(model_name)
        current_jcts = json.load(open(jct_file_name))
        jcts.append(current_jcts)
        task_accumulations.append(json.load(open(accumulation_file_name)))
        utilities.append(json.load(open(utility_file_name)))
        tmp = sorted(current_jcts)
        print("{}: JCT median:{}, JCT avg:{}".format(model_name, statistics.median(tmp), statistics.mean(tmp)))
    return jcts, task_accumulations, utilities

if __name__ == '__main__':
    data_folder = "./img/container_version/"

    # model_names = ["Tiresias", "Optimus", "TapFinger"]
    model_names = ["Tiresias", "Optimus", "TapFinger", "Random"]
    # model_names = ["TapFinger", "Vanilla DRL"]
    # model_names = ["TapFinger", "No-pointer", "Optimus", "Min"]
    # model_names = ["Optimus", "TapFinger", "No-HAN"]

    jcts, task_accumulations, utilities = get_data_container(model_names)
    # plot_jct(jcts, model_names, data_folder)
    plot_jct_container(jcts, model_names, data_folder)
    plot_task_accumulation(task_accumulations, model_names, data_folder)
    plot_cpu_utility(utilities, model_names, data_folder)
    plot_gpu_utility(utilities, model_names, data_folder)
    # plot_utility(utilities, model_names, data_folder)
