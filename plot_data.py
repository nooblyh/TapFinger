import json
import matplotlib
# matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
import statistics

from utils.plot import plot_jct, plot_task_accumulation, plot_cpu_utility, plot_gpu_utility

if __name__ == '__main__':
    data_folder = "./img/hetero_6_agent/"
    jcts = []
    task_accumulations = []
    utilities = []
    # model_names = ["Tiresias", "Optimus", "TapFinger"]
    model_names = ["Tiresias", "Optimus", "TapFinger", "Random", "Min"]
    # model_names = ["TapFinger", "Vanilla DRL"]
    # model_names = ["TapFinger", "No-pointer", "Optimus", "Min"]
    # model_names = ["Optimus", "TapFinger", "No-HAN"]

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
    plot_jct(jcts, model_names, data_folder)
    plot_task_accumulation(task_accumulations, model_names, data_folder)
    plot_cpu_utility(utilities, model_names, data_folder)
    plot_gpu_utility(utilities, model_names, data_folder)
    # plot_utility(utilities, model_names, data_folder)
