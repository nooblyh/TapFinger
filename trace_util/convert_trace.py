import csv

import numpy as np
# matplotlib.use("QtAgg")
# from matplotlib import pyplot as plt, cycler
from scipy.optimize import curve_fit

from utils import config


def csv_to_dict():
    # csv_file = open("./trace/trace.csv", "r")
    csv_file = open("./trace_container.csv", "r")
    reader = csv.reader(csv_file)

    result = {}
    for item in reader:
        val = float(item[3]) + 14
        if item[0] in result:
            if item[0] == "audio":
                result[item[0]][int(item[1]), int(item[2])] = 5 / val
            else:
                result[item[0]][int(item[1]), int(item[2])] = (5 / val, 0)
        else:
            if item[0] == "audio":
                result[item[0]] = np.zeros(config.discrete_action_dimension, dtype=float)
                result[item[0]][int(item[1]), int(item[2])] = 5 / val
            else:
                result[item[0]] = np.empty(config.discrete_action_dimension, dtype=object)
                result[item[0]][int(item[1]), int(item[2])] = (5 / val, 0)
    csv_file.close()
    return result

def old_csv_to_dict():
    # csv_file = open("./trace/trace.csv", "r")
    csv_file = open("./trace_container.csv", "r")
    reader = csv.reader(csv_file)

    result = {}
    for item in reader:
        if item[0] in result:
            if item[0] == "audio":
                result[item[0]][int(item[1]), int(item[2])] = 10 / float(item[3])
            else:
                if item[0] == "lm":
                    scale = 10
                else:
                    scale = 10
                x, y, result[item[0]][int(item[1]), int(item[2])] = fit(item, scale)
        else:
            if item[0] == "audio":
                result[item[0]] = np.zeros(config.discrete_action_dimension, dtype=float)
                result[item[0]][int(item[1]), int(item[2])] = 10 / float(item[3])
            else:
                if item[0] == "lm":
                    scale = 10
                else:
                    scale = 10
                result[item[0]] = np.empty(config.discrete_action_dimension, dtype=object)
                x, y, result[item[0]][int(item[1]), int(item[2])] = fit(item, scale)
    csv_file.close()
    return result


def progress_fit_func(x, a, b):
    return a * x + b


def get_delta_progress(y, a, b):
    step = (y - b) / a
    return progress_fit_func(step + 1, a, b) - y


def progress_curve_fitting(x_arr, y_arr):
    param_bounds = ([0, 0], [np.inf, np.inf])
    params = curve_fit(progress_fit_func, x_arr, y_arr, bounds=param_bounds)
    return params[0]


def fit(item, scale):
    x = [tuple(float(s) for s in v.strip("'()'").split(","))[0] for v in item[4:]]
    y = [tuple(float(s) for s in v.strip("'()'").split(","))[1] for v in item[4:]]
    x, y = np.array(x), np.array(y)
    x /= scale
    max_y_idx = y.argmax()
    y = (y - y[max_y_idx])
    min_y_idx = y.argmin()
    y = y / y[min_y_idx]
    y[y == 0] = 0
    # return x, y, scipy.interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
    return x, y, progress_curve_fitting(x, y)


def plot(x, y, result, item):
    plt.style.use('ieee')
    fig1, ax1 = plt.subplots()
    xnew = np.linspace(0, 20, 100)
    ax1.plot(x, y, 'o', xnew, progress_fit_func(xnew, *result[item[0]][int(item[1]), int(item[2])]), '-')
    ax1.legend(['data', 'linear'], loc='best')
    plt.savefig(item[0] + '.png')


def get_jct():
    csv_file = open("trace_container.csv", "r")
    reader = csv.reader(csv_file)
    result = np.zeros(config.discrete_action_dimension, dtype=float)
    for item in reader:
        if item[0] == "lm":
            result[int(item[1]), int(item[2])] = item[3]
    csv_file.close()
    return result


def plot_motivation_cpu_on_gpu():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()["color"]

    plt.rcParams['axes.facecolor'] = "whitesmoke"

    plt.style.use('ieee')
    linestyle = plt.rcParams['axes.prop_cycle'].by_key()["linestyle"]
    default_cycler = (cycler(color=colors[:len(linestyle)]) + cycler(linestyle=linestyle))
    plt.rcParams['axes.prop_cycle'] = default_cycler
    result = get_jct()
    fig1, ax1 = plt.subplots()
    for i in [1, 4, 8, 16]:
        x, y = [], []
        for j in range(1, 9):
            x.append(j)
            y.append(result[i, j])
        ax1.plot(x, y)
    ax1.legend(['CPU=1', 'CPU=4', 'CPU=8', 'CPU=16'], loc='best')
    ax1.set_xlabel("GPU")
    ax1.set_ylabel("Completion Time (s)")
    ax1.xaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()
    plt.xticks(x)
    plt.grid()
    plt.tight_layout()
    plt.savefig('motivation_CPU_on_GPU.pdf')


def plot_motivation_gpu_on_cpu():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()["color"]

    plt.rcParams['axes.facecolor'] = "whitesmoke"

    plt.style.use('ieee')
    linestyle = plt.rcParams['axes.prop_cycle'].by_key()["linestyle"]
    default_cycler = (cycler(color=colors[:len(linestyle)]) + cycler(linestyle=linestyle))
    plt.rcParams['axes.prop_cycle'] = default_cycler
    result = get_jct()
    fig1, ax1 = plt.subplots()
    for j in [2, 4, 6, 8]:
        x, y = [], []
        for i in range(1, 17):
            x.append(i)
            y.append(result[i, j])
        ax1.plot(x, y)
    ax1.legend(['GPU=2', 'GPU=4', 'GPU=6', 'GPU=8'], loc='best')
    ax1.set_xlabel("CPU")
    ax1.set_ylabel("Completion Time (s)")
    ax1.xaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()
    plt.xticks(x)
    plt.grid()
    plt.tight_layout()
    plt.savefig('motivation_GPU_on_CPU.pdf')


if __name__ == "__main__":
#     plot_motivation_cpu_on_gpu()
#     plot_motivation_gpu_on_cpu()
    csv_to_dict()
    result = csv_to_dict()
    print(get_delta_progress(0.8, *result["mnist"][16, 8]))
