import json
import statistics

import matplotlib
# matplotlib.use("QtAgg")
import numpy as np
from matplotlib import pyplot as plt, cycler
from matplotlib.pyplot import gca


def plot_all():
    return


def plot_jct(data, names, data_dir):
    plt.style.use(['default'])
    prop_cycle = plt.rcParams['axes.prop_cycle']

    plt.rcParams['axes.facecolor'] = "whitesmoke"

    plt.style.use(['ieee'])
    plt.rcParams['axes.prop_cycle'] = prop_cycle
    # don't show outlier points
    fig1, ax1 = plt.subplots()
    ax1.boxplot(data, 0, '*', labels=names)
    ax1.set_ylabel("Completion Time (Timestep)")
    plt.tight_layout()
    plt.savefig(data_dir + 'jct.pdf')


def plot_jct_container(data, names, data_dir):
    plt.style.use(['default'])
    prop_cycle = plt.rcParams['axes.prop_cycle']

    plt.rcParams['axes.facecolor'] = "whitesmoke"

    plt.style.use(['ieee'])
    plt.rcParams['axes.prop_cycle'] = prop_cycle
    # don't show outlier points
    fig1, ax1 = plt.subplots()
    ax1.boxplot(data, 0, '*', labels=names)
    ax1.set_ylabel("Completion Time (Second)")
    plt.tight_layout()
    plt.savefig(data_dir + 'jct.pdf')

def plot_task_accumulation(data, names, data_dir):
    plt.style.use(['default'])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()["color"]

    plt.rcParams['axes.facecolor'] = "whitesmoke"

    plt.style.use('ieee')
    linestyle = plt.rcParams['axes.prop_cycle'].by_key()["linestyle"]
    default_cycler = (cycler(color=colors[:len(linestyle)]) + cycler(linestyle=linestyle))
    plt.rcParams['axes.prop_cycle'] = default_cycler

    fig1, ax1 = plt.subplots()
    for i in range(len(data)):
        if i == 4:
            ax1.plot(data[i], label=names[i], marker='^', markersize=5, color=colors[4], markevery=12,
                     markerfacecolor='white')
        else:
            ax1.plot(data[i], label=names[i])
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Number of Tasks in the System")
    plt.tight_layout()
    plt.savefig(data_dir + 'ta.pdf')
    return


def plot_cpu_utility(data, names, data_dir):
    plt.style.use(['default'])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()["color"]

    plt.rcParams['axes.facecolor'] = "whitesmoke"

    plt.style.use('ieee')
    linestyle = plt.rcParams['axes.prop_cycle'].by_key()["linestyle"]
    default_cycler = (cycler(color=colors[:len(linestyle)]) + cycler(linestyle=linestyle))
    plt.rcParams['axes.prop_cycle'] = default_cycler

    fig1, ax = plt.subplots(1, 1)
    for i in range(len(data)):
        x = np.asarray(data[i])
        x = np.average(x, 1)
        if i == 4:
            ax.plot(x[:, 0], label=names[i], marker='^', markersize=5, color=colors[4], markevery=12,
                    markerfacecolor='white')
        else:
            ax.plot(x[:, 0], label=names[i])
    ax.legend()
    ax.grid()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average CPU Usage")
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()
    plt.savefig(data_dir + 'cpu_utility.pdf')
    return


def plot_gpu_utility(data, names, data_dir):
    plt.style.use(['default'])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()["color"]

    plt.rcParams['axes.facecolor'] = "whitesmoke"

    plt.style.use('ieee')
    linestyle = plt.rcParams['axes.prop_cycle'].by_key()["linestyle"]
    default_cycler = (cycler(color=colors[:len(linestyle)]) + cycler(linestyle=linestyle))
    plt.rcParams['axes.prop_cycle'] = default_cycler

    fig1, ax = plt.subplots(1, 1)
    for i in range(len(data)):
        x = np.asarray(data[i])
        x = np.average(x, 1)
        if i == 4:
            ax.plot(x[:, 1], label=names[i], marker='^', markersize=5, color=colors[4], markevery=12,
                    markerfacecolor='white')
        else:
            ax.plot(x[:, 1], label=names[i])
    ax.legend()
    ax.grid()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average GPU Usage")
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()
    plt.savefig(data_dir + 'gpu_utility.pdf')
    return

def plot_avg_jct(data, names, xticks_name, data_dir):
    plt.style.use(['default'])
    prop_cycle = plt.rcParams['axes.prop_cycle']

    plt.rcParams['axes.facecolor'] = "whitesmoke"

    plt.style.use(['ieee'])
    plt.rcParams['axes.prop_cycle'] = prop_cycle

    fig1, ax1 = plt.subplots()
    x = np.arange(data.shape[-1])
    total_width, n = 0.5, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    ax1.bar(x - 1 * width, data[0], width=width, label=names[0])
    ax1.bar(x, data[1], width=width, label=names[1])
    ax1.bar(x + 1 * width, data[2], width=width, label=names[2])

    ax1.set_xticks(x)
    ax1.set_xticklabels(xticks_name)

    ax1.set_ylabel("Average Completion Time (Timestep)")
    ax1.set_xlabel("Number of Tasks")
    ax1.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(data_dir + 'avg_jct.pdf')

def plot_5_avg_jct(data, names, xticks_name, data_dir):
    plt.style.use(['default'])
    prop_cycle = plt.rcParams['axes.prop_cycle']

    plt.rcParams['axes.facecolor'] = "whitesmoke"

    plt.style.use(['ieee'])
    plt.rcParams['axes.prop_cycle'] = prop_cycle

    fig1, ax1 = plt.subplots()
    x = np.arange(data.shape[-1])
    total_width, n = 0.5, 5
    width = total_width / n

    p1 = ax1.bar(x - 1 * width, data[0], width=width, label=names[0])
    p2 = ax1.bar(x - 2 * width, data[1], width=width, label=names[1])
    p3 = ax1.bar(x, data[2], width=width, label=names[2])
    p4 = ax1.bar(x + 1 * width, data[3], width=width, label=names[3])
    p5 = ax1.bar(x + 2 * width, data[4], width=width, label=names[4])

    ax1.set_xticks(x)
    ax1.set_xticklabels(xticks_name)

    ax1.set_ylabel("Average Completion Time (Timestep)")
    ax1.set_xlabel("Arrival Rate")
    l1 = ax1.legend([p1,p2,p3,p4], [names[0],names[1],names[2],names[3]], loc='lower right')
    l2 = ax1.legend([p5], [names[4]], loc='upper left')
    gca().add_artist(l1)
    plt.tight_layout()
    plt.savefig(data_dir + 'avg_jct.pdf')
