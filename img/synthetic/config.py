from enum import IntEnum

import numpy as np
import torch


# synthetic
class JobType(IntEnum):
    IMG = 0
    LM = 1

job_probs = [1 / 2, 1 / 2]

job_time = {JobType.IMG: 1, JobType.LM: 1}
task_name = {JobType.IMG: "mnist", JobType.LM: "lm"}

# trace
# class JobType(IntEnum):
#     IMG = 0
#     LM = 1
#     AUD = 2

# job_probs = [1 / 3, 1 / 3, 1 / 3]

# job_time = {JobType.IMG: 1, JobType.LM: 1, JobType.AUD: 1}
# task_name = {JobType.IMG: "mnist", JobType.LM: "lm", JobType.AUD: "audio"}

# multi agent
agent_num = 3

# synthetic
arrival_lambda = 2

# trace
# arrival_lambda = 3

# optional resource should be place after the required resource
# `requirement_nonzero_num` is the index of the first optional resource if any
cpu_dim = 0
gpu_dim = 1

# synthetic
job_min_requirement = {JobType.IMG: np.asarray([2, 1], dtype=int),
                       JobType.LM: np.asarray([2, 1], dtype=int),}

switch_weight = {JobType.IMG: 0.5, JobType.LM: 0.5}

discrete_action_dimension = (17, 17)
resource_capacity = (16, 16)

# trace
# job_min_requirement = {JobType.IMG: np.asarray([1, 1], dtype=int),
#                        JobType.LM: np.asarray([1, 1], dtype=int),
#                        JobType.AUD: np.asarray([1, 1], dtype=int), }
#
# switch_weight = {JobType.IMG: 0.5, JobType.LM: 0.5, JobType.AUD: 0.5}

# trace
# discrete_action_dimension = (17, 9)  # 4 cpus / 112 cpus, 1  gpu / 8 gpus
# resource_capacity = (16, 8)

requirement_nonzero_num = 2

img_weight = np.zeros(discrete_action_dimension)
lm_weight = np.zeros(discrete_action_dimension)
lm_weight[..., :] = np.asarray([np.tanh(x / 16) for x in range(lm_weight.shape[-1])])

resource_progress_weight = {JobType.IMG: img_weight,
                            JobType.LM: lm_weight}
resource_progress_base = {JobType.IMG: np.asarray([0.25]),
                          JobType.LM: np.asarray([0])}

# job_min_requirement = {JobType.IMG: np.asarray([1], dtype=int),
#                        JobType.AUD: np.asarray([1], dtype=int)}
#
# resource_progress_weight = {JobType.IMG: np.asarray([0]),
#                             JobType.AUD: np.asarray([0.125])}
#
# resource_progress_base = {JobType.IMG: np.asarray([0.125]),
#                           JobType.AUD: np.asarray([0])}
#
# discrete_action_dimension = (13,)

resource_dim = len(discrete_action_dimension)
one_discrete_action_space = np.prod(discrete_action_dimension)
episode_task_num = 64 * agent_num
pending_num = 10
hidden_dim = 256
preprocess_dim = hidden_dim * pending_num
stacking_dim = 1 + len(JobType) + 2 * resource_dim + 1

input_dim = {'servers': resource_dim + 1,
             'devices': 1 + len(JobType) + resource_dim,
             'running': 1 + len(JobType) + resource_dim,
             'central': 1,
             }

node_types = ['servers', 'devices', 'running', 'central']
edge_types = [
    ('servers', 'allocate1', 'devices'),
    ('devices', 'connect1', 'servers'),
    ('servers', 'allocate2', 'running'),
    ('running', 'connect2', 'servers'),
    ('servers', 'shortcut1', 'central'),
    ('central', 'shortcut2', 'servers'),
]
metadata = (node_types, edge_types)

servers_from_central = np.stack([np.arange(agent_num, dtype=np.int32), np.full(agent_num, 0, dtype=np.int32)])
central_to_servers = np.stack([np.full(agent_num, 0, dtype=np.int32), np.arange(agent_num, dtype=np.int32)])

gnn_heads = 4
num_layers = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
