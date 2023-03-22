import numpy as np

from utils import config


class DeviceNode(object):
    def __init__(self, device_id, job_type: config.JobType):
        super().__init__()
        self.device_id = device_id
        self.job_type = job_type
        self.progress = 0.0
        self.drl_index = np.full(config.agent_num, -1)
        self.gnn_index = -1
        self.time = 0
        self.running_time = 0
        self.attained_gpu_service = 0
        self.start_time = -1
        self.arrive_time = 0
