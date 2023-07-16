import requests
from urllib.parse import urljoin
import json

f = open("portainer/token.json")
tmp = json.load(f)
token = tmp["token"]
url = tmp["url"]
f.close()

list_docker = "docker/containers/json"
default_headers = {"X-API-Key": token}
create_docker = "docker/containers/create"

scheduler_name = "default"
tasks = {}
nodes = [2, 3, 5]
label_key = "yihong.scheduler"

def start_portainer_tracker(scheduler):
    global scheduler_name
    global tasks
    scheduler_name = scheduler
    task = {}

def update_tracker():
    for n in nodes:
        req = urljoin(url, "{}/".format(n))
        req = urljoin(req, list_docker)
        r = requests.get(req, headers=default_headers, verify=False, params={"all": True, "filters": json.dumps({"label": [label_key+"="+scheduler_name]})})
        for c in r.json():
            inspect_docker = "docker/containers/{}/json".format(c["Id"])
            req = urljoin(url, "{}/".format(n))
            req = urljoin(req, inspect_docker)
            r = requests.get(req, headers=default_headers, verify=False)
            start_time = r.json()["State"]["StartedAt"]
            end_time = r.json()["State"]["FinishedAt"]
            tasks[c["Names"][0]] = (start_time, end_time)
    with open("res.json", "w") as f:    
        json.dump(tasks, f)


def all_exit():
    for n in nodes:
        req = urljoin(url, "{}/".format(n))
        req = urljoin(req, list_docker)
        r = requests.get(req, headers=default_headers, verify=False, params={"all": True, "filters": json.dumps({"label": [label_key]})})
        for c in r.json():
            if c["State"] != "exited":
                return False
    return True


def start_task(name, cpu_num, gpu_list, node, task_type):
    params = {"name": name}
    node_url = "{}/".format(node)
    req = urljoin(url, node_url)
    req = urljoin(req, create_docker)
    data = {
        "Labels": {
        "yihong.scheduler": "{}".format(scheduler_name),
        },
        "Cmd": [
            "/root/miniconda3/bin/python",
            "/workspace/{}.py".format(task_type),
            "--sfdir",
            "/workspace/",
            "--tracedir",
            "/workspace/",
            "--cpus",
            "{}".format(cpu_num),
            "--gpus",
            "{}".format(len(gpu_list)),
            "--times",
            "1",
        ],
        "Entrypoint": "",
        "Image": "192.168.26.85:5000/yihong-base:latest",
        "WorkingDir": "/workspace",
        "HostConfig": {
            "NanoCpus": 4000000000,
            "DeviceRequests": [],
            "DeviceRequests": [
                {
                    "Driver": "nvidia",
                    "DeviceIDs": gpu_list,
                    "Capabilities": [
                        [
                            "utility",
                            "compute"
                        ]
                    ],
                }
            ],
            "ShmSize": 2048000000
        },
    }
    r = requests.post(req, headers=default_headers, verify=False, params=params, json=data)
    resp = r.json()

    start_docker = "docker/containers/{}/start".format(resp["Id"])
    req = urljoin(url, node_url)
    req = urljoin(req, start_docker)
    r = requests.post(req, headers=default_headers, verify=False)
    print(r.status_code)