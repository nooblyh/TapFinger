import requests
from urllib.parse import urljoin
import json

from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

f = open("portainer/token.json")
tmp = json.load(f)
token = tmp["token"]
url = tmp["url"]
f.close()

list_docker = "docker/containers/json"
default_headers = {"X-API-Key": token}
create_docker = "docker/containers/create"

scheduler_name = "default"
test_name = ""
tasks = {}
nodes = [3, 6, 5]
label_key = "yihong.scheduler"

def start_portainer_tracker(scheduler, test):
    global scheduler_name
    global test_name
    global tasks
    scheduler_name = scheduler
    test_name = test
    tasks = {}

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
            content = r.json()
            start_time = content["State"]["StartedAt"]
            end_time = content["State"]["FinishedAt"]
            tasks[c["Names"][0]] = (start_time, end_time)
    with open("img/{}/{}_start_exited.json".format(test_name, scheduler_name), "w") as f:    
        json.dump(tasks, f)

def delete_all():
    for n in nodes:
        req = urljoin(url, "{}/".format(n))
        req = urljoin(req, list_docker)
        r = requests.get(req, headers=default_headers, verify=False, params={"all": True, "filters": json.dumps({"label": [label_key]})})
        for c in r.json():
            inspect_docker = "docker/containers/{}".format(c["Id"])
            req = urljoin(url, "{}/".format(n))
            req = urljoin(req, inspect_docker)
            r = requests.delete(req, headers=default_headers, verify=False)
            print(r.status_code)

def all_exit():
    for n in nodes:
        req = urljoin(url, "{}/".format(n))
        req = urljoin(req, list_docker)
        r = requests.get(req, headers=default_headers, verify=False, params={"all": True, "filters": json.dumps({"label": [label_key]})})
        for c in r.json():
            if c["State"] != "exited":
                return False
    return True

def is_exit_without_error(id, node):
    inspect_docker = "docker/containers/{}/json".format(id)
    req = urljoin(url, "{}/".format(node))
    req = urljoin(req, inspect_docker)
    r = requests.get(req, headers=default_headers, verify=False)
    content = r.json()
    assert content["State"]["Error"] == ""
    return content["State"]["Status"] == 'exited'

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
    print(resp)
    start_docker = "docker/containers/{}/start".format(resp["Id"])
    req = urljoin(url, node_url)
    req = urljoin(req, start_docker)
    r = requests.post(req, headers=default_headers, verify=False, json={})
    print(r.content)
    return resp["Id"]

delete_all()
