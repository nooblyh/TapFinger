import subprocess
import time

# check_cmd = ['docker', 'ps', '-q', '|', 'wc', '-l']
# run_cmd = ['docker', 'run', '--gpus' '', '--cpus', '', '--rm', '-v', '`pwd`:/mnt', '--network=host', 'gpu-env', 'python', '', '', '']

run_cmd = ['python', '', '--sfdir', '/home/zhangxx/lyh/rl_demo/trace/', '--tracedir', '/home/zhangxx/lyh/rl_demo/trace/',
           '--cpus', '', '--gpus', '']


def check_status(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True)
        counter = 0
        while (output == '' or output == None):
            output = subprocess.check_output(cmd, shell=True).decode('utf-8')
            print(output)
            if output == 0:
                return
            time.sleep(10)
            counter = counter + 1
            if counter > 60:
                raise Exception("Error")
    except Exception as e:
        print(e)


def main():
    cpus = 16
    gpus = 8
    tasks = ["mnist", "lm", "audio"]
    for task_name in tasks:
        for cpu_n in range(1, cpus + 1):
            for gpu_n in range(1, gpus + 1):
                # run_cmd[4] = str(gpu_n)
                # run_cmd[6] = str(cpu_n)
                # run_cmd[-1] = str(gpu_n)
                # run_cmd[-2] = str(cpu_n)
                # run_cmd[-3] = str(task_name+".py")
                run_cmd[1] = str(task_name + ".py")
                run_cmd[-1] = str(gpu_n)
                run_cmd[-3] = str(cpu_n)
                try:
                    print(run_cmd)
                    retcode = subprocess.call(run_cmd)
                    if retcode < 0:
                        raise Exception("subprocess call Error retcode " + str(retcode))
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    main()
