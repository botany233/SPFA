import subprocess
import time
import sys

def get_time_str() -> str:
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))

def get_gpu_memory_usage(gpu_id:int) -> tuple[int, int, int]:
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', '--format=csv,nounits,noheader', f'--id={gpu_id}'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()
    total, free, used = map(int, output.split(', '))
    return total, free, used

def wait_gpu(gpu_id:int, free_memroy = 24000, check_interval = 60) -> str:
    free = -1
    while(1):
        total, free, used = get_gpu_memory_usage(gpu_id)
        print(f"{get_time_str()}: gpu_{gpu_id}_free_memory = {free}")
        sys.stdout.flush()
        if free < free_memroy:
            time.sleep(check_interval)
        else:
            break
    print("memory is enough! start training!")
    return f"cuda:{gpu_id}"

if __name__ == "__main__":
    device = wait_gpu(0)
    print(device)