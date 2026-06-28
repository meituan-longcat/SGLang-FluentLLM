#!/usr/bin/env python3
import os
import subprocess
import psutil
import re
import sys


_NPU_910B_NUM_GPUS_PER_NODE = int(os.popen("npu-smi info | grep 910B | wc -l").read().strip())

def get_sglang_processes():
    pid_dict = {}
    none_counter = 0
    try: # 获取所有进程
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try: # 检查进程名或命令行参数
                proc_info = proc.info
                cmdline = ' '.join(proc_info['cmdline']) if proc_info['cmdline'] else ''
                pattern = r'sglang::scheduler_(\d+|None|none)'
                match = re.search(pattern, cmdline)
                if match:
                    scheduler_id = len(pid_dict)
                    pid = proc_info['pid']
                    pid_dict[scheduler_id] = pid
                    print(f"找到进程: sglang::scheduler_{scheduler_id}, PID: {pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue # 跳过无法访问的进程
    except Exception as e:
        print(f"获取进程列表时出错: {e}")
        return {}
    return pid_dict

def calculate_core_ranges(total_cores=192, num_processes=16, cores_per_process=12):
    """
    计算每个进程应该绑定的 CPU 核心范围
    返回格式: {scheduler_id: (start_core, end_core)}
    """
    if _NPU_910B_NUM_GPUS_PER_NODE == 0:
        return {
            0: [8, 39], 1: [40, 71], 2: [72, 103], 3: [104, 135], 4: [160, 191], 5: [192, 223], 6: [224, 255],7: [256, 287],
            8: [320, 351], 9: [352, 383], 10: [384, 415], 11: [416, 447], 12: [480, 511], 13: [512, 543], 14: [544, 575], 15: [576, 607],
        }
    core_ranges = {}
    for i in range(num_processes):
        start_core = i * cores_per_process
        end_core = start_core + cores_per_process - 1
        if end_core >= total_cores:
            print(f"警告: 进程 {i} 的核心范围超出系统限制")
            end_core = total_cores - 1
        core_ranges[i] = (start_core, end_core)
    return core_ranges

def set_cpu_affinity_taskset(pid, start_core, end_core):
    """
    使用 taskset 命令设置进程的 CPU 亲和性
    """
    try:
        cmd = ['taskset', '-cp', f'{start_core}-{end_core}', str(pid)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except Exception as e:
        return False, str(e)

def set_cpu_affinity_python(pid, start_core, end_core):
    """
    使用 Python 的 os.sched_setaffinity 设置 CPU 亲和性
    """
    try:
        # 创建核心列表
        cpu_cores = list(range(start_core, end_core + 1))

        # 设置 CPU 亲和性
        os.sched_setaffinity(pid, cpu_cores)
        return True, f"成功设置 PID {pid} 到核心 {start_core}-{end_core}"
    except Exception as e:
        return False, str(e)

def verify_cpu_affinity(pid):
    """
    验证进程的 CPU 亲和性设置
    """
    try:
        cmd = ['taskset', '-p', str(pid)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except Exception as e:
        return False, str(e)

def main():
    print("=== SGLang Scheduler 进程 CPU 亲和性设置工具 ===\n")
    if len(sys.argv) > 1:
        rank = int(sys.argv[1])
        print("decode node rank:", rank)
    else:
        print("请输入decode node rank")
        return None

    # 1. 获取所有 sglang::scheduler 进程
    print("1. 获取 sglang::scheduler 进程...")
    pid_dict = get_sglang_processes()

    if not pid_dict:
        print("未找到任何 sglang::scheduler 进程")
        return
    print(f"共找到 {len(pid_dict)} 个进程\n")

    # 创建 PID 列表（按 scheduler_id 排序）
    pid_list = []
    for scheduler_id in sorted(pid_dict.keys()):
        pid_list.append(pid_dict[scheduler_id])
    print(f"PID 列表: {pid_list=}\n {pid_dict=}\n")

    # 2. 计算核心范围
    print("2. 计算 CPU 核心分配...")
    core_ranges = calculate_core_ranges()
    for scheduler_id, (start_core, end_core) in core_ranges.items():
        # scheduler_id = scheduler_id + 16 * rank
        if scheduler_id in pid_dict:
            print(f"scheduler{scheduler_id} (PID: {pid_dict[scheduler_id]}) -> 核心 {start_core}-{end_core}")

    # 3. 设置 CPU 亲和性
    print("3. 设置 CPU 亲和性...")
    success_count = 0
    for scheduler_id in sorted(pid_dict.keys()):
        # if scheduler_id >= 16:  # 只处理 scheduler0-15
        #     continue

        pid = pid_dict[scheduler_id]
        # start_core, end_core = core_ranges[scheduler_id - 16 * rank]
        start_core, end_core = core_ranges[scheduler_id]

        print(f"设置 scheduler{scheduler_id} (PID: {pid}) 到核心 {start_core}-{end_core}...")
        # 尝试使用 taskset 命令
        success, message = set_cpu_affinity_taskset(pid, start_core, end_core)
        if success:
            print(f"  ✓ 成功: {message}")
            success_count += 1
        else:
            print(f"  ✗ 失败: {message}")
            # 如果 taskset 失败，尝试使用 Python 方法
            print(f"  尝试使用 Python 方法...")
            success2, message2 = set_cpu_affinity_python(pid, start_core, end_core)
            if success2:
                print(f"  ✓ Python 方法成功: {message2}")
                success_count += 1
            else:
                print(f"  ✗ Python 方法也失败: {message2}")
    print(f"\n成功设置 {success_count}/{len([k for k in pid_dict.keys() if k < 16])} 个进程的 CPU 亲和性\n")

    # 4. 验证设置
    print("4. 验证 CPU 亲和性设置...")
    for scheduler_id in sorted(pid_dict.keys()):
        # if scheduler_id >= 16:
        #     continue
        pid = pid_dict[scheduler_id]
        success, message = verify_cpu_affinity(pid)
        if success:
            print(f"scheduler{scheduler_id} (PID: {pid}): {message}")
        else:
            print(f"scheduler{scheduler_id} (PID: {pid}): 验证失败 - {message}")

if __name__ == "__main__":
    # 检查是否有足够的权限
    if os.geteuid() != 0:
        print("警告: 建议以 root 权限运行此脚本以确保能够设置所有进程的 CPU 亲和性")
        print("可以使用: sudo python3 script.py\n")

    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        sys.exit(1)

