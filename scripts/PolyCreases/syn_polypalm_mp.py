import argparse
import multiprocessing
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

from polypalm import PolyPalmCreator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import parse_args_with_config

def task_function(idx: int, fnamelist: list, figcreator: PolyPalmCreator):
    figcreator.set_seed(idx)
    for i, f in enumerate(fnamelist):
        figcreator.draw_by_multi(f, scale=1.0)

        if i % 10 == 0:
            print(f'{idx} --> {f} saved!')


def build_parser():
    parser = argparse.ArgumentParser(description="PolyCreases multiprocess line / crease map synthesis.")
    parser.add_argument("--ids", type=int, default=2000, help="Number of identity indices to generate.")
    parser.add_argument("--output", type=str, default="outputs/line", help="Output directory for PNG crease maps.")
    parser.add_argument("--nproc", type=int, default=8, help="Worker processes.")
    return parser


if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间

    parser = build_parser()
    args = parse_args_with_config(parser)

    total_ids = args.ids
    start = 0
    num_processes = args.nproc

    step = total_ids // num_processes

    imsize = 256
    noise_weight = 0.3          
    draw_thickness = random.randint(20, 25)     
    is_prolong = False          
    curve_1_is_cut = False      
    curve_2_is_cut = False      
    fig = PolyPalmCreator(imsize=imsize, noise_weight=noise_weight, draw_thickness=draw_thickness)

    labeled_pkl = Path(__file__).resolve().parent / "labeled_data.pkl"
    fig.fit(str(labeled_pkl))

    results_dir = args.output
    os.makedirs(results_dir, exist_ok=True)
    shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    tasks = []
    for i in range(num_processes):
        fnamelist = []
        if i != num_processes - 1:
            for j in range(start + i * step, start + (i + 1) * step):
                fnamelist.append(os.path.join(results_dir, f'{j}_0.png'))
        else:
            for j in range(start + i * step, start + total_ids):
                fnamelist.append(os.path.join(results_dir, f'{j}_0.png'))

        tasks.append((i, deepcopy(fnamelist), deepcopy(fig)))

    pool = multiprocessing.Pool(processes=num_processes)

    pool.starmap(task_function, tasks)

    pool.close()
    pool.join()

    print("total {} images".format(len(os.listdir(results_dir))))
    end_time = time.time()  
    execution_time = end_time - start_time  
    print("Total execution time:", execution_time, "seconds") 
