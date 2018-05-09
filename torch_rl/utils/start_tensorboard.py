import glob
import subprocess
import os
import sys

from torch_rl.utils import init_parser, addarg, cmdl_args
from datetime import date
import time
import argparse

if __name__ == '__main__':

    description =   ("Starts tensorboard on all of the subdirectories in TRL_DATA_PATH unless"
                    " specified otherwise. The directory structure should be as follows:"
                    " TRL_DATA_PATH/EXPERIMENT_DIR/tensorboard, every directory should contain"
                    " a tensorboard directory with tensorboard logs.")

    init_parser(description)
    addarg('port', type=int, default=6006, info="port on which to start tensorboard")
    addarg('work-dir', type=str, default=os.environ['TRL_DATA_PATH'], info="path to directory that contains the data")
    addarg('reg', type=str, default=['*'], info="regex expressions to search for")
    addarg('delta', type=str, default=None, info="modified since DELTA something ago. [d|m|h]NUM")
    addarg('dry', type=bool, default=False, info="dry run")
    addarg('time-info', type=bool, default=False, info="add timestamps to logdir labels")



    def mtime_check(path, ds):
        t = os.path.getmtime(path)
        d1 = date.fromtimestamp(int(t))
        d2 = date.fromtimestamp(int(time.time()))
        delta = d2 - d1
        num = int(ds[1:])
        if ds[0] == 'm':
            delta = delta.minutes
        elif ds[0] == 'h':
            delta = delta.hours
        elif ds[0] == 'd':
            delta = delta.days
        return delta <= num

    assert 'TRL_DATA_PATH' in os.environ, 'TRL_DATA_PATH has to be set in environment'
    p = cmdl_args()
    work_dir = p.work_dir
    # Regex for files to run tensorboard with
    regexes = p.reg


    paths = []
    logdir_string='--logdir='
    for regex in regexes:
        dirs = glob.glob(work_dir+'/'+regex)
        paths.extend([os.path.abspath(x) for x in dirs])
    
    if p.delta:
        paths = list(filter(lambda x: mtime_check(x, p.delta), paths))

    print("Num files:", len(paths))
    if p.dry:
        sys.exit(1)

    for path in paths:
        dirname = path.split("/")[-1]
        date_prefix = ""
        if p.time_info:
            date_prefix = date.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%d-%m")
        if p.time_info:
            logdir_string+= "{}{}:{},".format(date_prefix, dirname, os.path.join(path, 'tensorboard'))
        else:
            logdir_string+= "{}:{},".format(dirname, os.path.join(path, 'tensorboard'))

    port_str = '--port='+str(p.port)

    print(logdir_string)
    subprocess.call(['tensorboard', logdir_string[:-1], port_str])
