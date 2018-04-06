import glob
import subprocess
import os
import sys



if __name__ == '__main__':


    assert 'TRL_DATA_PATH' in os.environ, 'TRL_DATA_PATH has to be set in environment'

    work_dir = os.environ['TRL_DATA_PATH']

    # Regex for files to run tensorboard with
    regexes = sys.argv[1:]


    paths = []
    logdir_string='--logdir='
    for regex in regexes:
        dirs = glob.glob(work_dir+'/'+regex)
        paths.extend([os.path.abspath(x) for x in dirs])

    print(paths)

    for path in paths:
        dirname = path.split("/")[-1]
        logdir_string+= "{}:{},".format(dirname, os.path.join(path, 'tensorboard'))


    print(logdir_string)
    subprocess.call(['tensorboard', logdir_string[:-1]])
