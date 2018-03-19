from collections import deque
import pandas as pd
import datetime
import os
import numpy as np
from torch_rl.utils import Parameters, prRed, Callback
import glob
import shutil

class TrainingStatsCallback(Callback):
    """
        Keeps training statistics and writes them to a file and loads them.
    """
    def __init__(self, episode_window=10, step_window=10,
                 sample_rate_episodes=1, sample_rate_steps=None, save_rate=10,
                 save_destination=None, hyperparameters=None):

        self.episode_window = episode_window
        self.episode_reward_buffer = deque(maxlen=episode_window)
        self.step_rewards = []
        self.episode_rewards = []
        self.sample_rate_episodes=sample_rate_episodes
        self.sample_rate_steps = sample_rate_steps
        self.rewards = []
        self.moving_average_rewards = []
        self.save_rate = save_rate
        self.hyperparameters = hyperparameters

        if save_destination is None:
            self.save_destination = 'training_stats_' + str(datetime.datetime.now())
        else:
            self.save_destination = save_destination

        if os.path.isdir(self.save_destination):
            prRed(self.save_destination + " is a directory already, delete for new training data? y/n")
            res = input()
            res = res.lower()
            if res == 'y':
                shutil.rmtree(self.save_destination)
            else:
                raise Exception("Start training with another save destination name.")
        os.makedirs(self.save_destination)


        # Save hyperparameters to a file
        self.save_hyperparameters()

        # Pandas data frames
        self.episode_data = None
        self.step_data = None

    def save_hyperparameters(self):
        if self.hyperparameters:
            if not isinstance(self.hyperparameters, Parameters):
                raise Exception("User Parameters from torch_rl.utils.Parameters to store parameters")
            df = pd.DataFrame.from_records(self.hyperparameters.__dict__)
            df.to_pickle(os.path.join(self.save_destination, "parameters.cfg"))

    def step(self, episode, step, reward,**kwargs):
        kwargs["reward"] = reward
        kwargs['episode'] = episode
        kwargs['step'] = step

        df = pd.DataFrame.from_records([kwargs], index=['step'])

        if not self.step_data is None:
            self.step_data = pd.concat([self.step_data, df])
        else:
            self.step_data = df

        if episode % self.save_rate == 0:
            self.save()

    def episode_step(self, **kwargs):
        self.episode_reward_buffer.append(kwargs['episode_reward'])
        episode = kwargs['episode']
        kwargs["mvavg_reward"] = np.mean(self.episode_reward_buffer)
        df = pd.DataFrame.from_records([kwargs], index=['episode'])

        if not self.episode_data is None:
            self.episode_data = pd.concat([self.episode_data, df])
        else:
            self.episode_data = df

        if episode % self.save_rate == 0:
            self.save()

    def save(self):

        time_stamp = str(datetime.datetime.now())
        path = self.save_destination

        print(" #Saving data to", path)

        if not self.episode_data is None:
            name = time_stamp + "_episode.stats"
            self.episode_data.to_pickle(os.path.join(path, name))
        if not self.step_data is None:
            name = time_stamp + "_step.stats"
            self.step_data.to_pickle(os.path.join(path, name))

        self.step_data = None
        self.episode_data = None

    @staticmethod
    def load(path="./training_stats"):
        files = glob.glob(path + "/*.stats")
        files = sorted(files)
        data = pd.read_pickle(os.path.join(path, files[0]))

        if len(files) > 1:
            for f in files[1:]:
                d = pd.read_pickle(os.path.join(path, f))
                data = pd.concat([data, d])

        return data

    @staticmethod
    def load_hyperparameters(path="./training_stats"):
        files = glob.glob(path + "/*.cfg")
        files = sorted(files)
        data = pd.read_pickle(os.path.join(path, files[0]))

        return data
