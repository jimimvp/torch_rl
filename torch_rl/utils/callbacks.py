from .utils import Callback, timestamp
import os

class CheckpointCallback(Callback):


    def __init__(self, models=None, save_path=".", interval=100, episodewise=True, dt=1):
        """
        Init the object
        :param models:      dict of name: model pairs, the names are used for saving
                            the models.
        :param save_path:   directory where all of the checkpoints are going to be stored in
                            a tree hierarchy.
        """
        super(CheckpointCallback, self).__init__(episodewise=episodewise, stepwise=not episodewise)
        self.models = models
        self.interval = interval
        self.save_path = os.path.join(save_path, "checkpoints")
        self.dt = dt

        os.makedirs(save_path, exist_ok=True)

        for k in self.models.keys():
            model_dir = os.path.join(self.save_path, k)
            os.makedirs(model_dir)

    def _step(self,  *args, **kwargs):

        step = kwargs['step']
        if step % self.interval == 0:
            for name, model in self.models.items():
                model.save(path=os.path.join(self.save_path, name, "{}_model.ckpt".format(timestamp())))


    def episode_step(self, *args, **kwargs):

        step = kwargs['episode']
        if step % self.interval == 0:
            for name, model in self.models.items():
                model.save(path=os.path.join(self.save_path, name,  "{}_model.ckpt".format(timestamp())))













