

def _test():

    from torch_rl import config
    print(config.tensorboard_path())
    print(config.benchmark_path())


if __name__ == '__main__':

    _test()
