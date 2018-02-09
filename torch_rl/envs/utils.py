


def wrapped_by(env, cls):
    """
    Checks if the environment is wrapped by a given wrapper
    :param env: The env
    :param cls: The wrapping class
    :return:
    """
    if isinstance(env, cls):
        return True

    while hasattr(env, "env"):
        env = env.env
        if isinstance(env, cls):
            return True