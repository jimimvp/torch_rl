from torch_rl.training.core import HorizonTrainer, mse_loss
from torch_rl.memory import GeneralisedlMemory
from torch.optim import Adam
from torch_rl.utils import to_tensor as tt
import torch as tor
from collections import deque
from torch_rl.utils import prGreen
import time
import sys
from torch_rl.utils import logger

def queue_to_array(q):
    q.put(False)
    arr = []
    while True:
        item = q.get()
        if item:
            arr.append(item)
        else:
            break

    return np.asarray(arr)




class AdvantageEstimator(object):

    mvavg_rewards = deque(maxlen=100)

    def __init__(self, env, network, nsteps, gamma, lam):
        self.env = env
        self.network = network
        nenv = 1
        self.obs = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.state = None
        self.done = False
        self.acc_reward = 0
        self.global_step = 0

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logpacs = [], [], [], [], [], []
        mb_state = self.state
        epinfos = []
        acc_reward = 0

        for _ in range(self.nsteps):
            actions, values = self.network(tt(self.obs, cuda=False).view(1,-1))
            logpacs = -self.network.logprob(actions)

            mb_obs.append(self.obs.copy().reshape(-1))
            mb_actions.append(actions.data.numpy().reshape(-1))
            mb_values.append(values.detach().data.numpy().reshape(-1))
            mb_logpacs.append(logpacs.data.numpy().reshape(-1))

            mb_dones.append(self.done)

            obs, reward, self.done, infos = self.env.step(actions.data.numpy())
            self.obs = obs
            acc_reward += reward
            self.global_step += 1
            mb_rewards.append(reward)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    print(maybeepinfo)
                    self.mvavg_rewards.append(maybeepinfo['r'])
                    epinfos.append(maybeepinfo)

            if self.done:
                self.mvavg_rewards.append(acc_reward)
                acc_reward = 0
                self.obs = self.env.reset()



        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).reshape(self.nsteps, -1)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape(self.nsteps, -1)
        mb_actions = np.asarray(mb_actions, dtype=np.float32).reshape(self.nsteps, -1)
        mb_values = np.asarray(mb_values, dtype=np.float32).reshape(self.nsteps, -1)
        mb_logpacs = np.asarray(mb_logpacs, dtype=np.float32).reshape(self.nsteps, -1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).reshape(self.nsteps, -1)

        action, last_values = self.network(tt(self.obs.reshape(1,-1), cuda=False))
        action, last_values = action.data.numpy().reshape(-1), last_values.data.numpy().reshape(-1)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.done
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        def sf01(arr):
            """
            swap and then flatten axes 0 and 1
            """
            s = arr.shape
            return arr

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_logpacs)),
                mb_state, epinfos)

        # obs, returns, masks, actions, values, neglogpacs, states = runner.run()

    def constfn(val):
        def f(_):
            return val

        return f


class GPUPPO(HorizonTrainer):

    mvavg_reward = deque(maxlen=100)


    def __init__(self, env, network, network_old, max_episode_len=500, gamma=.99,
                 replay_memory=GeneralisedlMemory(12000, window_length=1), lr=3e-4, n_steps=40,
                 epsilon=0.2, optimizer=None, lmda=0.95, ent_coeff=0., n_update_steps=10, num_threads=5, n_minibatches=1):
        super(GPUPPO, self).__init__(env)

        self.n_minibatches = n_minibatches
        self.lr = lr
        self.replay_memory = replay_memory
        self.max_episode_len = max_episode_len
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmda = lmda
        self.optimizer = Adam(network.parameters(), lr=lr) if optimizer is None else optimizer
        self.goal_based = hasattr(env, "goal")
        self.network = network
        self.network_old = network_old
        self.ent_coeff = ent_coeff
        self.num_episodes = 0
        self.num_threads = num_threads
        self.sigma_log = -0.7
        self.T = 30
        self.epinfobuf = deque(maxlen=100)
        self.n_update_steps = n_update_steps
        self.n_steps = n_steps
        self.advantage_estimator = AdvantageEstimator(env, self.network, n_steps, self.gamma, self.lmda)

        print(self.network)

    def _horizon_step(self):


        obs, returns, masks, actions, values, logpacs, states, epinfos = self.advantage_estimator.run() #pylint: disable=E0632
        self.epinfobuf.extend(epinfos)
        nbatch_train = self.n_steps // self.n_minibatches


        self.optimizer = Adam(self.network.parameters(), self.lr)
        self.network.cuda()
        if states is None: # nonrecurrent version
            inds = np.arange(self.n_steps)
            for _ in range(self.n_update_steps):
                np.random.shuffle(inds)
                for start in range(0, self.n_steps, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    bobs, breturns, bmasks, bactions, bvalues, blogpacs = map(lambda arr: arr[mbinds], (obs, returns, masks, actions, values, logpacs))
                    advs = breturns - bvalues
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                    BATCH = tt(bobs)
                    A = tt(bactions)
                    ADV = tt(advs)
                    R = tt(breturns)
                    OLDLOGPAC = tt(blogpacs)
                    OLDVPRED = tt(bvalues)

                    self.network(BATCH)
                    logpac = self.network.logprob(A)
                    entropy = tor.mean(self.network.entropy())

                    #### Value function loss ####
                    #print(bobs)
                    actions_new, v_pred = self.network(tt(bobs))
                    v_pred_clipped = OLDVPRED + tor.clamp(v_pred - OLDVPRED, -self.epsilon, self.epsilon)
                    v_loss1 = (v_pred - R)**2
                    v_loss2 = (v_pred_clipped - R)**2

                    v_loss = .5 * tor.mean(tor.max(v_loss1, v_loss2))

                    ### Ratio calculation ####
                    # In the baselines implementation these are negative logits, then it is flipped
                    ratio = tor.exp(logpac - OLDLOGPAC)

                    ### Policy gradient calculation ###
                    pg_loss1 = -ADV * ratio
                    pg_loss2 = -ADV * tor.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon)
                    pg_loss = tor.mean(tor.max(pg_loss1, pg_loss2))
                    approxkl = .5 * tor.mean((logpac - OLDLOGPAC)**2)


                    loss = v_loss  + pg_loss

                    #clipfrac = tor.mean((tor.abs(ratio - 1.0) > self.epsilon).type(tor.FloatTensor))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


        #Push to CPU
        self.network.cpu()
        logger.logkv("mvavg_reward", np.mean(self.advantage_estimator.mvavg_rewards))
        logger.logkv("siglog", np.mean(self.network.siglog.data.numpy()))
        logger.logkv("pgloss", pg_loss.cpu().data.numpy())
        logger.logkv("vfloss", v_loss.cpu().data.numpy())
        logger.dumpkvs()



if __name__ == '__main__':

    from torch_rl.envs.wrappers import *
    import gym
    from gym.wrappers import Monitor
    from torch_rl.models.ppo import ActorCriticPPO
    from torch_rl.utils import *
    from torch_rl.utils import logger
    import sys


    monitor = Monitor(NormalisedActionsWrapper(gym.make("Pendulum-v0")), directory="./stats", force=True, 
        video_callable=False, write_upon_reset=True)
    env = BaselinesNormalize(monitor)
    logger.configure()
    print(env.observation_space.shape)


    with tor.cuda.device(1):
        network = ActorCriticPPO([env.observation_space.shape[0], 64, 64, env.action_space.shape[0]])
        network_old = ActorCriticPPO([env.observation_space.shape[0], 64, 64, env.action_space.shape[0]])
        network.apply(gauss_init(0, np.sqrt(2)))

        trainer = GPUPPO(network=network, network_old=network_old, env=env, n_update_steps=4, n_steps=40)
        trainer.train(horizon=100000, max_episode_len=500)




