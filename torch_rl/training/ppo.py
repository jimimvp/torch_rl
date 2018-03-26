from torch_rl.training.core import HorizonTrainer, mse_loss
from torch_rl.memory import GeneralisedlMemory
from torch.optim import Adam
from torch_rl.utils import to_tensor
import torch as tor
from collections import deque
from torch_rl.utils import prGreen
import time
import sys

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
        self.states = None
        self.dones = [False for _ in range(nenv)]
        self.acc_reward = 0
        self.global_step = 0

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values = self.network(to_tensor(self.obs).view(1,-1))
            neglogpacs = self.network.logprob(actions)

            mb_obs.append(self.obs.copy().reshape(-1))
            mb_actions.append(actions.data.numpy().reshape(-1))
            mb_values.append(values.detach().data.numpy().reshape(-1))
            mb_neglogpacs.append(neglogpacs.data.numpy().reshape(-1))

            mb_dones.append(self.dones)

            obs, rewards, self.dones, infos = self.env.step(actions.data.numpy())
            self.dones = self.dones[0]
            obs = obs[0]
            rewards = rewards[0]
            self.obs = obs
            self.acc_reward += rewards
            self.global_step += 1
            mb_rewards.append(rewards)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    print(maybeepinfo)
                    self.mvavg_rewards.append(maybeepinfo['r'])
                    epinfos.append(maybeepinfo)

            if self.dones or self.global_step % 500 == 0:
                self.acc_reward = 0
                self.obs = self.env.reset()


        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).reshape(self.nsteps, -1)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape(self.nsteps, -1)
        mb_actions = np.asarray(mb_actions, dtype=np.float32).reshape(self.nsteps, -1)
        mb_values = np.asarray(mb_values, dtype=np.float32).reshape(self.nsteps, -1)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32).reshape(self.nsteps, -1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).reshape(self.nsteps, -1)

        action, last_values = self.network(to_tensor(self.obs.reshape(1,-1)))
        action, last_values = action.data.numpy().reshape(-1), last_values.data.numpy().reshape(-1)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
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

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)

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


        obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.advantage_estimator.run() #pylint: disable=E0632
        self.epinfobuf.extend(epinfos)
        nbatch_train = self.n_steps // self.n_minibatches

        self.optimizer = Adam(self.network.parameters(), self.lr)
        if states is None: # nonrecurrent version
            inds = np.arange(self.n_steps)
            for _ in range(self.n_update_steps):
                np.random.shuffle(inds)
                for start in range(0, self.n_steps, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    bobs, breturns, bmasks, bactions, bvalues, bneglogpacs = map(lambda arr: arr[mbinds], (obs, returns, masks, actions, values, neglogpacs))
                    advs = breturns - bvalues
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                    A = to_tensor(bactions)
                    ADV = to_tensor(advs)
                    R = to_tensor(breturns)
                    OLDNEGLOGPAC = to_tensor(bneglogpacs)
                    OLDVPRED = to_tensor(bvalues)

                    neglogpac = self.network.logprob(A)
                    entropy = tor.mean(self.network.entropy())

                    #### Value function loss ####
                    #print(bobs)
                    actions_new, v_pred = self.network(to_tensor(bobs))
                    v_pred_clipped = tor.clamp(v_pred - OLDVPRED, -self.epsilon, self.epsilon)
                    v_loss1 = (v_pred- R)**2
                    v_loss2 = (v_pred_clipped - R)**2

                    v_loss = .5 * tor.mean(tor.max(tor.cat((v_loss1, v_loss2), dim=1), dim=1)[0])

                    ### Ratio calculation #### d
                    ratio = tor.exp(OLDNEGLOGPAC - neglogpac)

                    ### Policy gradient calculation ###
                    pg_loss1 = -ADV * ratio
                    pg_loss2 = -ADV * tor.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon)
                    pg_loss = tor.mean(tor.max(tor.cat((pg_loss1, pg_loss2), dim=1), dim=1)[0])
                    approxkl = .5 * tor.mean((neglogpac - OLDNEGLOGPAC)**2)


                    loss = v_loss  + pg_loss

                    #clipfrac = tor.mean((tor.abs(ratio - 1.0) > self.epsilon).type(tor.FloatTensor))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def __init__(self, env, network, network_old, num_episodes=2000, max_episode_len=500, batch_size=64, gamma=.99,
                 replay_memory=GeneralisedlMemory(12000, window_length=1), lr=1e-3, memory_fill_steps=2048,
                 epsilon=0.2, optimizer=None, lmda=0.95, ent_coeff=0., policy_update_epochs=10, num_threads=5):
        super(DPPOTrainer, self).__init__(env)
        self.lr = lr
        self.batch_size = batch_size
        self.replay_memory = replay_memory
        self.max_episode_len = max_episode_len
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmda = lmda
        self.optimizer = Adam(network.parameters(), lr=lr) if optimizer is None else optimizer
        self.goal_based = hasattr(env, "goal")
        self.memory_fill_steps = memory_fill_steps
        self.network = network
        self.network_old = network_old
        self.policy_update_epochs = policy_update_epochs
        self.ent_coeff = ent_coeff
        self.num_episodes = 0
        self.num_threads = num_threads
        self.sigma_log = -0.7
        self.T = 30
        self.reward_std = deque(maxlen=100)

        # Convenience buffers for faster iteration over advantage calculation
    def add_to_replay_memory(self,s,a,r,d):
        if self.goal_based:
            self.replay_memory.append(self.state, self.env.goal, a, r, d, training=True)
        else:
            self.replay_memory.append(self.state, a, r, d, training=True)




    def gather_experience_and_calc_advantages(self, network, q, reward_q, max_episode_len, gamma, lmda, env, step, pid, T, sigma_log):
        np.random.seed(pid)
        tor.manual_seed(pid)
        env.seed(pid)
        state = env.reset()
        episode_length = 0
        acc_reward = 0
        adv_states = np.zeros((T, env.observation_space.shape[0]))
        adv_actions = np.zeros((T, env.action_space.shape[0]))
        adv_rewards = np.zeros(T)
        adv_values = np.zeros(T)
        adv_returns = np.zeros(T)
        adv_advantages = np.zeros(T)
        while True:

            counter = 0
            for i in range(T):
                counter += 1
                adv_states[i] = state
                action, value = network(to_tensor(state).view(1,-1), sigma_log)
                adv_actions[i] = action.data.numpy()
                adv_values[i] = value.data.numpy()

                env_action = action.data.squeeze().numpy()
                state, reward, done, _ = env.step(np.clip(env_action, -1, 1))
                # reward /= 100.
                # print(reward)

                self._async_step(state=state, reward=reward)

                episode_length += 1
                acc_reward += reward
                done = done or episode_length >= max_episode_len

                # step(state, env_action, reward, done)

                #reward = max(min(reward, 1), -1)
                adv_rewards[i] = reward

                R = np.zeros((1,1))
                if not done:
                    R = value.data.numpy()
                adv_values[i] = R

                if done:
                    state = env.reset()
                    episode_length = 0
                    reward_q.put(acc_reward)
                    self._async_episode_step(acc_reward=acc_reward)
                    #print("Acc reward in episode: ", acc_reward)
                    acc_reward = 0
                    break


            if done:
                continue

            A = np.zeros((1, 1))
            for j in reversed(range(counter-1)):
                td = adv_rewards[j] + gamma * adv_values[j + 1] - adv_values[j]
                A = td + gamma * lmda * A
                adv_advantages[j] = A
                R = A + adv_values[j]
                adv_returns[j] = R
                q.put([adv_states[j], adv_actions[j], adv_rewards[j],
                                          False, [adv_returns[j], adv_advantages[j]]])


    def multithreaded_explore(self):

        self.network.share_memory()
        processes = []
        experience_queue = Queue()
        reward_queue = Queue()
        import copy

        for i in range(self.num_threads):
            process = mltip.Process(target=DPPOTrainer.gather_experience_and_calc_advantages,
                                    args=(self,self.network, experience_queue, reward_queue, self.max_episode_len,
                                          self.gamma, self.lmda, copy.deepcopy(self.env),
                                          self._episode_step, np.random.randint(0,10000), self.T, self.sigma_log))
            processes.append(process)
            process.start()
        for i in tqdm(range(self.memory_fill_steps)):
            self.replay_memory.append(*experience_queue.get(block=True))
        experience_queue.empty()
        for process in processes:
            process.terminate()

        reward_arr = queue_to_array(reward_queue)

        self.mvavg_reward.append(reward_arr.mean())
        self.reward_std.append(reward_arr.std())

        prGreen("#Horizon step: {} Average reward: {} "
                "Reward step average std: {}".format(self.hstep, np.mean(self.mvavg_reward), np.mean(self.reward_std)))


    def sample_and_update(self):
        av_loss = 0
        # update old model
        self.network_old.load_state_dict(self.network.state_dict())


        self.multithreaded_explore()
        self.sample_and_update()
        self.sigma_log -= 1e-2


#
#
# from torch_rl.envs.wrappers import *
# import gym
# from torch_rl.models.ppo import ActorCriticPPO
#
# env = NormalisedObservationsWrapper(
#     NormalisedActionsWrapper(gym.make("Pendulum-v0")))
# print(env.reward_range)
#
#
# network = ActorCriticPPO([env.observation_space.shape[0], 64, 64, env.action_space.shape[0]])
# network_old = ActorCriticPPO([env.observation_space.shape[0], 64, 64, env.action_space.shape[0]])
# trainer = DPPOTrainer(network=network, network_old=network_old, env=env)
# trainer.train(horizon=100000, max_episode_len=500)

