from torch_rl.training.core import HorizonTrainer, mse_loss
from torch_rl.memory import GeneralisedlMemory
from torch.optim import Adam
from torch_rl.utils import to_tensor
from torch.autograd import Variable
import torch as tor
import numpy as np
import multiprocessing as mltip
from torch import multiprocessing as pmltip
from multiprocessing import Queue, Process
from tqdm import tqdm
from collections import deque
from torch_rl.utils import prGreen


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



class PPOTrainer(HorizonTrainer):


    """
        Implementation of proximal policy optimization.

        A = Q(s,a) - V(s)
        ratio = pi(s)/pi_old(s)

    """

    critic_criterion = mse_loss

    def __init__(self, env, network, network_old, num_episodes=2000, max_episode_len=500, batch_size=64, gamma=.99,
                 replay_memory=GeneralisedlMemory(10000, window_length=1), lr=1e-4, memory_fill_steps=10000,
                 epsilon=0.2, optimizer=None, lmda=0.95, ent_coeff=0.01, policy_update_epochs=512):
        super(PPOTrainer, self).__init__(env)
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
        self.reward_std = deque(maxlen=100)

        # Convenience buffers for faster iteration over advantage calculation

        self.adv_states = np.zeros((self.memory_fill_steps, env.observation_space.shape[0]))
        self.adv_actions = np.zeros((self.memory_fill_steps, env.action_space.shape[0]))
        self.adv_rewards = np.zeros(self.memory_fill_steps)
        self.adv_values = np.zeros(self.memory_fill_steps)
        self.adv_returns = np.zeros(self.memory_fill_steps)
        self.adv_terminal = np.zeros(self.memory_fill_steps)
        self.adv_advantages = np.zeros(self.memory_fill_steps)


    def add_to_replay_memory(self,s,a,r,d):
        if self.goal_based:
            self.replay_memory.append(self.state, self.env.goal, a, r, d, training=True)
        else:
            self.replay_memory.append(self.state, a, r, d, training=True)


    def gather_experience_and_calc_advantages(self):
        acc_reward = 0
        episode_length = 0
        for i in tqdm(range(self.memory_fill_steps)):
            self.adv_states[i] = self.state
            action, value = self.network(to_tensor(self.state).view(1,-1))
            self.adv_actions[i] = action.data.numpy()
            self.adv_values[i] = value.data.numpy()

            env_action = action.data.squeeze().numpy()
            state, reward, done, _ = self.env.step(env_action)
            episode_length += 1
            acc_reward += reward

            self.state = state
            done = done or episode_length >= self.max_episode_len
            self.adv_terminal[i] = done

            self._episode_step(state, env_action, reward, done)

            if done:
                self.env.reset()
                self.num_episodes += 1
                episode_length = 0

            #reward = max(min(reward, 1), -1)
            self.adv_rewards[i] = reward

            R = np.zeros((1,1))
            if not done:
                action, value = self.network(to_tensor(state).view(1,-1))
                R = value.data.numpy()
            self.adv_values[i] = R

        A = np.zeros((1,1))
        for j in reversed(range(i)):
            td = self.adv_rewards[j] + self.gamma * self.adv_values[j + 1] - self.adv_values[j]
            A = td + self.gamma * self.lmda * A
            self.adv_advantages[j] = A
            R = A + self.adv_values[j]
            self.adv_returns[j] = R

            self.replay_memory.append(self.adv_states[j], self.adv_actions[j], self.adv_rewards[j],
                                      self.adv_terminal[j], [self.adv_returns[j], self.adv_advantages[j]])


    def sample_and_update(self):
        av_loss = 0
        # update old model
        self.network_old.load_state_dict(self.network.state_dict())
        for step in tqdm(range(self.policy_update_epochs)):
            # cf https://github.com/openai/baselines/blob/master/baselines/pposgd/pposgd_simple.py
            batch_states, batch_actions, batch_rewards, batch_states1, batch_terminals, extra = \
                self.replay_memory.sample_and_split(self.batch_size)


            batch_returns = to_tensor(extra[:, 0])
            batch_advantages = to_tensor(extra[:, 1])
            batch_states = to_tensor(batch_states)
            batch_actions = to_tensor(batch_actions)

            # old probas
            actions_old, v_pred_old = self.network_old(batch_states.detach())
            probs_old = self.network_old.log_prob(batch_actions)

            # new probabilities
            actions, v_pred = self.network(batch_states)
            probs = self.network.log_prob(batch_actions)

            # ratio
            ratio = probs / (1e-15 + probs_old)
            # clip loss
            surr1 = ratio * tor.stack([batch_advantages] * batch_actions.shape[1],
                                      1)  # surrogate from conservative policy iteration
            surr2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * tor.stack(
                [batch_advantages] * batch_actions.shape[1], 1)
            loss_clip = -tor.mean(tor.min(surr1, surr2))
            # value loss
            vfloss1 = (v_pred - batch_returns) ** 2
            #v_pred_clipped = v_pred_old + (v_pred - v_pred_old).clamp(-self.epsilon, self.epsilon)
            vfloss2 = (v_pred - batch_returns) ** 2
            loss_value = 0.5 * tor.mean(tor.max(vfloss1, vfloss2))  # also clip value loss
            # entropy
            loss_ent = -self.ent_coeff * tor.mean((np.e * probs)* probs)
            # total
            total_loss = (loss_clip + loss_value + loss_ent)
            av_loss += total_loss.data[0] / float(self.num_episodes)
            # step
            self.optimizer.zero_grad()
            # model.zero_grad()
            total_loss.backward()
            # print(list(self.network.parameters())[0].grad)
            self.optimizer.step()
        self.num_episodes = 0
        self.replay_memory.clear()

    def _horizon_step(self):

        self.gather_experience_and_calc_advantages()
        self.sample_and_update()




class DPPOTrainer(HorizonTrainer):


    """
        Implementation of distributed proximal policy optimization.

        A = Q(s,a) - V(s)
        ratio = pi(s)/pi_old(s)

    """

    critic_criterion = mse_loss

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




    @staticmethod
    def gather_experience_and_calc_advantages(network, q, reward_q, max_episode_len, gamma, lmda, env, step, pid, T, sigma_log):
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
                                    args=(self.network, experience_queue, reward_queue, self.max_episode_len,
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
        for step in tqdm(range(self.policy_update_epochs)):
            # cf https://github.com/openai/baselines/blob/master/baselines/pposgd/pposgd_simple.py
            batch_states, batch_actions, batch_rewards, batch_states1, batch_terminals, extra = \
                self.replay_memory.sample_and_split(self.batch_size)


            batch_returns = to_tensor(extra[:, 0])
            batch_advantages = to_tensor(extra[:, 1])
            batch_states = to_tensor(batch_states)
            batch_actions = to_tensor(batch_actions)

            # old probas
            actions_old, v_pred_old = self.network_old(batch_states.detach(), self.sigma_log)
            log_probs_old  = self.network_old.log_prob(batch_actions.detach())

            # new probabilities
            actions, v_pred = self.network(batch_states, self.sigma_log)
            log_probs = self.network.log_prob(batch_actions)


            # ratio
            log_ratio = log_probs - log_probs_old
            ratio = np.e ** log_ratio


            # clip loss
            surr1 = ratio * tor.stack([batch_advantages] * batch_actions.shape[1],
                                      1)  # surrogate from conservative policy iteration
            surr2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * tor.stack(
                [batch_advantages] * batch_actions.shape[1], 1)
            loss_clip = -tor.mean(tor.min(surr1, surr2))
            # value loss
            vfloss1 = (v_pred - batch_returns) ** 2
            v_pred_clipped = v_pred_old + (v_pred - v_pred_old).clamp(-self.epsilon, self.epsilon)
            vfloss2 = (v_pred_clipped - batch_returns) ** 2
            loss_value = 0.5 * tor.mean(tor.max(vfloss1, vfloss2))  # also clip value loss
            loss_value = tor.mean(loss_value)
            # entropy
            #loss_ent = -self.ent_coeff * tor.mean(probs * tor.log(probs+1e-15))
            # total
            total_loss = (loss_clip + loss_value)
            av_loss += total_loss / self.policy_update_epochs


            # step
            self.optimizer.zero_grad()
            # model.zero_grad()
            total_loss.backward()
            tor.nn.utils.clip_grad_norm(network.parameters(), 0.5)
            # print(list(self.network.parameters())[0].grad)
            self.optimizer.step()

        print("Total loss: ", av_loss.data.numpy())
        print("Loss value: ", loss_value.data.numpy())
        print("Loss policy: ", loss_clip.data.numpy())
        self.num_episodes = 0
        self.replay_memory.clear()

    def _horizon_step(self):

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
#
