from torch_rl.training.core import HorizonTrainer, mse_loss
from torch_rl.memory import GeneralisedMemory
from torch.optim import Adam
from torch_rl.utils import to_tensor as tt
import torch as tor
from collections import deque
from torch_rl.utils import prGreen
import time
import sys
from torch_rl.utils import *
import numpy as np

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

    def __init__(self, env, policy_network, critic_network, nsteps, gamma, lam, replay_memory, hindsight_points=None):
        self.env = env
        self.policy_network = policy_network
        self.critic_network = critic_network
        nenv = 1
        self.obs = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.state = None
        self.done = False
        self.global_step = 0
        self.episodes = 0
        self.replay_memory = replay_memory

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logpacs = [], [], [], [], [], []
        mb_state = self.state
        epinfos = []
        self.critic_network.cpu()

        # 

        for _ in range(self.nsteps):
            actions, values = self.policy_network(tt(self.obs, cuda=False).view(1,-1))
            logpacs = self.policy_network.logprob(actions)

            mb_obs.append(self.obs.copy().flatten())
            mb_actions.append(actions.data.numpy().flatten())
            mb_values.append(values.detach().data.numpy().flatten())
            mb_logpacs.append(logpacs.data.numpy().flatten())

            mb_dones.append(self.done)

            a = actions.data.numpy().flatten()
            obs, reward, self.done, infos = self.env.step(a)

            q = self.critic_network(tt(self.obs.reshape(1,-1), cuda=False), actions)


            #Additional step in comparison to PPO
            self.replay_memory.append(obs, a, reward, self.done, extra_info=np.hstack((mb_logpacs[-1],q.cpu().data.numpy().flatten())))

            self.obs = obs
            self.global_step += 1
            mb_rewards.append(reward)

            if self.done:
                self.episodes+=1
                logger.logkv("episodes", self.episodes)
                self.obs = self.env.reset()


        self.critic_network.cuda()
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).reshape(self.nsteps, -1)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape(self.nsteps, -1)
        mb_actions = np.asarray(mb_actions, dtype=np.float32).reshape(self.nsteps,self.env.action_space.shape[0])
        mb_values = np.asarray(mb_values, dtype=np.float32).reshape(self.nsteps, -1)
        mb_logpacs = np.asarray(mb_logpacs, dtype=np.float32).reshape(self.nsteps,  self.env.action_space.shape[0])
        mb_dones = np.asarray(mb_dones, dtype=np.bool).reshape(self.nsteps, -1)

        action, last_values = self.policy_network(tt(self.obs.reshape(1,-1), cuda=False))
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


        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_logpacs, mb_state

        # obs, returns, masks, actions, values, neglogpacs, states = runner.run()

    def constfn(val):
        def f(_):
            return val

        return f

import copy
from torch_rl.memory import GeneralisedHindsightMemory

class HERIPGGPUPPOTrainer(HorizonTrainer):
    """
        Implementation of interpolated policy gradient for PPO
    """


    mvavg_reward = deque(maxlen=100)


    def __init__(self, env, policy_network, critic_network, replay_memory, max_episode_len=500, gamma=.99,
                lr=3e-4, n_steps=40, epsilon=0.2, optimizer=None, lmda=0.95, ent_coef=0., n_update_steps=10, 
                 n_minibatches=1, v=0.5, tau=1e-3):
        super(HERIPGGPUPPOTrainer, self).__init__(env)

        self.n_minibatches = n_minibatches
        self.lr = lr

        # Replay memory for calculating the online policy gradient
        self.replay_memory = replay_memory
        self.max_episode_len = max_episode_len
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmda = lmda
        self.optimizer = Adam(policy_network.parameters(), lr=lr, weight_decay=0.001) if optimizer is None else optimizer
        self.goal_based = hasattr(env, "goal")
        self.policy_network = policy_network
        self.ent_coef = ent_coef
        self.n_update_steps = n_update_steps
        self.n_steps = n_steps
        self.advantage_estimator = AdvantageEstimator(env, self.policy_network, critic_network, n_steps, self.gamma, self.lmda, self.replay_memory)
        
        self.v = v
        self.tau = tau
        self.critic_network = critic_network
        self.target_critic_network = cuda_if_available(copy.deepcopy(self.critic_network))
        self.target_policy_network = cuda_if_available(copy.deepcopy(self.policy_network))
        self.critic_optimizer = Adam(critic_network.parameters(), lr=3e-4, weight_decay=0.001)


    def _off_policy_loss(self, batch_size): 

        s1, a1, r, s2, terminal, goal, add_info = self.replay_memory.sample_and_split(batch_size)
        s1 = tt(s1).cuda()
        a1 = tt(a1).cuda()
        r = tt(r).cuda()
        s2 = tt(s2).cuda()
        oldlogpac = tt(add_info[:, :-1])
        oldq = tt(add_info[:, -1])



        #import pdb; pdb.set_trace()
        a2, v_pred = self.target_policy_network(s2)
        # Take deterministic step by taking the mean of the distribution
        a2 = self.target_policy_network.mu()

        q = self.critic_network(s1, a1)
        #q_clipped = oldq + tor.clamp(q - oldq, -self.epsilon, self.epsilon)
        q_target =  r + self.gamma*(self.target_critic_network(s2,a2))

        critloss1 = (q_target - q)**2
        # critloss2 = (q_target - q_clipped)**2
        # critloss = .5 * tor.mean(tor.max(critloss1, critloss2))

        critloss = .5 * tor.mean(critloss1)

        a, v = self.policy_network(s1)
        a = self.policy_network.mu()

        ratio =  tor.exp(self.policy_network.logprob(a) - oldlogpac)
        qestimate = self.critic_network(s1, a)

        #pgloss1 = -qestimate * ratio
        #pgloss2 = -qestimate * tor.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon)
        #pgloss = -tor.mean(tor.max(pgloss1, pgloss2))
        pgloss = -tor.mean(qestimate)


        mean_q_estimate = tor.mean(qestimate)
        mean_ratio = tor.mean(ratio)

        logger.logkv("erpgloss", pgloss.cpu().data.numpy())
        logger.logkv("qloss", critloss.cpu().data.numpy())
        logger.logkv("meanq", mean_q_estimate.cpu().data.numpy())
        logger.logkv("ratio", mean_ratio.cpu().data.numpy())
        logger.logkv("reward_mean", r.cpu().data.numpy().mean())

        return pgloss + critloss


    def _ppo_loss(self, bobs, bactions, badvs, breturns, blogpacs, bvalues):

        OBS = tt(bobs)
        A = tt(bactions)
        ADV = tt(badvs)
        R = tt(breturns)
        OLDLOGPAC = tt(blogpacs)
        OLDVPRED = tt(bvalues)

        self.policy_network(OBS)
        logpac = self.policy_network.logprob(A)
        entropy = tor.mean(self.policy_network.entropy())

        #### Value function loss ####
        #print(bobs)
        actions_new, v_pred = self.policy_network(tt(bobs))
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


        ppo_loss = v_loss  + pg_loss + self.ent_coef*entropy

        logger.logkv("siglog", self.policy_network.siglog.cpu().data.numpy()[0])
        logger.logkv("pgloss", pg_loss.cpu().data.numpy())
        logger.logkv("vfloss", v_loss.cpu().data.numpy())
        logger.logkv("vfloss", v_loss.cpu().data.numpy())
        logger.logkv("approxkl", approxkl.cpu().data.numpy())
        logger.logkv("pentropy", entropy.cpu().data.numpy())

        return ppo_loss

    def _horizon_step(self):


        obs, returns, masks, actions, values, logpacs, states = self.advantage_estimator.run() #pylint: disable=E0632
        
        # Normalize advantages over episodes
        advs = returns - values
        prev_ind = 0
        for ind in np.argwhere(masks == True)[:, 0]:
            episode_advs = advs[prev_ind:ind+1]
            advs[prev_ind:ind+1] = (episode_advs - episode_advs.mean())/(episode_advs.std() + 1e-8)
            prev_ind = ind+1

        episode_advs = advs[prev_ind:-1]
        advs[prev_ind:-1] = (episode_advs - episode_advs.mean())/(episode_advs.std() + 1e-8)
    

        nbatch_train = self.n_steps // self.n_minibatches

        self.policy_network.cuda()
        #self.optimizer = Adam(self.policy_network.parameters(), lr=self.lr) 

        if states is None: # nonrecurrent version
            inds = np.arange(self.n_steps)
            for _ in range(self.n_update_steps):
                np.random.shuffle(inds)
                for start in range(0, self.n_steps, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    bobs, breturns, bmasks, bactions, bvalues, blogpacs, badvs = map(lambda arr: arr[mbinds], (obs, returns, 
                        masks, actions, values, logpacs, advs))

                    # This introduces bias since the advantages can be normalized over more episodes
                    #advs = (advs - advs.mean()) / (advs.std() + 1e-8)



                    ppo_loss = self._ppo_loss(bobs, bactions, badvs, breturns, blogpacs, bvalues)
                    off_loss = self._off_policy_loss(nbatch_train)

                    loss = self.v*ppo_loss + (1-self.v) * off_loss

                    self.optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.critic_optimizer.step()

                    # Soft updates for target policies and critic
                    # Soft updates of critic don't help
                    soft_update(self.target_policy_network, self.policy_network, self.tau)
                    soft_update(self.target_critic_network, self.critic_network, self.tau)


        #Push to CPU
        self.policy_network.cpu()
        logger.dumpkvs()



if __name__ == '__main__':

    from torch_rl.envs.wrappers import *
    import gym
    from gym.wrappers import Monitor
    from torch_rl.models.ppo import ActorCriticPPO
    from torch_rl.utils import *
    from torch_rl.utils import logger
    from torch_rl.envs import EnvLogger
    import sys

    logger.configure(clear=False)
    monitor = Monitor(EnvLogger(NormalisedActionsWrapper(gym.make("Pendulum-v0"))), directory="./stats", force=True, 
        video_callable=False, write_upon_reset=True)
    env = RunningMeanStdNormalize(monitor)
    print(env.observation_space.shape)


    with tor.cuda.device(1):
        policy_network = ActorCriticPPO([env.observation_space.shape[0], 64, 64, env.action_space.shape[0]])
        policy_network.apply(gauss_init(0, np.sqrt(2)))

        trainer = GPUPPO(policy_network=policy_network, env=env, n_update_steps=4, n_steps=40)
        trainer.train(horizon=100000, max_episode_len=500)




