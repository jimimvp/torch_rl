from torch_rl.training.core import Trainer
import copy

class PPOTrainer(Trainer):


    """
        Implementation of proximal policy optimization.

        A = Q(s,a) - V(s)
        ratio = pi(s)/pi_old(s)

    """

***REMOVED***

    def __init__(self, env, actor, critic, num_episodes=2000, max_episode_len=500, batch_size=32, gamma=.99,
***REMOVED***
***REMOVED***
***REMOVED***
        super(PPOTrainer, self).__init__(env)
        if exploration_process is None:
            self.random_process = OrnsteinUhlenbeckActionNoise(self.env.action_space.shape[0])
***REMOVED***
            self.random_process = exploration_process
        self.lr_critic = lr_critic
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.replay_memory = replay_memory
        self.max_episode_len = max_episode_len
        self.epsilon = epsilon
        self.warmup = warmup
        self.gamma = gamma
        self.optimizer_actor = Adam(actor.parameters(), lr=lr_actor) if optimizer_actor is None else optimizer_actor
        self.optimizer_critic = Adam(critic.parameters(), lr=lr_critic) if optimizer_critic is None else optimizer_critic
        self.goal_based = hasattr(env, "goal")

        self.old_actor = copy.deepcopy(actor)
        self.old_agent = ActorCriticAgent(self.target_actor,self.target_critic)
        self.agent = ActorCriticAgent(actor, critic)

    def add_to_replay_memory(self,s,a,r,d):
***REMOVED***
            self.replay_memory.append(self.state, self.env.goal, a, r, d, training=True)
***REMOVED***
            self.replay_memory.append(self.state, a, r, d, training=True)

***REMOVED***

        for i in range(self.warmup):
            a = self.env.action_space.sample()
***REMOVED***
            self.add_to_replay_memory(self.state, a, r, d)
***REMOVED***

***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***
            action = self.agent.action(np.hstack((self.state, self.env.goal))).cpu().data.numpy()
***REMOVED***
            action = self.agent.action(self.state).cpu().data.numpy()

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***

        self.add_to_replay_memory(self.state, action, reward, done)
***REMOVED***

***REMOVED***
***REMOVED***
            s1, g, a1, r, s2, terminal = self.replay_memory.sample_and_split(self.batch_size)
***REMOVED***
***REMOVED***
***REMOVED***
            s1, a1, r, s2, terminal = self.replay_memory.sample_and_split(self.batch_size)


***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***


