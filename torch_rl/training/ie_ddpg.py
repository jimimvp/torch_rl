from torch_rl.training.core import *
from torch_rl.memory import SequentialMemory
from torch_rl.models.core import ActorCriticAgent
from torch.optim import Adam
import copy

class IEDDPGTrainer(Trainer):

    critic_criterion = mse_loss

    def __init__(self, env, actor, critic, num_episodes=2000, max_episode_len=500, batch_size=32, gamma=.99,
              replay_memory=SequentialMemory(1000000, window_length=1), tau=1e-3, lr_critic=1e-3, lr_actor=1e-4, warmup=2000, depsilon=1./5000,
                 epsilon=1., exploration_process=None,
                 optimizer_critic=None, optimizer_actor=None):
        super(IEDDPGTrainer, self).__init__(env)
        if exploration_process is None:
            self.random_process = OrnsteinUhlenbeckActionNoise(self.env.action_space.shape[0])
        else:
            self.random_process = exploration_process
        self.action_choice_function = random_process_action_choice(self.random_process)
        self.tau = tau
        self.lr_critic = lr_critic
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.replay_memory = replay_memory
        self.max_episode_len = max_episode_len
        self.epsilon = epsilon
        self.depsilon = depsilon
        self.warmup = warmup
        self.gamma = gamma
        self.target_critic = copy.deepcopy(critic)
        self.target_actor = copy.deepcopy(actor)
        self.optimizer_actor = Adam(actor.parameters(), lr=lr_actor) if optimizer_actor is None else optimizer_actor
        self.optimizer_critic = Adam(critic.parameters(), lr=lr_critic) if optimizer_critic is None else optimizer_critic

        self.goal_based = hasattr(env, "goal")

        self.target_agent = ActorCriticAgent(self.target_actor,self.target_critic)
        self.agent = ActorCriticAgent(actor, critic)

    def add_to_replay_memory(self,s,a,r,d):
        if self.goal_based:
            self.replay_memory.append(self.state, self.env.goal, a, r, d, training=True)
        else:
            self.replay_memory.append(self.state, a, r, d, training=True)

    def _warmup(self):

        for i in range(self.warmup):
            a = self.env.action_space.sample()
            s, r, d, _ = self.env.step(a)
            self.add_to_replay_memory(self.state, a, r, d)
            self.state = s

    def _episode_start(self):

        self.random_process.reset()

    def _episode_step(self, episode, acc_reward):
        if self.goal_based:
            action = self.agent.action(np.hstack((self.state, self.env.goal))).cpu().data.numpy()
        else:
            action = self.agent.action(self.state).cpu().data.numpy()

        # No extrinsic exploration needed here, the network approximates a distribution

        state, reward, done, info = self.env.step(action)

        self.add_to_replay_memory(self.state, action, reward, done)
        self.state = state

        # Optimize over batch
        if self.goal_based:
            s1, g, a1, r, s2, terminal = self.replay_memory.sample_and_split(self.batch_size)
            s1 = np.hstack((s1,g))
            s2 = np.hstack((s2,g))
        else:
            s1, a1, r, s2, terminal = self.replay_memory.sample_and_split(self.batch_size)


        a2 = self.target_agent.actions(s2, volatile=True)

        q2 = self.target_agent.values(to_tensor(s2, volatile=True), a2, volatile=False)
        q2.volatile = False

        q_expected = to_tensor(np.asarray(r), volatile=False) + self.gamma * q2
        q_predicted = self.agent.values(to_tensor(s1), to_tensor(a1), requires_grad=True)

        self.optimizer_critic.zero_grad()
        critic_loss = IEDDPGTrainer.critic_criterion(q_expected, q_predicted)
        critic_loss.backward(retain_graph=True)
        self.optimizer_critic.step()
        # Actor optimization

        a1 = self.agent.actions(s1, requires_grad=True)
        q_input = tor.cat([to_tensor(s1), a1], 1)
        q = self.agent.values(q_input, requires_grad=True)
        loss_actor = -q.mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward(retain_graph=True)
        self.optimizer_actor.step()

        soft_update(self.target_agent.policy_network, self.agent.policy_network, self.tau)
        soft_update(self.target_agent.critic_network, self.agent.critic_network, self.tau)

        return state, reward, done, {}

    def _episode_end(self, episode):
        pass


