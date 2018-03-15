from torch_rl.memory.core import *

class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)

        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)
        self.goals = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each
            # index.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]
            goal = self.goals[idx - 1] if self.goals.length > 0 else None

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, goal=goal, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def sample_and_split(self, batch_size, batch_idxs=None):
        experiences = self.sample(batch_size, batch_idxs)

        state0_batch = []
        goal_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            goal_batch.append(e.goal)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size, -1)
        state1_batch = np.array(state1_batch).reshape(batch_size, -1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size, -1)
        reward_batch = np.array(reward_batch).reshape(batch_size, -1)
        action_batch = np.array(action_batch).reshape(batch_size, -1)

        if self.goals.length > 0:
            return state0_batch, goal_batch, action_batch, reward_batch, state1_batch, terminal1_batch
***REMOVED***
            return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch

    def _append(self, observation, action, reward, terminal, training=True):
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    def _appendg(self, observation, goal, action, reward, terminal, training=True):
        self._append(observation, action, reward, terminal, training=training)
        if training:
            self.goals.append(goal)

    def append(self, *args, **kwargs):
        try:
            self._appendg(*args, **kwargs)
        except Exception as e:
            self._append(*args, **kwargs)

    @property
    def nb_entries(self):
        return len(self.observations)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config





class GeneralisedlMemory(SequentialMemory):

    def __init__(self, limit,**kwargs):
        super(GeneralisedlMemory, self).__init__(limit, **kwargs)
        self.extra_info = RingBuffer(limit)
        self.limit = limit

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each
            # index.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]
            goal = self.goals[idx - 1] if self.goals.length > 0 else None

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            extra0 = self.extra_info[idx]

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append([state0, action,reward,
                                          state1, terminal1, extra0])
        assert len(experiences) == batch_size
        return experiences

    def _append(self, observation, action, reward, terminal, extra_info, training=True):
            super(SequentialMemory, self).append(observation, action, reward, False, training=training)

            # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
            # and weather the next state is `terminal` or not.
            if training:
                self.observations.append(observation)
                self.actions.append(action)
                self.rewards.append(reward)
                self.terminals.append(terminal)
                self.extra_info.append(extra_info)



    def sample_and_split(self, batch_size, batch_idxs=None):
        experiences = self.sample(batch_size, batch_idxs)

        state0_batch = []
        goal_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        extra_info = []
        for e in experiences:
            state0_batch.append(e[0])
            state1_batch.append(e[1])
            reward_batch.append(e[2])
            action_batch.append(e[3])
            terminal1_batch.append(0. if e[4] else 1.)
            extra_info.append(e[-1])

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size, -1)
        state1_batch = np.array(state1_batch).reshape(batch_size, -1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size, -1)
        reward_batch = np.array(reward_batch).reshape(batch_size, -1)
        action_batch = np.array(action_batch).reshape(batch_size, -1)
        extra_info_batch = np.array(extra_info).reshape(batch_size, -1)

        if self.goals.length > 0:
            return state0_batch, goal_batch, action_batch, reward_batch, state1_batch, terminal1_batch
***REMOVED***
            return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch, extra_info_batch


# memory = GeneralisedSequentialMemory(limit=10000)
#
# for i in range(400):
#     a = np.zeros(10)
#     memory.append(a, a, 4., False, [0.1, 3.5])
#
# s0, a, r, s1, terminal, ei = memory.sample_and_split(40)
#

