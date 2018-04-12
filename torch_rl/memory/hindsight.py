from torch_rl.memory.core import *

class HindsightMemory(Memory):
    """
        Implementation of replay memory for hindsight experience replay with future
        transition sampling.
    """

    def __init__(self, limit, hindsight_size=8, goal_indices=None, reward_function=lambda observation,goal: 1, **kwargs):
        super(HindsightMemory, self).__init__(**kwargs)
        self.hindsight_size = hindsight_size
        self.reward_function = reward_function
        self.hindsight_buffer = RingBuffer(limit)
        self.goals = RingBuffer(limit)
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

        self.limit = limit
        self.last_terminal_idx = 0
        self.goal_indices = goal_indices

    def append(self, observation, action, reward, terminal, goal=None, training=True):
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            if goal is None:
                goal = observation[self.goal_indices]
            self.goals.append(goal)
            if terminal:
                """
                Sample hindsight_size of states added from recent terminal state to this one.
                """
                self.add_hindsight()
                self.last_terminal_idx = self.goals.last_idx

            super(HindsightMemory, self).append(observation, action, reward, terminal, training=True)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.nb_entries:
            raise KeyError()
        return self.observations[idx], self.goals[idx], self.actions[idx], self.rewards[idx], self.terminals[idx]


    def pop(self):
        """
        Remove last hindsight_size of elements because they were not
        used to generate hindsight.
        """
        self.observations.pop(self.hindsight_size)
        self.actions.pop(self.hindsight_size)
        self.goals.pop(self.hindsight_size)
        self.terminals.pop(self.hindsight_size)
        self.rewards.pop(self.hindsight_size)

    def add_hindsight(self):
        for i in range(self.last_terminal_idx+1, self.observations.last_idx-self.hindsight_size):
            # For every state in episode sample hindsight_size from future states
            # hindsight_idx = sample_batch_indexes(i+1, self.observations.last_idx, self.hindsight_size)
            hindsight_experience = (self.observations.last_idx - i - 1)*[None]
            for j,idx in enumerate(range(i+1, self.observations.last_idx)):
                hindsight_experience[j] = [i,idx]
            self.hindsight_buffer.append(np.asarray(hindsight_experience))

    def sample_and_split(self, num_transitions, batch_idxs=None, split_goal=False):
        batch_size = num_transitions*self.hindsight_size + num_transitions
        batch_idxs = sample_batch_indexes(0, self.hindsight_buffer.length, size=num_transitions)

        state0_batch = []
        reward_batch = []
        goal_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for idx in batch_idxs:
            # Add hindsight experience to batch
            hindsight_idxs = sample_batch_indexes(0, len(self.hindsight_buffer[idx]), self.hindsight_size)
            for root_idx, hindsight_idx in self.hindsight_buffer[idx][hindsight_idxs]:
                state0_batch.append(self.observations[hindsight_idx])
                state1_batch.append(self.observations[hindsight_idx+1])
                reward_batch.append(1.)
                action_batch.append(self.actions[hindsight_idx])
                goal_batch.append(self.observations[hindsight_idx+1] if self.goal_indices is None else \
                                  self.observations[hindsight_idx+1][self.goal_indices])
            state0_batch.append(self.observations[root_idx])
            state1_batch.append(self.observations[root_idx + 1])
            reward_batch.append(self.rewards[root_idx])
            action_batch.append(self.actions[root_idx])
            goal_batch.append(self.goals[root_idx])

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size,-1)
        state1_batch = np.array(state1_batch).reshape(batch_size,-1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size,-1)
        reward_batch = np.array(reward_batch).reshape(batch_size,-1)
        action_batch = np.array(action_batch).reshape(batch_size,-1)
        goal_batch = np.array(goal_batch).reshape(batch_size, -1)

        if split_goal:
            return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch, goal_batch
        else:
            state0_batch[:, self.goal_indices] = goal_batch
            state1_batch[:, self.goal_indices] = goal_batch
            return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch



    @property
    def nb_entries(self):
        return len(self.observations)





class SpikingHindsightMemory(HindsightMemory):
    """
        Implementation of replay memory for hindsight experience replay with future
        transition sampling.
    """

    def __init__(self, limit, hindsight_size=8, goal_indices=None, reward_function=lambda observation,goal: 1, **kwargs):
        super(SpikingHindsightMemory, self).__init__(limit, hindsight_size, goal_indices, reward_function, **kwargs)
        self.sobservations = RingBuffer(limit)

    def append(self, observation, sobservation, goal, action, reward, terminal, training=True):
        if training:
            self.sobservations.append(sobservation)
            super(SpikingHindsightMemory, self).append(observation, goal, action, reward, terminal, training=True)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.nb_entries:
            raise KeyError()
        return self.observations[idx], self.sobservations[idx], self.goals[idx], self.actions[idx], self.rewards[idx], self.terminals[idx]


    def add_hindsight(self):
        for i in range(self.last_terminal_idx+1, self.observations.last_idx-self.hindsight_size):
            # For every state in episode sample hindsight_size from future states
            # hindsight_idx = sample_batch_indexes(i+1, self.observations.last_idx, self.hindsight_size)
            hindsight_experience = (self.observations.last_idx - i - 1)*[None]
            for j,idx in enumerate(range(i+1, self.observations.last_idx)):
                hindsight_experience[j] = [i,idx]
            self.hindsight_buffer.append(np.asarray(hindsight_experience))

    def sample_and_split(self, num_transitions, batch_idxs=None):
        batch_size = num_transitions*self.hindsight_size + num_transitions
        batch_idxs = sample_batch_indexes(0, self.hindsight_buffer.length, size=num_transitions)

        state0_batch = []
        sstate0_batch = []
        reward_batch = []
        goal_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        sstate1_batch = []
        for idx in batch_idxs:
            # Add hindsight experience to batch
            hindsight_idxs = sample_batch_indexes(0, len(self.hindsight_buffer[idx]), self.hindsight_size)
            for root_idx, hindsight_idx in self.hindsight_buffer[idx][hindsight_idxs]:
                state0_batch.append(self.observations[hindsight_idx])
                sstate0_batch.append(self.sobservations[hindsight_idx])
                state1_batch.append(self.observations[hindsight_idx+1])
                sstate1_batch.append(self.sobservations[hindsight_idx+1])
                reward_batch.append(1)
                action_batch.append(self.actions[hindsight_idx])
                goal_batch.append(self.observations[hindsight_idx+1] if self.goal_indices is None else \
                                  self.observations[hindsight_idx+1][self.goal_indices])
            state0_batch.append(self.observations[root_idx])
            sstate0_batch.append(self.sobservations[root_idx])
            state1_batch.append(self.observations[root_idx + 1])
            sstate1_batch.append(self.sobservations[root_idx + 1])
            reward_batch.append(self.rewards[root_idx])
            action_batch.append(self.actions[root_idx])
            goal_batch.append(self.goals[root_idx])

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size,-1)
        sstate0_batch = np.array(sstate0_batch).reshape(batch_size, -1)
        state1_batch = np.array(state1_batch).reshape(batch_size,-1)
        sstate1_batch = np.array(sstate1_batch).reshape(batch_size, -1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size,-1)
        reward_batch = np.array(reward_batch).reshape(batch_size,-1)
        action_batch = np.array(action_batch).reshape(batch_size,-1)

        return state0_batch, sstate0_batch, goal_batch, action_batch, reward_batch, \
               state1_batch, sstate1_batch, terminal1_batch


    @property
    def nb_entries(self):
        return len(self.observations)


from .sequential import GeneralisedMemory

class GeneralisedHindsightMemory(GeneralisedMemory):
    """
        Implementation of replay memory for hindsight experience replay with future
        transition sampling.
    """

    def __init__(self, limit, hindsight_size=8, goal_indices=None, reward_function=lambda observation,goal: 1, **kwargs):
        super(GeneralisedHindsightMemory, self).__init__(limit,**kwargs)
        self.hindsight_size = hindsight_size
        self.reward_function = reward_function
        self.hindsight_buffer = RingBuffer(limit)
        self.goals = RingBuffer(limit)

        self.limit = limit
        self.last_terminal_idx = 0
        self.goal_indices = goal_indices

    def append(self, observation, action, reward, terminal, extra_info=None,training=True,  goal=None):
        if training:
            if goal is None:
                goal = observation[self.goal_indices]
            self.goals.append(goal)
            if terminal:
                """
                Sample hindsight_size of states added from recent terminal state to this one.
                """
                self.add_hindsight()
                self.last_terminal_idx = self.goals.last_idx

            super(GeneralisedHindsightMemory, self).append(observation, action, reward, terminal, extra_info=extra_info, training=True)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.nb_entries:
            raise KeyError()
        return self.observations[idx], self.goals[idx], self.actions[idx], self.rewards[idx], self.terminals[idx]


    def add_hindsight(self):
        for i in range(self.last_terminal_idx+1, self.observations.last_idx-self.hindsight_size):
            # For every state in episode sample hindsight_size from future states
            # hindsight_idx = sample_batch_indexes(i+1, self.observations.last_idx, self.hindsight_size)
            hindsight_experience = (self.observations.last_idx - i - 1)*[None]
            for j,idx in enumerate(range(i+1, self.observations.last_idx)):
                hindsight_experience[j] = [i,idx]
            self.hindsight_buffer.append(np.asarray(hindsight_experience))

    def sample_and_split(self, num_transitions, batch_idxs=None, split_goal=False):
        batch_size = num_transitions*self.hindsight_size + num_transitions
        batch_idxs = sample_batch_indexes(0, self.hindsight_buffer.length, size=num_transitions)

        state0_batch = []
        reward_batch = []
        goal_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        extra_info = []
        for idx in batch_idxs:
            # Add hindsight experience to batch
            hindsight_idxs = sample_batch_indexes(0, len(self.hindsight_buffer[idx]), self.hindsight_size)
            for root_idx, hindsight_idx in self.hindsight_buffer[idx][hindsight_idxs]:
                state0_batch.append(self.observations[hindsight_idx])
                state1_batch.append(self.observations[hindsight_idx+1])
                reward_batch.append(1.)
                action_batch.append(self.actions[hindsight_idx])
                goal_batch.append(self.observations[hindsight_idx+1] if self.goal_indices is None else \
                                  self.observations[hindsight_idx+1][self.goal_indices])
                extra_info.append(self.extra_info[hindsight_idx])
            state0_batch.append(self.observations[root_idx])
            state1_batch.append(self.observations[root_idx + 1])
            reward_batch.append(self.rewards[root_idx])
            action_batch.append(self.actions[root_idx])
            goal_batch.append(self.goals[root_idx])
            extra_info.append(self.extra_info[root_idx])

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size,-1)
        state1_batch = np.array(state1_batch).reshape(batch_size,-1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size,-1)
        reward_batch = np.array(reward_batch).reshape(batch_size,-1)
        action_batch = np.array(action_batch).reshape(batch_size,-1)
        extra_info_batch = np.array(extra_info).reshape(batch_size,-1)
        goal_batch = np.array(goal_batch).reshape(batch_size,-1)
        if split_goal:
            return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch, extra_info, goal_batch
        else:
            state0_batch[:, self.goal_indices] = goal_batch
            state1_batch[:, self.goal_indices] = goal_batch
            return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch, extra_info_batch


    @property
    def nb_entries(self):
        return len(self.observations)





class SpikingHindsightMemory(HindsightMemory):
    """
        Implementation of replay memory for hindsight experience replay with future
        transition sampling.
    """

    def __init__(self, limit, hindsight_size=8, goal_indices=None, reward_function=lambda observation,goal: 1, **kwargs):
        super(SpikingHindsightMemory, self).__init__(limit, hindsight_size, goal_indices, reward_function, **kwargs)
        self.sobservations = RingBuffer(limit)

    def append(self, observation, sobservation, goal, action, reward, terminal, training=True):
        if training:
            self.sobservations.append(sobservation)
            super(SpikingHindsightMemory, self).append(observation, goal, action, reward, terminal, training=True)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.nb_entries:
            raise KeyError()
        return self.observations[idx], self.sobservations[idx], self.goals[idx], self.actions[idx], self.rewards[idx], self.terminals[idx]


    def add_hindsight(self):
        for i in range(self.last_terminal_idx+1, self.observations.last_idx-self.hindsight_size):
            # For every state in episode sample hindsight_size from future states
            # hindsight_idx = sample_batch_indexes(i+1, self.observations.last_idx, self.hindsight_size)
            hindsight_experience = (self.observations.last_idx - i - 1)*[None]
            for j,idx in enumerate(range(i+1, self.observations.last_idx)):
                hindsight_experience[j] = [i,idx]
            self.hindsight_buffer.append(np.asarray(hindsight_experience))

    def sample_and_split(self, num_transitions, batch_idxs=None):
        batch_size = num_transitions*self.hindsight_size + num_transitions
        batch_idxs = sample_batch_indexes(0, self.hindsight_buffer.length, size=num_transitions)

        state0_batch = []
        sstate0_batch = []
        reward_batch = []
        goal_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        sstate1_batch = []
        for idx in batch_idxs:
            # Add hindsight experience to batch
            hindsight_idxs = sample_batch_indexes(0, len(self.hindsight_buffer[idx]), self.hindsight_size)
            for root_idx, hindsight_idx in self.hindsight_buffer[idx][hindsight_idxs]:
                state0_batch.append(self.observations[hindsight_idx])
                sstate0_batch.append(self.sobservations[hindsight_idx])
                state1_batch.append(self.observations[hindsight_idx+1])
                sstate1_batch.append(self.sobservations[hindsight_idx+1])
                reward_batch.append(1)
                action_batch.append(self.actions[hindsight_idx])
                goal_batch.append(self.observations[hindsight_idx+1] if self.goal_indices is None else \
                                  self.observations[hindsight_idx+1][self.goal_indices])
            state0_batch.append(self.observations[root_idx])
            sstate0_batch.append(self.sobservations[root_idx])
            state1_batch.append(self.observations[root_idx + 1])
            sstate1_batch.append(self.sobservations[root_idx + 1])
            reward_batch.append(self.rewards[root_idx])
            action_batch.append(self.actions[root_idx])
            goal_batch.append(self.goals[root_idx])

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size,-1)
        sstate0_batch = np.array(sstate0_batch).reshape(batch_size, -1)
        state1_batch = np.array(state1_batch).reshape(batch_size,-1)
        sstate1_batch = np.array(sstate1_batch).reshape(batch_size, -1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size,-1)
        reward_batch = np.array(reward_batch).reshape(batch_size,-1)
        action_batch = np.array(action_batch).reshape(batch_size,-1)

        return state0_batch, sstate0_batch, goal_batch, action_batch, reward_batch, \
               state1_batch, sstate1_batch, terminal1_batch


    @property
    def nb_entries(self):
        return len(self.observations)