import gym
from torch_rl.models import Reservoir
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nengolib.neurons import  Tanh
import nengo


steps = 1000
spectral_radius = 0.8
neuron_type = Tanh()
network_size = 300
simulation_time_per_step = 0.1
simulation_dt = 0.01
input_size = 4

np.random.seed(666)

state_prev = np.zeros(200)

# Initial input
x = np.ones(input_size)

reservoir = Reservoir(simulation_time_per_step, simulation_dt, spectral_radius=spectral_radius,
                      network_size=network_size, input_size=input_size, recursive=True, neuron_type=neuron_type)
state = reservoir.forward(x)
activities = np.zeros(steps)


#Loop without 0 input
x = np.zeros(input_size)
for i in range(steps):

    # state, reward, done, _ = env.step(env.action_space.sample())
    state = reservoir.forward(x)
    activity = np.abs(np.sum(state))
    activities[i] = activity
    if i%10 == 0:
        print(i, "activitiy: ", activity)
    state_prev = state

#Take log of activities for better visualisation

activities[activities>0] = np.log(activities[activities>0])
activities[activities==0] = -1000
df = pd.DataFrame(activities, columns=['log_activity'])
df.index.name = 'step'

df.plot(y='log_activity')
plt.show()