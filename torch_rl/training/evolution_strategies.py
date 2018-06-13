"""
    Implementation of evolution strategies for RL environments in PyTorch
"""

from torch import nn
import numpy as np
import torch as tor
from torch import nn
import copy
import numpy as np
from multiprocessing import Pool
from torch.distributions import MultivariateNormal, Normal, Uniform


class ESModel(nn.Module):
    def __init__(self, architecture, activation_functions=None):
        super(ESModel, self).__init__()
        self.model = nn.Sequential(*params)

    def sample_from_distribution(self, distribution):

        #Sample parameters from distribution
        parameters = distribution.rsample()
        self.set_flattened_parameters(parameters)

    def flattened_parameters(self):
        return tor.cat([param.data.view(-1) for param in self.parameters()])


    def set_flattened_parameters(self, parameters):
        ind = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            num_params = np.prod(param_shape)
            param.data = parameters[ind:ind+num_params].view(param_shape)
            ind+=num_params

    def __call__(self, x):
        return self.model.forward(x)

    def forward(self, x):
        return self.model.forward(x)


def es_eval_policy(policy, env, seed, sigma,steps=200):
        import numpy as np
        import scipy

        s = env.reset()
        dist = Normal(tor.zeros_like(policy.flattened_parameters()), 1.)
        rewards = []
        #Apply gaussian noise to parameters
        tor.manual_seed(seed)
        policy.set_flattened_parameters(policy.flattened_parameters() + dist.rsample()*sigma)

        for i in range(steps):
            a = policy(tor.from_numpy(s.reshape(1,-1).astype(np.float32))).data.numpy().flatten()
            s, r, d, _ = env.step(a)
            if d:
                s = env.reset()
            rewards.append(r)

        rewards = np.asarray(rewards)
        #rewards = (rewards - rewards.mean())/rewards.std()
        #Use mean reward
        F = np.sum(rewards)
        return F



def cma_eval_policy(policy, env ,steps=200):
        import numpy as np
        import scipy

        s = env.reset()
        rewards = []

        for i in range(steps):
            a = policy(tor.from_numpy(s.reshape(1,-1).astype(np.float32))).data.numpy().flatten()
            s, r, d, _ = env.step(a)
            if d:
                s = env.reset()
            rewards.append(r)

        rewards = np.asarray(rewards)
        #rewards = (rewards - rewards.mean())/rewards.std()
        #Use mean reward
        F = np.sum(rewards)
        return F


from torch_rl.training.core import HorizonTrainer

class ESTrainer(HorizonTrainer):

    def __init__(self, env, model, population_size=100, policy_eval_function=es_eval_policy,
        num_threads=8, learning_rate=1e-3, sigma=0.1, tau=0.0):
        super(ESTrainer, self).__init__(env)
        self.population_size = population_size
        self.policy_eval_function = policy_eval_function
        self.learning_rate = learning_rate
        self.model = model
        self.rollout_models =  np.asarray([copy.deepcopy(model) for i in range(population_size)])
        self.sigma = np.full(population_size, sigma)
        self.pool = Pool(num_threads)
        self.envs = np.asarray([copy.deepcopy(env) for _ in range(self.population_size)])
        self.eligibility_trace = tor.zeros_like(model.flattened_parameters())
        self.tau = tau

    def _horizon_step(self):
        

        from collections import deque

        d = deque(maxlen=100)
            
        seeds = np.random.randint(0,100000000, self.population_size)
        params = list(zip(self.rollout_models, self.envs, seeds, self.sigma))
        scores = self.pool.starmap(self.policy_eval_function, params)
        real_scores = np.asarray(scores)
        scores = real_scores

        #Ranked score
        ranked_scores = np.arange(0, len(scores))
        sorted_indices = np.flip(np.argsort(scores),0)
        scores = ranked_scores[sorted_indices]

        sscores = (scores-scores.mean())/(scores.std()+1e-7)
        nscores = (scores-scores.min())/scores.max()
        dist = Normal(tor.zeros_like(model.flattened_parameters()), 1.)
        param_updates = []
        for score, seed in zip(sscores, seeds):
            tor.manual_seed(seed)
            params_update = dist.rsample() * score
            param_updates.append(params_update)

        param_updates = tor.stack(param_updates)
        self.eligibility_trace = tor.mean(param_updates, 0) + self.eligibility_trace*self.tau
        model.set_flattened_parameters(model.flattened_parameters() + (self.learning_rate/self.sigma[0])*self.eligibility_trace)

        #Update main model
        [m.load_state_dict(model.state_dict()) for m in self.rollout_models]

        d.append(np.mean(real_scores))
        if self.hstep%1 == 0:
            print("Step {}, avg score {}, max score {}".format(self.hstep, np.mean(d), np.max(real_scores)), end='\r')
        #sigma -= sigma_decay







class CMAESTrainer(HorizonTrainer):
    #TODO: better way to guarantee a positive-definite covariance matrix

    def __init__(self, env, model, population_size=100, lmbda=25, 
        policy_eval_function=cma_eval_policy, num_threads=8,
        learning_rate=1e-3, sigma=0.1):
        super(CMAESTrainer, self).__init__(env)
        self.population_size = population_size
        self.lmbda = lmbda
        self.policy_eval_function = cma_eval_policy
        self.learning_rate = learning_rate
        self.model = model
        self.rollout_models =  np.asarray([copy.deepcopy(model) for i in range(population_size)])
        self.sigma = np.full(population_size, sigma)
        self.pool = Pool(num_threads)
        self.envs = np.asarray([copy.deepcopy(env) for _ in range(self.population_size)])

        dim = np.prod([param.data.view(-1).shape[0] for param in model.parameters()])
        flat_params = model.flattened_parameters()
        self.flattened_param_mean_old = tor.zeros_like(flat_params)

        mean = tor.from_numpy(np.zeros(flat_params.shape[0]).astype(np.float32))
        covar_matrix = tor.from_numpy(np.identity(flat_params.shape[0]).astype(np.float32)+1e-7)
        distribution = MultivariateNormal(mean, covar_matrix)
        #Update main model, takes a long time
        [m.sample_from_distribution(distribution) for m in self.rollout_models]
    
    def _horizon_step(self):
        #Layer-wise randomness

        from collections import deque

        d = deque(maxlen=100)
            
        params = list(zip(self.rollout_models, self.envs))
        scores = self.pool.starmap(self.policy_eval_function, params)
        scores = np.asarray(scores)
        
        env_scores = scores
        parameters_stacked = tor.stack([m.flattened_parameters() for m in self.rollout_models], 0)
        #L2 loss
        l2_loss = tor.mean(parameters_stacked * parameters_stacked, 1)         
        scores -= l2_loss.data.numpy()

        sscores = (scores-scores.mean())/(scores.std()+1e-7).astype(np.float32)
        nscores = (scores-scores.min())/scores.max()

        #Sort scores and take first lambda best individuals
        best_indices = np.flip(np.argsort(sscores), axis=0)[:self.lmbda]
        worst_indices = np.full(scores.shape[0], True)
        worst_indices[best_indices[0]] = False

        self.flattened_param_mean_old = model.flattened_parameters()

        tor_scores  = tor.from_numpy(sscores[:self.lmbda])
        best_parameters_stacked = tor.stack([m.flattened_parameters() for m in self.rollout_models[best_indices]], 0)
        mean_parameters_new = tor.mean(best_parameters_stacked, 0)

        best_new_parameters = mean_parameters_new 

        #Calculate covariance matrix


        covar_matrix = best_parameters_stacked - self.flattened_param_mean_old
        covar_matrix = (covar_matrix.transpose(1,0).matmul(covar_matrix))/float(self.lmbda)
        #Make the matrix positive definite
        #dist = Uniform(tor.zeros(covar_matrix.shape[0])+5e-2, tor.zeros(covar_matrix.shape[0]) + 5e-1)
        covar_matrix+=tor.eye(covar_matrix.shape[0])*5e-2

        self.flattened_param_mean_old = mean_parameters_new
        #assert not np.any(scores == np.nan)

        mean = mean_parameters_new
        #print("Covar max: ", tor.max(covar_matrix), "Mean max: ", tor.max(mean))

        distribution = MultivariateNormal(mean, covar_matrix)


        #Sample new population from calculated mean and 
        [m.sample_from_distribution(distribution) for m in self.rollout_models]

        d.append(np.mean(env_scores[best_indices]))
        if self.hstep%1 == 0:
            print("Step {}, avg score {}, max score {}".format(self.hstep, np.mean(d), np.max(env_scores)), end='\r')
        #sigma -= sigma_decay







if __name__=='__main__':


    import gym
    env = gym.make('Pendulum-v0')
    params = []
    tor.manual_seed(222)
    architecture = [env.observation_space.shape[0],64,64,env.action_space.shape[0]]
    for i in range(len(architecture)-1):
        params.append(nn.Linear(architecture[i], architecture[i+1]))
        params.append(nn.Tanh())

    model = ESModel(nn.Sequential(*params))
    trainer = CMAESTrainer(env, model, population_size=100, num_threads=8)

    trainer.train(1000, 500)


#Test covariance
    
    # print(np.sort(np.random.randint(0, 100, 100)))



    

