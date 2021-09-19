"""Testing and visualising saved policies in different environments."""

import numpy as np
from environment import env_loader
from scenes.poet_env import create_poet_env
import random

# Normalizer class
class Normalizer():

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        # print(x.shape, self.mean.shape, self.n.shape)
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std
    
    def set_n_mu_diff(self, n, mu, diff):
        self.n = n
        self.mean = mu
        self.mean_diff = diff
    
    def save_n_mu_diff(self):
        self._n = self.n
        self._mean = self.mean
        self._mean_diff = self.mean_diff    

    def get_saved_n_mu_diff(self):
        return self._n, self._mean, self._mean_diff


def sigmoid(x):
    return 1/(1+np.exp(-x))


class Policy():
    """Policy, consists of GRU on top 2-layer MLP
    """

    def __init__(self, input_size, hidden_size, gru_input, gru_hidden, output_size):
        self.gru_input = gru_input
        self.gru_hidden = gru_hidden
        self.input_size = input_size
        self.normalizer = Normalizer(input_size)
    
        # print("Starting policy theta=", self.theta1, self.theta2, self.theta3)
        # parameters = self.theta1.size+self.theta2.size+self.theta3.size+3*self.x2h.size+3*self.h2h.size
        # print(f"{parameters = }")
        
        
    def load_policy(self, policy_file):  
        self.policy_file = policy_file
        _policy = np.load(policy_file)
        
        # MLP
        self.theta1 = _policy['arr_0']
        self.theta2 = _policy['arr_1']
        self.theta3 = _policy['arr_2']
        
        # GRU
        self.x2r = _policy['arr_3']
        self.h2r = _policy['arr_4']
        self.x2z = _policy['arr_5']
        self.h2z = _policy['arr_6']
        self.x2h = _policy['arr_7']
        self.h2h = _policy['arr_8']
        
        self.normalizer.set_n_mu_diff(_policy['arr_9'],
                                      _policy['arr_10'],
                                      _policy['arr_11'])

    def reset(self):
        self.hx = np.zeros((self.gru_hidden,))
        # pass

    def evaluate(self, input):
        # WITH GRU
        r = sigmoid(self.x2r.dot(input[:self.gru_input])+self.h2r.dot(self.hx))
        z = sigmoid(self.x2z.dot(input[:self.gru_input])+self.h2z.dot(self.hx))
        
        h_tilde = np.tanh(self.x2h.dot(input[:self.gru_input])+r*(self.h2h.dot(self.hx)))
        self.hx = self.hx*(1-z)+z*h_tilde
        
        return np.tanh(self.theta3.dot
                        (np.tanh(self.theta2.dot
                                    (np.tanh(self.theta1.dot(
                        np.concatenate((input,self.hx),axis=0)))))))

    def observe(self, x):
        self.normalizer.observe(x)
        
    def normalize(self, inputs):
        return self.normalizer.normalize(inputs)

def main():

    env = env_loader.load("GUI", "control_velocity")  
    nb_inputs = env.get_state_space()
    nb_outputs = env.get_action_space()
    policy = Policy(nb_inputs, 64, 29, 8, nb_outputs)
    
    # policy.load_policy('./saved_policies/pre_trained.npz')
    policy.load_policy('./saved_policies/generalist.npz')
    # policy.load_policy('./saved_policies/close_generalist.npz')
    
    # Visualise the entire population
    encoding = np.load('./saved_policies/encoding.npz')['arr_0']
    
    total_reward_count = 0
    
    # flat_encoding = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    for i in range(len(encoding)):
        state = env.hard_reset(create_poet_env(encoding[i]))[0]
        # env.start_recording(str(i))
        policy.reset()
        done = False
        num_plays = 0
        sum_rewards = 0
        while not done and num_plays < 600:
            env.focus_camera_on_robot()
            
            policy.observe(state)
            state = policy.normalize(state)
            action = policy.evaluate(state)
            state, reward, done = env.step(action)

            sum_rewards += reward
            num_plays += 1
        print(sum_rewards)
        total_reward_count += sum_rewards
    #     # env.stop_recording(str(i))
        
    # TEST THE POLICY ON SPECIFIC ENVIRONMENTS
    # mountains = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    # steps = np.array([0, 0, 1, 0.1, 0, 0, 0, 0])
    # hills = np.array([0, 0, 0, 0, 1, 0.8, 0, 0])
    # stairs = np.array([0, 0, 0, 0, 0, 0, 1.2, 0.1])
    
    # # Choose the encoding 
    # encoding = stairs
    
    # np.random.seed(1)
    # random.seed(1)
    # state = env.hard_reset(create_poet_env(encoding))[0]
    # # env.start_recording('hills')
    # policy.reset()
    # done = False
    # num_plays = 0
    # sum_rewards = 0
    # while not done and num_plays < 1200:
    #     env.focus_camera_on_robot()
        
    #     policy.observe(state)
    #     state = policy.normalize(state)
    #     action = policy.evaluate(state)
    #     state, reward, done = env.step(action)

    #     sum_rewards += reward
    #     num_plays += 1
    # print(sum_rewards)
    # total_reward_count += sum_rewards
    # # env.stop_recording('hills')

if __name__ == '__main__':
    main()