# AI 2018
# Implementation of ARS adapted from PyBullet
# https://github.com/bulletphysics/bullet3

# Used for the pre-training phase.

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# Importing the libraries
import pickle
import math
import os
import numpy as np
import pybullet_envs
import time
import multiprocessing as mp
from multiprocessing import Process, Pipe
import argparse

from environment import env_loader

# Setting the Hyper Parameters
class Hp():

    def __init__(self):
        self.nb_steps = 3000
        self.episode_length = 480
        self.learning_rate = 0.001
        self.nb_directions = 48
        self.nb_best_directions = 24
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.005
        self.seed = 187
        #self.env_name = 'HalfCheetahBulletEnv-v0'
        
    def change_lr_noise(self, lr, noise):
        self.learning_rate += lr
        self.noise += noise

# Multiprocess Exploring the policy on one specific direction and over one episode

_RESET = 1
_CLOSE = 2
_EXPLORE = 3

def ExploreWorker(rank, childPipe):
    # nb_inputs = env.observation_space.shape[0]
    env = env_loader.load("DIRECT", "control_velocity")
    nb_inputs = env.get_state_space()
    normalizer = Normalizer(nb_inputs)
    n = 0
    while True:
        n += 1
        try:
            # Only block for short times to have keyboard exceptions be raised.
            if not childPipe.poll(0.001):
                continue
            message, payload = childPipe.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if message == _RESET:
            observation_n = env.soft_reset()[0]
            childPipe.send(["reset ok"])
            continue
        if message == _EXPLORE:
            #normalizer = payload[0] #use our local normalizer
            policy = payload[1]
            hp = payload[2]
            direction = payload[3]
            delta = payload[4]
            state = env.soft_reset()[0]
            done = False
            num_plays = 0.
            sum_rewards = 0
            while not done and num_plays < hp.episode_length:
                normalizer.observe(state)
                state = normalizer.normalize(state)
                action = policy.evaluate(state, delta, direction, hp)
                state, reward, done = env.step(action)
                #reward += reward#max(min(reward, 1), -1)
                sum_rewards += reward
                num_plays += 1
            childPipe.send([sum_rewards])
            continue
        if message == _CLOSE:
            childPipe.send(["close ok"])
            break
    childPipe.close()


# Normalizing the states
class Normalizer():

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std
    
    def get_n_mu_diff(self):
        return self.n, self.mean, self.mean_diff
    
    def save_n_mu_diff(self):
        self._n = self.n
        self._mean = self.mean
        self._mean_diff = self.mean_diff    

    def get_saved_n_mu_diff(self):
        return self._n, self._mean, self._mean_diff


def sigmoid(x):
    return 1/(1+np.exp(-x))


class Policy():

    def __init__(self, input_size, hidden_size, gru_hidden, output_size, normalizer, args):
        self.gru_hidden = gru_hidden
        self.input_size = input_size
        try:
            # MLP
            self.theta1 = np.load(args.policy[0])
            self.theta2 = np.load(args.policy[1])
            self.theta3 = np.load(args.policy[2])
            
        except:
            # MLP
            self.theta1 = np.zeros((hidden_size*2, input_size))
            self.theta2 = np.zeros((hidden_size, hidden_size*2))
            self.theta3 = np.zeros((output_size, hidden_size))
            
        print("Starting policy theta=", self.theta1, self.theta2, self.theta3)
        self.normalizer = normalizer
        print(self.theta1.shape)
        parameters = self.theta1.size+self.theta2.size+self.theta3.size
        print(f"{parameters = }")
    

    def evaluate(self, input, delta, direction, hp):
        if direction is None:            
            return np.tanh(self.theta3.dot
                           (np.tanh(self.theta2.dot
                                       (np.tanh(self.theta1.dot(
                            input))))))
            
            # return np.tanh(self.theta3.dot
            #                (np.maximum(self.theta2.dot
            #                            (np.maximum(self.theta1.dot(
            #                 input),0)),0)))
            
        elif direction == "positive":
            return np.tanh((self.theta3 + hp.noise * delta[2]).dot(
                        np.tanh((self.theta2 + hp.noise * delta[1]).dot(
                        np.tanh((self.theta1 + hp.noise * delta[0]).dot(
                        input))))))
            
            # return np.tanh((self.theta3 + hp.noise * delta[2]).dot(
            #             np.maximum((self.theta2 + hp.noise * delta[1]).dot(
            #             np.maximum((self.theta1 + hp.noise * delta[0]).dot(
            #             input),0)),0)))
            
        else:
            return np.tanh((self.theta3 - hp.noise * delta[2]).dot(
                        np.tanh((self.theta2 - hp.noise * delta[1]).dot(
                        np.tanh((self.theta1 - hp.noise * delta[0]).dot(
                        input))))))
            
            # return np.tanh((self.theta3 + hp.noise * delta[2]).dot(
            #             np.maximum((self.theta2 + hp.noise * delta[1]).dot(
            #             np.maximum((self.theta1 + hp.noise * delta[0]).dot(
            #             input),0)),0)))

    def sample_deltas(self):
        return [(np.random.randn(*self.theta1.shape),
                np.random.randn(*self.theta2.shape),
                np.random.randn(*self.theta3.shape)) for _ in range(
            hp.nb_directions)]

    def update(self, rollouts, sigma_r, args):
        step1 = np.zeros(self.theta1.shape)
        step2 = np.zeros(self.theta2.shape)
        step3 = np.zeros(self.theta3.shape)

        for r_pos, r_neg, d in rollouts:
            step1 += (r_pos - r_neg) * d[0]
            step2 += (r_pos - r_neg) * d[1]
            step3 += (r_pos - r_neg) * d[2]

        self.theta1 += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step1
        self.theta2 += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step2
        self.theta3 += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step3
        

    def save(self, name):
        print(f"Save {name}")
        n, mu, diff = self.normalizer.get_saved_n_mu_diff()
        np.savez(args.logdir + name, self.theta1, self.theta2, self.theta3, 
                 n, mu, diff, allow_pickle=True)
        

# Exploring the policy on one specific direction and over one episode
def explore(env, normalizer, policy, direction, delta, hp):
    state = env.soft_reset()[0]
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction, hp)
        state, reward, done = env.step(action)
        #reward += reward#max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards, num_plays


# Training the AI

def train(env, policy, normalizer, hp, parentPipes, args):
    reward_list = [] 
    num_plays_per_rollout = []
    num_steps_save = 1000
    best_reward = 0
    
    for step in range(hp.nb_steps):
        
        # if (step%num_steps_save==0):
        #     normalizer.save_n_mu_diff()
        
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        if parentPipes:
            for k in range(hp.nb_directions):
                parentPipe = parentPipes[k]
                parentPipe.send([_EXPLORE, [normalizer, policy, hp, "positive", deltas[k]]])
            for k in range(hp.nb_directions):
                positive_rewards[k] = parentPipes[k].recv()[0]

            for k in range(hp.nb_directions):
                parentPipe = parentPipes[k]
                parentPipe.send([_EXPLORE, [normalizer, policy, hp, "negative", deltas[k]]])
            for k in range(hp.nb_directions):
                negative_rewards[k] = parentPipes[k].recv()[0]

        else:
            # Getting the positive rewards in the positive directions
            for k in range(hp.nb_directions):
                positive_rewards[k], _ = explore(env, normalizer, policy, "positive", deltas[k], hp)

             # Getting the negative rewards in the negative/opposite directions
            for k in range(hp.nb_directions):
                negative_rewards[k], _ = explore(env, normalizer, policy, "negative", deltas[k], hp)

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {
            k: max(r_pos, r_neg)
            for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
        }
        order = sorted(scores.keys(), key=lambda x: -scores[x])[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating our policy
        policy.update(rollouts, sigma_r, args)

        # Printing the final reward of the policy after the update
        reward_evaluation, time_steps = explore(env, normalizer, policy, None, None, hp)
        print('Step:', step, 'Reward:', reward_evaluation)
        
        if ((step+1)%num_steps_save==0):
            normalizer.save_n_mu_diff()
            policy.save("/" + str(step+1)+"_"+str(round(reward_evaluation,3)))
        if reward_evaluation>best_reward:
            best_reward = reward_evaluation
            normalizer.save_n_mu_diff()
            policy.save("/best")
        
        num_plays_per_rollout.append(time_steps)
        reward_list.append(round(reward_evaluation,3))
        hp.change_lr_noise(-1.8e-8,-1.8e-8)
    
    with open(args.logdir+"/reward_list.txt", "wb") as fp:
        pickle.dump(reward_list, fp)
        
    with open(args.logdir+"/num_plays_per_rollout.txt", "wb") as fp:
        pickle.dump(num_plays_per_rollout, fp)

# Running the main code

if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--steps', help='Number of steps', type=int, default=50000)
    parser.add_argument('--policy', help='Starting policy file (npy)', type=str, default='')
    parser.add_argument(
        '--logdir', help='Directory root to log policy files (npy)', type=str, default='.')
    parser.add_argument('--mp', help='Enable multiprocessing', type=int, default=1)

    args = parser.parse_args()

    hp = Hp()
    # hp.env_name = args.env
    hp.seed = args.seed
    hp.nb_steps = args.steps
    print("seed = ", hp.seed)
    np.random.seed(hp.seed)

    parentPipes = None
    if args.mp:
        num_processes = hp.nb_directions
        processes = []
        childPipes = []
        parentPipes = []

        for pr in range(num_processes):
            parentPipe, childPipe = Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        for rank in range(num_processes):
            p = mp.Process(target=ExploreWorker, args=(rank, childPipes[rank]))
            p.start()
            processes.append(p)
            
    env = env_loader.load("DIRECT", "control_velocity")
    nb_inputs = env.get_state_space()
    nb_outputs = env.get_action_space()
    print(nb_outputs)
    normalizer = Normalizer(nb_inputs)
    policy = Policy(nb_inputs, 64, nb_inputs, nb_outputs, normalizer, args)

    print("start training")
    train(env, policy, normalizer, hp, parentPipes, args)

    if args.mp:
        for parentPipe in parentPipes:
            parentPipe.send([_CLOSE, "pay2"])

        for p in processes:
            p.join()
