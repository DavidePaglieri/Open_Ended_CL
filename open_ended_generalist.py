# Re-implmenetation of POET with ARS instead of ES and with A1 robot on
# procedurally generated environments

import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
import argparse
from scenes.poet_env import create_poet_env
from copy import deepcopy
import random
import time

from environment import env_loader

class Hp():

    def __init__(self):
        self.nb_steps = 5
        self.episode_length = 420
        self.learning_rate = 0.001
        self.nb_directions = 32
        self.nb_best_directions = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.005
        self.seed = 187
        #self.env_name = 'HalfCheetahBulletEnv-v0'

# Multiprocess Exploring the policy on one specific direction and over one episode

_RESET = 1
_CLOSE = 2
_EXPLORE = 3
_CHANGE = 4
_EVALUATE = 5

def ExploreWorker(rank, childPipe):
    env = env_loader.load("DIRECT", "control_velocity")
    nb_inputs = env.get_state_space()
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
            policy.reset()
            childPipe.send(["reset ok"])
            continue
        
        if message == _CHANGE:
            E = payload[0]
            env.hard_reset(E)
            childPipe.send(["change ok"])
            continue
        
        if message == _EVALUATE:
            E = payload[0]
            policy = payload[1]
            hp = payload[2]
            state = env.hard_reset(E)[0]
            policy.reset()
            done = False
            num_plays = 0.
            sum_rewards = 0
            while not done and num_plays < hp.episode_length:
                policy.observe(state)
                state = policy.normalize(state)
                action = policy.evaluate(state, None, None, hp)
                state, reward, done = env.step(action)
                sum_rewards += reward
                num_plays += 1
            childPipe.send([sum_rewards])
            continue
        
        if message == _EXPLORE:
            policy = payload[0]
            hp = payload[1]
            direction = payload[2]
            delta = payload[3]
            state = env.soft_reset()[0]
            policy.reset()
            done = False
            num_plays = 0.
            sum_rewards = 0
            while not done and num_plays < hp.episode_length:
                policy.observe(state)
                state = policy.normalize(state)
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

    def __init__(self, input_size, hidden_size, gru_input, gru_hidden, output_size, args):
        self.gru_input = gru_input
        self.gru_hidden = gru_hidden
        self.input_size = input_size
        self.normalizer = Normalizer(input_size)
        try:
            # MLP
            # self.theta1 = np.load(args.policy[0])
            # self.theta2 = np.load(args.policy[1])
            
            # GRU
            self.theta1 = np.load(args.policy[0])
            self.theta2 = np.load(args.policy[1])
            self.theta3 = np.load(args.policy[2])
            self.x2h = np.load(args.policy[3])
            self.h2h = np.load(args.policy[4])
            self.x2r = np.load(args.policy[5])
            self.h2r = np.load(args.policy[6])
            self.x2z = np.load(args.policy[7])
            self.h2z = np.load(args.policy[8])
        except:
            # MLP
            # self.theta1 = np.zeros((hidden_size, input_size))
            # self.theta2 = np.zeros((output_size, hidden_size))
            
            # GRU
            self.theta1 = np.zeros((hidden_size, input_size+gru_hidden))
            self.theta2 = np.zeros((hidden_size, hidden_size))
            self.theta3 = np.zeros((output_size, hidden_size))
            self.x2h = np.zeros((gru_hidden, gru_input))
            self.h2h = np.zeros((gru_hidden, gru_hidden))
            self.x2r = np.zeros((gru_hidden, gru_input))
            self.h2r = np.zeros((gru_hidden, gru_hidden))
            self.x2z = np.zeros((gru_hidden, gru_input))
            self.h2z = np.zeros((gru_hidden, gru_hidden))
        
            self.hx = np.zeros((gru_hidden,))
            
        print("Starting policy theta=", self.theta1, self.theta2, self.theta3)
        parameters = self.theta1.size+self.theta2.size+self.theta3.size+3*self.x2h.size+3*self.h2h.size
        print(f"{parameters = }")
        
        
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
    

    def evaluate(self, input, delta, direction, hp):
        if direction is None:
            # WITH GRU
            r = sigmoid(self.x2r.dot(input[:self.gru_input])+self.h2r.dot(self.hx))
            z = sigmoid(self.x2z.dot(input[:self.gru_input])+self.h2z.dot(self.hx))
            
            h_tilde = np.tanh(self.x2h.dot(input[:self.gru_input])+r*(self.h2h.dot(self.hx)))
            self.hx = self.hx*(1-z)+z*h_tilde
            
            return np.tanh(self.theta3.dot
                           (np.tanh(self.theta2.dot
                                       (np.tanh(self.theta1.dot(
                            np.concatenate((input,self.hx),axis=0)))))))
            
            # return np.tanh(self.theta3.dot(np.maximum(
            #     self.theta2.dot(np.maximum(self.theta1.dot(
            #                 np.concatenate((input,self.hx),axis=0)),0)),0)))
            
            # NO GRU
            # return np.tanh(self.theta2.dot(np.maximum(self.theta1.dot(
            #                 input),0)))
            
        elif direction == "positive":
            # WITH GRU            
            r = sigmoid((self.x2r+hp.noise*delta[3]).dot(input[:self.gru_input])+
                        (self.h2r+hp.noise*delta[4]).dot(self.hx))
            z = sigmoid((self.x2z+hp.noise*delta[5]).dot(input[:self.gru_input])+
                        (self.h2z+hp.noise*delta[6]).dot(self.hx))
            h_tilde = np.tanh((self.x2h+hp.noise*delta[7]).dot(input[:self.gru_input])+
                              r*((self.h2h+hp.noise*delta[8]).dot(self.hx)))
            self.hx = self.hx*(1-z)+z*h_tilde
            
            return np.tanh((self.theta3 + hp.noise * delta[2]).dot(
                        np.tanh((self.theta2 + hp.noise * delta[1]).dot(
                        np.tanh((self.theta1 + hp.noise * delta[0]).dot(
                        np.concatenate((input,self.hx),axis=0)))))))
            
            # return np.tanh((self.theta3 + hp.noise * delta[2]).dot(
            #     np.maximum((self.theta2 + hp.noise * delta[1]).dot(
            #     np.maximum((self.theta1 + hp.noise * delta[0]).dot(
            #          np.concatenate((input,self.hx),axis=0)),0)),0)))
            
            # NO GRU 
            # return np.tanh((self.theta2 + hp.noise * delta[1]).dot(
            #     np.maximum((self.theta1 + hp.noise * delta[0]).dot(input),0)))
            
        else:
           # WITH GRU 
            r = sigmoid((self.x2r-hp.noise*delta[3]).dot(input[:self.gru_input])+
                        (self.h2r-hp.noise*delta[4]).dot(self.hx))
            z = sigmoid((self.x2z-hp.noise*delta[5]).dot(input[:self.gru_input])+
                        (self.h2z-hp.noise*delta[6]).dot(self.hx))
            h_tilde = np.tanh((self.x2h-hp.noise*delta[7]).dot(input[:self.gru_input])+
                              r*((self.h2h-hp.noise*delta[8]).dot(self.hx)))
            self.hx = self.hx*(1-z)+z*h_tilde
            
            return np.tanh((self.theta3 - hp.noise * delta[2]).dot(
                        np.tanh((self.theta2 - hp.noise * delta[1]).dot(
                        np.tanh((self.theta1 - hp.noise * delta[0]).dot(
                        np.concatenate((input,self.hx),axis=0)))))))
            
            
            # return np.tanh((self.theta3 - hp.noise * delta[2]).dot(
            #     np.maximum((self.theta2 - hp.noise * delta[1]).dot(
            #     np.maximum((self.theta1 - hp.noise * delta[0]).dot(
            #          np.concatenate((input,self.hx),axis=0)),0)),0)))
            
            # NO GRU
            # return np.tanh((self.theta2 - hp.noise * delta[1]).dot(
            #     np.maximum((self.theta1 - hp.noise * delta[0]).dot(
            #          input),0)))

    def sample_deltas(self):
        # WITH GRU
        return [(np.random.randn(*self.theta1.shape),
                np.random.randn(*self.theta2.shape),
                np.random.randn(*self.theta3.shape),
                np.random.randn(*self.x2r.shape),
                np.random.randn(*self.h2r.shape),
                np.random.randn(*self.x2z.shape),
                np.random.randn(*self.h2z.shape),
                np.random.randn(*self.x2h.shape),
                np.random.randn(*self.h2h.shape)) for _ in range(
            hp.nb_directions)]
        
        # NO GRU
        # return [(np.random.randn(*self.theta1.shape),
        #         np.random.randn(*self.theta2.shape)) for _ in range(
        #     hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step1 = np.zeros(self.theta1.shape)
        step2 = np.zeros(self.theta2.shape)
        step3 = np.zeros(self.theta3.shape)
        step4 = np.zeros(self.x2r.shape)
        step5 = np.zeros(self.h2r.shape)
        step6 = np.zeros(self.x2z.shape)
        step7 = np.zeros(self.h2z.shape)
        step8 = np.zeros(self.x2h.shape)
        step9 = np.zeros(self.h2h.shape)
        for r_pos, r_neg, d in rollouts:
            step1 += (r_pos - r_neg) * d[0]
            step2 += (r_pos - r_neg) * d[1]
            step3 += (r_pos - r_neg) * d[2]
            step4 += (r_pos - r_neg) * d[3]
            step5 += (r_pos - r_neg) * d[4]
            step6 += (r_pos - r_neg) * d[5]
            step7 += (r_pos - r_neg) * d[6]
            step8 += (r_pos - r_neg) * d[7]
            step9 += (r_pos - r_neg) * d[8]
        self.theta1 += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step1
        self.theta2 += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step2
        self.theta3 += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step3
        self.x2r += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step4
        self.h2r += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step5
        self.x2z += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step6
        self.h2z += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step7
        self.x2h += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step8
        self.h2h += hp.learning_rate/(hp.nb_best_directions * sigma_r)*step9
        

    def save(self, name):
        print(f"Save {name}")
        self.normalizer.save_n_mu_diff()
        n, mu, diff = self.normalizer.get_saved_n_mu_diff()
        # print(mu)
        
        # WITH GRU
        np.savez(args.logdir + name, self.theta1, self.theta2, self.theta3,
                 self.x2r, self.h2r, self.x2z, self.h2z, self.x2h, self.h2h, 
                 n, mu, diff, allow_pickle=True)
        
        # NO GRU
        # np.savez(args.logdir + name, self.theta1, self.theta2, 
        #          n, mu, diff, allow_pickle=True)
        
        
    def observe(self, x):
        self.normalizer.observe(x)
        
    def normalize(self, inputs):
        return self.normalizer.normalize(inputs)
           

# Exploring the policy on one specific direction and over one episode
def explore_new_env(env, heightfield, policy, direction, delta, hp):
    state = env.hard_reset(heightfield)[0]
    policy.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        policy.observe(state)
        state = policy.normalize(state)
        action = policy.evaluate(state, delta, direction, hp)
        state, reward, done = env.step(action)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards


def explore(env, policy, direction, delta, hp):
    state = env.soft_reset()[0]
    policy.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        policy.observe(state)
        state = policy.normalize(state)
        action = policy.evaluate(state, delta, direction, hp)
        state, reward, done = env.step(action)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards


def train(env, heightfield, policy, hp, parentPipes):

    # Change environment
    for k in range(hp.nb_directions):
        parentPipe = parentPipes[k]
        parentPipe.send([_CHANGE, [heightfield]])
    for k in range(hp.nb_directions):
        parentPipes[k].recv()
    
    for step in range(hp.nb_steps):
        # print(f"{step = }")
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        if parentPipes:
            for k in range(hp.nb_directions):
                parentPipe = parentPipes[k]
                parentPipe.send([_EXPLORE, [policy, hp, "positive", deltas[k]]])
            for k in range(hp.nb_directions):
                positive_rewards[k] = parentPipes[k].recv()[0]

            for k in range(hp.nb_directions):
                parentPipe = parentPipes[k]
                parentPipe.send([_EXPLORE, [policy, hp, "negative", deltas[k]]])
            for k in range(hp.nb_directions):
                negative_rewards[k] = parentPipes[k].recv()[0]

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
        policy.update(rollouts, sigma_r)
        # print(f"{step = }")
    
    
    return policy, explore_new_env(env, heightfield, policy, None, None, hp)


def evaluate_agents(E, policies, hp, parentPipes):
    
    reward_list = []
    # Change environment E
    heightfield = create_poet_env(E)
    for k in range(len(policies)):
        parentPipe = parentPipes[k]
        parentPipe.send([_EVALUATE, [heightfield, policies[k], hp]])
    for k in range(len(policies)):
        reward_list.append(parentPipes[k].recv()[0])

    best_reward = max(reward_list)

    return deepcopy(policies[reward_list.index(best_reward)]), best_reward


def mutate(encoding):
    return np.minimum(np.maximum(encoding +\
                        np.random.uniform(low=-0.4,high=0.4, size=8), 
                        np.zeros(8)), np.ones(8)) + np.ones(8)*0.01


def env_reproduce_MC(parent_list, A, hp, parentPipes):
    child_list = []
    new_env = []
    reward_list = []
    for k in range(len(parent_list)):
        new_env.append(mutate(parent_list[k][0]))
        # print(new_env[k])
        parentPipe = parentPipes[k]
        parentPipe.send([_EVALUATE, [create_poet_env(new_env[k]), 
                                     A, hp]])
    for k in range(len(parent_list)):
        reward = parentPipes[k].recv()[0]
        if 38<reward:
            child_list.append((new_env[k], reward))
    return child_list


def compute_novelty_vs_archive(child, archive):
    distances = []
    for point, _ in archive:
        distances.append(np.linalg.norm(point-child[0]))
    top_5_indices = np.argsort(distances)[:5]
    top_k = np.take(distances, top_5_indices)
    return top_k.mean()


def rank_by_novelty(child_list, archive):
    if len(archive)<5:
        return child_list
    return sorted(child_list, key=lambda x: compute_novelty_vs_archive(x, archive), reverse=True)


def remove_oldest(child_list, num_removals):
    for i in range(num_removals):
        del child_list[0]


def mutate_envs(ER_list, A, parentPipes, archive):
    # ADD INTERPOLATION OF HEIGHTFIELD
    print("Mutating")
    
    MAX_CHILDREN = 1
    MAX_ADMITTED = 1
    CAPACITY = 32
    
    parent_list = []
    M = len(ER_list)
    for m in range(M):
        E, R = ER_list[m]
        # Eligible to reproduce
        print(f"Parent Reward {R}")
        if R>38:
            parent_list.append((E,R))

    # Reproduce and MC satisfied together
    child_list = env_reproduce_MC(parent_list, A, hp, parentPipes)
    child_list = rank_by_novelty(child_list, archive)
    admitted = 0

    ER_list.append(child_list[0])
    archive.append(child_list[0])
    
    M = len(ER_list)
    if M>CAPACITY:
        print("Removing oldest")
        num_removals = M-CAPACITY
        remove_oldest(ER_list, num_removals)
    return ER_list


def full_test(A, hp, parentPipes):
    
    E = np.array([
        [0.8, 0.8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.9, 0.2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.7, 0.80, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.8, 0.30],
        
        [0.70, 0.7, 0.2, 0.2, 0.2, 0.2, 0.2, 0.20],
        [0.20, 0.2, 0.8, 0.2, 0.2, 0.2, 0.2, 0.20],
        [0.20, 0.2, 0.2, 0.2, 0.6, 0.7, 0.2, 0.20],
        [0.20, 0.2, 0.2, 0.2, 0.2, 0.2, 0.7, 0.30]
    ])
    
    total_reward_count = 0
    for i in range(len(E)):
        random.seed(1)
        np.random.seed(1)   
        heightfield = create_poet_env(E[i])
        reward_list = []

        parentPipe = parentPipes[i]
        parentPipe.send([_EVALUATE, [heightfield, A, hp]])
        
    for k in range(len(E)):
        reward = parentPipes[k].recv()[0]
        reward_list.append(reward*32)
    total_reward_count += sum(reward_list)
        
    return total_reward_count


# Running the main code

if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--steps', help='Number of steps per training', type=int, default=10)
    parser.add_argument('--iterations', help='Number of iterations', type=int, default=101)
    parser.add_argument('--policy', help='Starting policy file (npy)', type=str, default='')
    parser.add_argument(
        '--logdir', help='Directory root to log policy files (npy)', type=str, default='.')
    parser.add_argument('--max_num_envs', type=int, default=32)
    
    args = parser.parse_args()

    hp = Hp()
    # hp.env_name = args.env
    hp.seed = int(time.time()*1000)%100000
    hp.nb_steps = args.steps
    print("seed = ", hp.seed)
    np.random.seed(hp.seed)

    parentPipes = None
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
    E = np.zeros(8)
    A = Policy(nb_inputs, 64, 29, 8, nb_outputs, args)
    A.load_policy('pre_trained.npz')
    R = explore_new_env(env, create_poet_env(E), A, None, None, hp)
    
    # Environment - Policy Pair List
    ER_list = []
    archive = []
    
    print("Initial loading")
    for i in range(32):
        E = np.zeros(8)
        pair_init = (E, R)
        ER_list.append(pair_init)
        archive.append(pair_init)

    N_mutate = 1# mutation interval/freqeuncy
    N_transfer = 1# mutation transfer/frequency
    
    compare = []

    #---------- POET ----------#
    for i in range(args.iterations):
        print(f"\nIteration {i}")
        #----- GENERATE NEW ENVIRONMENTS -----#
        if i>0 and i%N_mutate==0:
            print("GENERATING TERRAINS")
            ER_list = mutate_envs(ER_list, A, parentPipes, archive)

        num_pairs = len(ER_list)

        #----- OPTIMIZE POLICY IN EACH ENV -----#
        print("TRAINING")
        for j in range(num_pairs):
            print(j)
            E, _ = ER_list[j]
            A, R = train(env, create_poet_env(E), A, hp, parentPipes)
            ER_list[j] = (E, R)

        if (i % 10) == 0:
            compare.append(full_test(A, hp, parentPipes))
            print(compare[-1])

    encoding_list = []
    reward_list = []
    # Saving everything
    for count, ER_pair in enumerate(ER_list):
        encoding_list.append(ER_pair[0])
        reward_list.append(ER_pair[1])
    
    A.save('/generalist.npz')
    
    print(f" {encoding_list}")
    print(f" {reward_list}")
    
    encoding_array = np.asarray(encoding_list)
    reward_array = np.asarray(reward_list)
    comparison = np.asarray(compare)

    print(comparison)
    # np.savez(args.logdir + '/archive', encoding_array, allow_pickle=True)
    np.savez(args.logdir + '/full_test', comparison, allow_pickle=True)
    np.savez(args.logdir + '/encoding', encoding_array, allow_pickle=True)
    np.savez(args.logdir + '/reward', reward_array, allow_pickle=True)

    for parentPipe in parentPipes:
        parentPipe.send([_CLOSE, "pay2"])

    for p in processes:
        p.join()