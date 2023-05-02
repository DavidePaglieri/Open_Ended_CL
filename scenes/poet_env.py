import numpy as np
from scenes.fractal import (generate_perlin_noise_2d, generate_fractal_noise_2d)
from opensimplex import OpenSimplex
import math
import time
import random
import pybullet as p
from environment import env_loader


size = 256

def create_fractal_map(amplitude, persistence):
    return generate_fractal_noise_2d(shape=(size,size), 
                                res=(2,2), 
                                octaves=6,
                                amplitude=amplitude,
                                # tileable=(True, True),
                                persistence=persistence)


def create_steps_map(amplitude, step):
    step = math.ceil(step)+1
    a_ = np.zeros((size,size))
    a = np.random.random( [size//step, size//step] )*amplitude
    a_[:(size//step*step),:(size//step)*step] = a.repeat( step, axis=0 ).repeat( step, axis=1 )
    return a_
    
    
def create_opensimplex_map(amplitude, feature_size=24):
    num = random.randint(0,1000)
    a = np.zeros((size, size))    
    simplex = OpenSimplex(0)
    a = simplex.noise2array((np.arange(size)+num)/feature_size, (np.arange(size)+num)/feature_size)*amplitude
    return a


def create_stairs_map(height, width):
    
    a = np.zeros((1,size))
    width = math.ceil(width)
    inc = 0
    i = 0
    while i<size:
        for j in range(i, i+1+width,1):
            if j>=size:
                break
            a[0,j] = inc
        i += 1+width
        # if not 46<i<64:
        inc += height

    return a.repeat(size, axis=0)


def create_poet_env(feature_list):
    # scaling the feature list and creating the environment.
    enc = feature_list * [0.1, 0.8, 0.1, 30, 0.1, 10, 0.05, 30]
    enc[5] += 10
    enc[7] += 8
    a = create_fractal_map(enc[0], enc[1])
    b = create_steps_map(enc[2], enc[3])
    c = create_opensimplex_map(enc[4], enc[5])
    d = create_stairs_map(enc[6], enc[7])
    return (a+b+c+d).reshape(-1)


# Sensible ranges
# Mountains [ ~U(0 , 0.01) ; ~U(0 , 1.3) ]
# Steps [ ~U(0 , 0.1 ) ; ~U(0 , 10) ]
# Hills [ ~U(0, 0.05) ; ~U(10,20) ]
# Stairs [ ~U(0 , 0.05) ; ~U(8 , 32) ]


class Environment():
    # This class is not used anymore
    """Environment class, the encoding values are in the range [0,1]
    """
    def __init__(self, encoding=None):
        
        if encoding is None:
            self.encoding = np.zeros(8)
        else:
            self.encoding = encoding
            
    def mutate(self):
        self.encoding += np.minimum(np.maximum(np.random.uniform(low=-0.2,high=0.2, size=8),
                                np.ones(8)), np.zeros(8))
        
    def mutate_copy(self):
        return Environment(np.minimum(np.maximum(np.random.uniform(low=-0.2,high=0.2, size=8),
                                np.ones(8)), np.zeros(8)))
    
    def get_heightfield(self):
        return create_poet_env(self.encoding)
    
    def get_encoding(self):
        return self.encoding
    
    def copy(self):
        return Environment(self.encoding)