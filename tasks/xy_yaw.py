"""A simple task where the reward is based on how close the robot velociy is
to the desired velocity"""

from tasks import sample_commands
import math
import numpy as np
import random

class Task(object):
    """A simple task where the reward is based on how close the robot velociy is
    to the desired velocity"""
    
    
    def __init__(self, robot=None):
        """Init of the Task

        Args:
            desired_velocity: velocity that will achieve the maximum reward
            max_desired_velocity: maximum desired velocity
        """
        self.robot = robot
        self.desired_lv = 0.25
        self.command = np.array([1.0, 0., 0.])
        self.stop_command = False
        self.sample = sample_commands.Sample_Command('RANDOM', 'HILLS')
        self.foot_position = np.zeros((3,12))
        self.idx = -1
        self.r_lv = 0
        self.r_av = 0
        self.r_s = 0
        self.r_br = 0
        self.r_bp = 0
        self.r_t = 0  
        
        
    def reset(self, robot):
        """Initializes a new instance of the robot and resets the 
        desired velocity"""
        self.robot = robot
        
        
    def set_desired_yaw_rate(self, yaw_rate):
        print()
        """Sets a new desired yaw rate"""
        self.command[2] = yaw_rate
        
        
    def get_desired_velocities_and_yaw_rate(self):
        """Get desired direction of the robot CoM and yaw rate"""
        self.command = self.sample.get_command(
            self.robot.get_base_position()[:2],
            self.robot.get_base_roll_pitch_yaw()[2])
        return self.command
        # return np.array([1.0, 0.])
        # return (self.desired_velocity_x, self.desired_velocity_y,
        #     self.desired_yaw_rate)
    
    
    def stop(self, bool):
        self.stop_command = bool
        
        
    def get_reward_distribution(self):
        r_lv, r_av, r_s, r_br, r_bp, r_t = self.r_lv, self.r_av, self.r_s, self.r_br, self.r_bp, self.r_t
        self.r_lv, self.r_av, self.r_s, self.r_br, self.r_bp, self.r_t = 0, 0, 0, 0, 0, 0
        return r_lv, r_av, r_s, r_br, r_bp, r_t
    
    
    # MY REWARD
    
    def get_reward(self, measurements, action):
        """Get the reward for the current time step
        
        Args:
            measurements: The robot's current filtered x,y velocity, roll rate,
                pitch rate, and yaw rate.
        
        Returns:
            float: reward obtained in the current time step
        """
        # print(self.command)
        
        
        # LINEAR VELOCITY
        v_pr = np.dot(measurements[:2], self.command[:2])
        if v_pr > self.desired_lv:
            r_lv = 1
        else:
            r_lv =  0.5*math.exp(-15*((measurements[0]-self.command[0]*self.desired_lv)**2))+\
                    0.5*math.exp(-15*((measurements[1]-self.command[1]*self.desired_lv)**2))

        # MY ANGULAR REWARD
        r_av = math.exp(-10*((measurements[4]-self.command[2]*0.3)**2))

        # TARGET SMOOTHNESS        
        self.idx = (self.idx + 1)%3
        self.foot_position[self.idx, :] = action
        r_s = - np.linalg.norm(self.foot_position[self.idx,:]
                        -2*self.foot_position[self.idx-1,:]
                        + self.foot_position[self.idx-2,:], ord=2)

        # BASE MOTION REWARD (Roll rate and pitch rate)
        r_br = math.exp(-2*(abs(measurements[2])))
        r_bp = math.exp(-2*(abs(measurements[3])))

        # TORQUE PENALTY
        r_t = -self.robot.get_torques()
        
        # PLOTTING
        # self.r_lv += 0.07*r_lv
        # self.r_av += 0.02*r_av
        # self.r_br += 0.005*r_br
        # self.r_bp += 0.01*r_bp
        # self.r_t -= 0.00002*r_t
        # self.r_s -= 0.025*r_s

        return 0.07*r_lv + 0.02*r_av + 0.025*r_s + 0.005*r_br + 0.01*r_bp + 0.00002*r_t
    
    
    def get_num_commands(self):
        return self.command.shape[0]
    
    
    def check_default_terminal_condition(self):
        """Returns true if the robot is in a position that should terminate
        the episode, false otherwise"""
        
        roll, pitch, _ = self.robot.get_base_roll_pitch_yaw()
        pos = self.robot.get_base_position()
        # print(pos[2])
        # print(roll, pitch, pos[2])
        return abs(roll) > 1 or abs(pitch) > 1 or pos[2] < 0.22