import math
import numpy as np
import random

class Task(object):
    
    def __init__(self, robot=None):
        """Init of the Task

        Args:
            desired_velocity: velocity that will achieve the maximum reward
            max_desired_velocity: maximum desired velocity
        """
        self.starting_command = np.array([1., 0., 0.])
        self.sample = Sample_Command()
        self.reset(robot)
        self.desired_lv = 0.25
        self.angle = 0
        self.foot_position = np.zeros((3,12))
        self.idx = -1
        self.r_lv = 0
        self.r_av = 0
        self.r_s = 0
        self.r_br = 0
        self.r_bp = 0
        self.r_t = 0  
        
        
    def reset(self, robot, command_mode=1):
        """Initializes a new instance of the robot and resets the 
        desired velocity"""
        self.robot = robot
        self.command = self.starting_command
        # self.command[0] = random.uniform(0,1)
        # self.command[1] = random.uniform(-1.0, 1.0)
        # print(self.command)
        # self.sample.reset(command_mode)
        
        # 3 conditioned
        
        
    def set_desired_yaw_rate(self, yaw_rate):
        """Sets a new desired yaw rate"""
        self.command[1] = yaw_rate
        
        
    def change_desired_yaw_rate(self, change):
        self.command[1] += change
        self.command[1] = min(max(self.command[1],-1),1)
        # print(self.command[2])
        
    
    def change_desired_forward_velocity(self, change):
        self.command[0] += change
        self.command[0] = min(max(self.command[0],0),1)
        
    def enable_command(self):
        self.stop_command = False
        
        
    def get_desired_velocities_and_yaw_rate(self):
        """Get desired direction of the robot CoM and yaw rate"""
        # self.command = self.sample.sample_command(
        #                 self.robot.get_base_position()[:2],
        #                 self.robot.get_base_roll_pitch_yaw()[2],
        #                 1)
        # print(self.command)
        return self.command

    
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
        
        # MY LINEAR VELOCITY
        v_pr = np.dot(measurements[0], self.command[0])
        if v_pr > self.desired_lv:
            r_lv = 1
        else:
            r_lv =  0.5*math.exp(-15*((measurements[0]-self.command[0]*self.desired_lv)**2))+\
                    0.5*math.exp(-15*((measurements[1]-self.command[1]*self.desired_lv)**2))

        # MY ANGULAR REWARD
        v_ar = np.dot(measurements[4], self.command[2])
        if v_ar > 0.3:
            r_av = 1
        else:
            r_av = math.exp(-15*((measurements[4]-self.command[2]*0.3)**2))

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

        return 0.05*r_lv + 0.05*r_av + 0.025*r_s + 0.005*r_br + 0.01*r_bp + 0.00002*r_t
    
    
    def get_num_commands(self):
        return self.command.shape[0]
    
    
    def check_default_terminal_condition(self):
        """Returns true if the robot is in a position that should terminate
        the episode, false otherwise"""
        
        roll, pitch, _ = self.robot.get_base_roll_pitch_yaw()
        pos = self.robot.get_base_position()
        # return False
        return abs(roll) > 1 or abs(pitch) > 1 #or pos[2] < 0.22
        
        
class Sample_Command():
    # Used to train x,y,yaw command conditioned.
    
    def __init__(self):
        self.reset()
        self._goal_position = np.array([100,0])
        
        
    def sample_command(self, robot_position, heading_angle, command_mode=0):
        
        dist = self._goal_position - robot_position
        direction = np.arctan2(dist[1], dist[0])

        direction_body_frame = direction - heading_angle

        self._command[0] = math.cos(direction_body_frame)
        self._command[1] = math.sin(direction_body_frame)
        # self._command[2] = max(min(self._command[2]+0.01*self.desired_yaw, 
        #                         1), -1)
        return self._command
    
    
    def reset(self, command_mode=0):
        self._command = np.array([0., 0., 0.])
        # if command_mode == 1:
        #     self.desired_yaw = 1
        # elif command_mode == 2:
        #     self.desired_yaw = -1
        # else:
        #     self.desired_yaw = 0
        self._goal_position = np.random.randn(2)*100
        # print(self._goal_position)
        self._command[2] = random.uniform(-1,1)