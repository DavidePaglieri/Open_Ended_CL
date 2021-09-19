# Code adapted from:
# https://github.com/leggedrobotics
# learning_quadrupedal_locomotion_over_challenging_terrain_supplementary

import numpy as np
import math
import random


class Sample_Command():
    
    def __init__(self, command_mode, terrain_type):
        self._command_mode = command_mode
        self._terrain_prop = np.array([10,10])
        self._terrain_type = terrain_type
        self._command = np.array([1., 0., 0.])
        self._goal_position = np.zeros(2)
        self._stop = False
    
    
    def update_command(self, robot_position, heading_angle):
        if self._command_mode == 'FIXED_DIR':
            return
        
        if np.linalg.norm(self._command[:2]) != 0.0:
            dist = self._goal_position - robot_position
            if (np.linalg.norm(dist))<0.5:
                self.sample_new_command(robot_position, heading_angle)
                return
            direction = np.arctan2(dist[1],dist[0])
            # heading_angle = np.arctan2(robot_heading[1], robot_heading[0])
            direction_body_frame = direction - heading_angle
            
            self._command[0] = math.cos(direction_body_frame)
            self._command[1] = math.sin(direction_body_frame)

        # print(f"Goal {self._goal_position}\nPosition {robot_position}\nRobot Heading {heading_angle}\nGoal Direction {direction}\nCommand {self._command}\n\n")
    
    
    def get_command(self):
        return self._command
    
    
    def get_goal_position(self):
        return self._goal_position
    
    
    def sample_goal(self, robot_position):
        self._goal_position[0] = robot_position[0]
        self._goal_position[1] = robot_position[1]
        
        if self._terrain_type == 'STAIRS':
            self._goal_position[0] += self._terrain_prop[0]*0.1*random.uniform(-1,1)
            if self._goal_position[1] < 0.3:
                self._goal_position[1] = self._terrain_prop[1]*(0.3+0.2*random.uniform(0,1))
            else:
                self._goal_position[1] = -self._terrain_prop[1]*(0.3+0.2*random.uniform(0,1))
        else:
            self._goal_position[0] += self._terrain_prop[0]*0.4*random.uniform(-1,1)
            self._goal_position[1] += self._terrain_prop[1]*0.4*random.uniform(-1,1)
            if self._command_mode == 'STRAIGHT':
                self._goal_position[0] = robot_position[0] + self._terrain_prop[0]*0.1*random.uniform(-1,1)
                self._goal_position[1] = robot_position[1] + self._terrain_prop[1]*0.4
                
        clamp = lambda goal, prop: max(min(0.5*prop-1.0,goal), -0.5*prop+1.0)
        self._goal_position[0] = clamp(self._goal_position[0],self._terrain_prop[0])
        self._goal_position[1] = clamp(self._goal_position[1],self._terrain_prop[1])
        
        
    def sample_new_command(self, robot_position, heading_angle):
        
        if self._command_mode == 'FIXED_DIR':
            self._goal_position[0] = 0.0
            self._goal_position[1] = 0.0
            self._command = np.array([1., 0., 0.])
            return
        elif self._command_mode == 'ZERO':
            self._command = np.zeros(3)
            self._goal_position[0] = 10.0
            self._goal_position[1] = 10.0
            return
        self.sample_goal(robot_position)
        dist = self._goal_position - robot_position
        direction = np.arctan2(dist[1], dist[0])
        
        # heading_angle = np.arctan2(robot_heading[1], robot_heading[0])
        direction_body_frame = direction - heading_angle

        self._command[0] = math.cos(direction_body_frame)
        self._command[1] = math.sin(direction_body_frame)
        self._command[2] = 0.0
        
        if self._command_mode != 'STRAIGHT':
            self._command[2] = 1.0 - 2.0*random.randint(0,1)
            self._command[2] *= random.uniform(0,1)
            if self._command_mode != 'NOZERO' and random.uniform(0,1)>0.8:
                self._command[0] = 0.0
                self._command[1] = 0.0

        if self._terrain_type == 'STAIRS':
            if random.uniform(0,1) < 0.5:
                self._command[2] = 0


# sample = Sample_Command('NOZERO','MOUNTAINS')
# robot_position = np.array([0, 0.])
# robot_heading = np.array([1, 0])
# sample.update_command(robot_position, robot_heading)


# print(sample.get_command())
# while True:
#     key = input()
#     if key == 'p':
#         goal = sample.get_goal_position()
#         command = sample.get_command()
#         movement = 0.2*(command[:2])
#         robot_position += movement
#         print(f"The goal is {goal}\nThe movement is {command}\nThe new position is {robot_position}")
#         sample.update_command(robot_position, robot_heading)