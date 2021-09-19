# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pybullet simulation of an A1 robot.
The implementation is partially adapted from functions already available for the
Laikago and Minitaur robots on the PyBullet repository
Adapted by Davide Paglieri, MSc student at Imperial College London (2020-2021)"""

import numpy as np
import math
from robot import a1_motor
import re

URDF_PATH = "robot/a1.urdf"

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [-0.5, 0,0.42]
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi


COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                        ]) + COM_OFFSET

HIP_OFFSETS_FOOT_ALLIGNED = HIP_OFFSETS = np.array(
    [[0.183, -0.135, 0.], [0.183, 0.13, 0.],
    [-0.183, -0.135, 0.], [-0.183, 0.13, 0.]
    ]) + COM_OFFSET

FOOT_POSITION_IN_HIP_FRAME_FIXED = np.array([
    [0, -0.08505, -0.25], [0, 0.08505, -0.25],
    [0, -0.08505, -0.25], [0, 0.08505, -0.25]]
)

FOOT_POSITION_IN_HIP_FRAME = np.array([
    [0, -0.08505, 0], [0, 0.08505, 0],
    [0, -0.08505, 0], [0, 0.08505, 0]]
)

# Original tuning, completely broken (they used hybrid control)
# ABDUCTION_P_GAIN = 100.0
# ABDUCTION_D_GAIN = 1.
# HIP_P_GAIN = 100.0
# HIP_D_GAIN = 2.0
# KNEE_P_GAIN = 100.0
# KNEE_D_GAIN = 2.0

# Tuned while fixed in mid air (PyBullet solver)
# ABDUCTION_P_GAIN = 100
# ABDUCTION_D_GAIN = 2
# HIP_P_GAIN = 200
# HIP_D_GAIN = 2
# KNEE_P_GAIN = 100
# KNEE_D_GAIN = 1

# Tuned while standing still (PyBullet solver)
# ABDUCTION_P_GAIN = 100
# ABDUCTION_D_GAIN = 0.2
# HIP_P_GAIN = 250
# HIP_D_GAIN = 2
# KNEE_P_GAIN = 110
# KNEE_D_GAIN = 1

# Tuned while standing still (Analytical IK)
ABDUCTION_P_GAIN = 100
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 200
HIP_D_GAIN = 1
KNEE_P_GAIN = 200
KNEE_D_GAIN = 1

P_GAINS = [100, 100, 100,
           100, 100, 100,
           100, 200, 200,
           100, 200, 200]

D_GAINS = [1,1,1,
           1,1,1,
           1,1,1,
           1,1,1]

ACTION_REPEAT = 5

# FR, FL, RR, RL
_DEFAULT_HIP_POSITIONS = (
    (0.17, -0.135, 0),
    (0.17, 0.13, 0),
    (-0.195, -0.135, 0),
    (-0.195, 0.13, 0),
)

# FR, FL, RR, RL
_DEFAULT_FOOT_POSITIONS = (
    (0.17, -0.135, -0.25),
    (0.17, 0.13, -0.25),
    (-0.195, -0.135, -0.25),
    (-0.195, 0.13, -0.25),
)

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2

#Actual max motor torque of the A1 robot
MAX_MOTOR_TORQUE = np.array([33.5]*NUM_MOTORS)


_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

_IDENTITY_ORIENTATION=[0,0,0,1]
HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

#Use a PD controller
MOTOR_CONTROL_POSITION = 1 
# Apply motor torques directly.
MOTOR_CONTROL_TORQUE = 2
# Apply a tuple (q, qdot, kp, kd, tau) for each motor. Here q, qdot are motor
# position and velocities. kp and kd are PD gains. tau is the additional
# motor torque. This is the most flexible control mode.
MOTOR_CONTROL_HYBRID = 3

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_LEGS)

# Optional parameters for constraining the inverse kinematics solution
#lower limits for null space
ll = [-0.5, 0.4, -2.3]*NUM_LEGS
#upper limits for null space
ul = [0.5, 1.4, -1.3]*NUM_LEGS
#joint ranges for null space
jr = [4, 4, 4]*NUM_LEGS
#restposes for null space
rp = [0, 0.9, -1.8]*NUM_LEGS
#joint damping coefficents
jd = [0.1, 0.1, 0.1]*NUM_LEGS


def foot_position_in_hip_frame_to_joint_angle(foot_position, 
                                                l_hip_sign=1):
    """Starting from the cartesian coordinates of the foot position it
    computes the angles of the joints.

    Args:
        foot_position (np.array): [Numpy array with the x, y, z coordiantes
                                of the foot]
        l_hip_sign (int, optional): [Defines whether it's a left (1)
                                    or right (-1) leg. Defaults to 1.

    Returns:
        [theta_ab, theta_hip, theta_knee]: [The angles in radians of the 
                                            the joint angles]
    """
    
    clamp_input = lambda x: -1 if x < -1 else 1 if x > 1 else x

    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -math.acos(clamp_input((x**2 + y**2 + z**2 - l_hip**2 - 
                                    l_low**2 - l_up**2)/(2 * l_low * l_up)))
    l = math.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * math.cos(theta_knee))\
        +1e-4
    theta_hip = math.asin(clamp_input(-x / l)) - theta_knee / 2
    c1 = l_hip * y - l * math.cos(theta_hip + theta_knee / 2) * z
    s1 = l * math.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = math.atan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])


def foot_position_in_hip_frame(angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    leg_distance = math.sqrt(l_up**2 + l_low**2 +
                            2 * l_up * l_low * math.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * math.sin(eff_swing)
    off_z_hip = -leg_distance * math.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = math.cos(theta_ab) * off_y_hip - math.sin(theta_ab) * off_z_hip
    off_z = math.sin(theta_ab) * off_y_hip + math.cos(theta_ab) * off_z_hip
    return np.array([off_x, off_y, off_z])


class A1(object):
    """A simulation for the A1 robot."""
    
    def __init__(self, pybullet_client, robot_uid, simulation_time_step):
        self.pybullet_client = pybullet_client
        self.quadruped = robot_uid
        self.time_step = simulation_time_step
        self.num_legs = NUM_LEGS
        self.num_motors = NUM_MOTORS
        self._BuildJointNameToIdDict()
        self._BuildUrdfIds()
        self._BuildMotorIdList()
        #self.reset_pose_position_control()
        self.reset_pose_velocity_control()
        self._enable_clip_motor_commands = True
        self._motor_enabled_list = [True] * self.num_motors
        self._step_counter = 0
        self._state_action_counter = 0
        self._motor_offset= np.array([0]*12)
        self._motor_direction= np.ones(12)
        self.ReceiveObservation()
        self._kp = self.GetMotorPositionGains()
        self._kd = self.GetMotorVelocityGains()
        self._motor_model = a1_motor.A1MotorModel(kp=self._kp, kd=self._kd, 
                                    motor_control_mode=MOTOR_CONTROL_POSITION,
                                    torque_limits=MAX_MOTOR_TORQUE)
        self._torques = 0
        self._set_ccd_foot_links()
        # self._SettleDownForReset(reset_time=1.0)
        # self.set_foot_friction()
        
        
    def _BuildJointNameToIdDict(self):
        """Builds a dictionary with keys the name of the joint and
        value the ID of the joint.
        """
        num_joints = self.pybullet_client.getNumJoints(self.quadruped)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
    
    
    def _BuildUrdfIds(self):
        """Build the link Ids from its name in the URDF file.

        Raises:
            ValueError: Unknown category of the joint name.
        """
        num_joints = self.pybullet_client.getNumJoints(self.quadruped)
        self._hip_link_ids = [-1]
        self._leg_link_ids = []
        self._motor_link_ids = []
        self._lower_link_ids = []
        self._foot_link_ids = []
        self._imu_link_ids = []

        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if HIP_NAME_PATTERN.match(joint_name):
                self._hip_link_ids.append(joint_id)
            elif UPPER_NAME_PATTERN.match(joint_name):
                self._motor_link_ids.append(joint_id)
            # We either treat the lower leg or the toe as the foot link, 
            # depending on the urdf version used.
            elif LOWER_NAME_PATTERN.match(joint_name):
                self._lower_link_ids.append(joint_id)
            elif TOE_NAME_PATTERN.match(joint_name):
                #assert self._urdf_filename == URDF_WITH_TOES
                self._foot_link_ids.append(joint_id)
            elif IMU_NAME_PATTERN.match(joint_name):
                self._imu_link_ids.append(joint_id)
            else:
                raise ValueError("Unknown category of joint %s" % joint_name)

        self._leg_link_ids.extend(self._lower_link_ids)
        self._leg_link_ids.extend(self._foot_link_ids)

        #assert len(self._foot_link_ids) == NUM_LEGS
        self._hip_link_ids.sort()
        self._motor_link_ids.sort()
        self._lower_link_ids.sort()
        self._foot_link_ids.sort()
        self._leg_link_ids.sort()
    
    
    def set_foot_friction(self, foot_friction=0.9):
        """Set the lateral friction of the feet.

        Args:
        foot_friction: The lateral friction coefficient of the foot. This value 
            is shared by all four feet.
        """
        for link_id in self._foot_link_ids:
            self.pybullet_client.changeDynamics(
                self.quadruped, link_id, lateralFriction=foot_friction)


    def reset_pose_velocity_control(self):
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self.pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=(joint_id),
                controlMode=self.pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)
        for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
            if "hip_joint" in name:
                angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
            elif "upper_joint" in name:
                angle = INIT_MOTOR_ANGLES[i] + UPPER_LEG_JOINT_OFFSET
            elif "lower_joint" in name:
                angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
            else:
                raise ValueError("Name %s not recognized as a motor joint." 
                                 %name)
            self.pybullet_client.resetJointState(
                        self.quadruped, 
                        self._joint_name_to_id[name], 
                        angle, 
                        targetVelocity=0)


    def reset_pose_position_control(self):
        
        for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
            joint_id = self._joint_name_to_id[name]
            self.pybullet_client.setJointMotorControl2(self.quadruped, 
                                    joint_id, 
                                    self.pybullet_client.POSITION_CONTROL,
                                    INIT_MOTOR_ANGLES[i])
            self.pybullet_client.resetJointState(self.quadruped, 
                                joint_id,
                                INIT_MOTOR_ANGLES[i])


    def _SettleDownForReset(self, reset_time):
        self.ReceiveObservation()
        if reset_time <= 0:
            return
        for _ in range(240):
            self._StepInternal(
                INIT_MOTOR_ANGLES,
                motor_control_mode=MOTOR_CONTROL_POSITION)
    
    
    def _GetMotorNames(self):
        return MOTOR_NAMES
    
    
    def _BuildMotorIdList(self):
        self._motor_id_list = [
            self._joint_name_to_id[motor_name]
            for motor_name in self._GetMotorNames()
        ]
        
        
    def GetMotorPositionGains(self):
        # return np.array(P_GAINS)
        return np.array([ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, 
                         ABDUCTION_P_GAIN,
            HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN])
    
    
    def GetMotorVelocityGains(self):
        # return np.array(D_GAINS)
        return np.array([ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, 
                         ABDUCTION_D_GAIN,
            HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN])
        
        
    def GetDeafaultInitOrientation(self):
        """Get the initial orientation"""
        return [0, 0, 0, 1]
    
    
    def GetDefaultInitPosition(self):
        """Get the initial position"""
        return INIT_POSITION
    
    
    def get_base_velocity(self):
        """Get the base velocity (x,y,z) of the robot"""
        velocity, _ = self.pybullet_client.getBaseVelocity(self.quadruped)
        return velocity


    def inverse_kinematics_action(self, action=None):
        """Uses inverse kinematics to calculate the desired joint angles
        starting from the x,y,z coordinates of the foot links. It then uses a PD
        controller to apply the torques to the motors. 
        
        Args:
            action np.array (4,3): x,y,z coordinates of the foot links
            
        Returns:
            a list of observations
        """
        
        if action is None:
            # Analytic IK
            link_positions = FOOT_POSITION_IN_HIP_FRAME_FIXED
            # PyBullet IK solver
            link_positions = HIP_OFFSETS_FOOT_ALLIGNED
        else:
            # Analytic IK
            link_positions = action + FOOT_POSITION_IN_HIP_FRAME
            # PyBullet IK solver
            # link_positions = action + HIP_OFFSETS_FOOT_ALLIGNED
            
        # Joint angles using PyBullet inverse kinematics solver
        # joint_angles = (np.asarray((
        #     self.motor_angles_from_foot_position(0, link_positions[0])[1],
        #     self.motor_angles_from_foot_position(1, link_positions[1])[1],
        #     self.motor_angles_from_foot_position(2, link_positions[2])[1],
        #     self.motor_angles_from_foot_position(3, link_positions[3])[1]
        #     )).reshape(-1)).tolist()
        
        # Joint angles using analytic inverse kinematics
        joint_angles = []
        for leg_id in range(NUM_LEGS):
            foot_position = link_positions[leg_id]
            joint_angles[leg_id*3:((leg_id*3)+3)] = \
                foot_position_in_hip_frame_to_joint_angle(
                    foot_position, l_hip_sign=(-1)**(leg_id+1))
    
        self._StepInternal(joint_angles, MOTOR_CONTROL_POSITION)
    
    
    def motor_angles_from_foot_position(self, leg_id,
                                              foot_local_position):
        """Use IK to compute the motor angles, given the foot link's local 
            position. Local position means that it's offset to the centre of
            mass of the robot.

        Args:
            leg_id: The leg index.
            foot_local_position: The foot link's position in the base frame.

        Returns:
            A tuple. The position indices and the angles for all joints along 
            the leg. The position indices is consistent with the joint orders as
            returned by GetMotorAngles API.
        """
        return self._EndEffectorIK(
            leg_id, foot_local_position, position_in_world_frame=False)
    
    
    def _EndEffectorIK(self, leg_id, position, position_in_world_frame):
        """Calculate the joint positions from the end effector position."""
        assert len(self._foot_link_ids) == self.num_legs
        toe_id = self._foot_link_ids[leg_id]
        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = [
            i for i in range(leg_id * motors_per_leg, leg_id * motors_per_leg +
                         motors_per_leg)
        ]
    
        joint_angles = self.joint_angles_from_link_position(
            robot=self,
            link_position=position,
            link_id=toe_id,
            joint_ids=joint_position_idxs,
            position_in_world_frame=position_in_world_frame)
        # Joint offset is necessary for A1.
        # joint_angles = np.multiply(
        #     np.asarray(joint_angles) -
        #     np.asarray(self._motor_offset)[joint_position_idxs],
        #     self._motor_direction[joint_position_idxs])
        
        # Return the joint index (the same as when calling GetMotorAngles) 
        # as well as the angles.
        #print(f"indexes{joint_position_idxs}, with angles {joint_angles}")
        return joint_position_idxs, joint_angles#.tolist()
    
    
    def joint_angles_from_link_position(
        self,
            robot,
            link_position,
            link_id,
            joint_ids,
            position_in_world_frame,
            base_translation = (0, 0, 0),
            base_rotation = (0, 0, 0, 1)):
        """Uses Inverse Kinematics to calculate joint angles.

        Args:
            robot: A robot instance.
            link_position: The (x, y, z) of the link in the body or the world 
                frame, depending on whether the argument position_in_world_frame 
                is true.
            link_id: The link id as returned from loadURDF.
            joint_ids: The positional index of the joints. This can be different 
                from the joint unique ids.
            position_in_world_frame: Whether the input link_position is 
                specified in the world frame or the robot's base frame.
            base_translation: Additional base translation.
            base_rotation: Additional base rotation.

        Returns:
            A list of joint angles.
        """
        if not position_in_world_frame:
            # Projects to local frame.
            base_position, base_orientation = \
                self.pybullet_client.getBasePositionAndOrientation(
                    self.quadruped)
            base_position, base_orientation = \
                robot.pybullet_client.multiplyTransforms(
                    base_position, base_orientation, base_translation, 
                    base_rotation)

            # Projects to world space.
            world_link_pos, _ = robot.pybullet_client.multiplyTransforms(
                base_position, base_orientation, link_position, 
                    _IDENTITY_ORIENTATION)
        else:
            world_link_pos = link_position

        ik_solver = 0
        all_joint_angles = robot.pybullet_client.calculateInverseKinematics(
            robot.quadruped,link_id, world_link_pos,solver=ik_solver)

        #print(all_joint_angles)
        # Extract the relevant joint angles.
        joint_angles = [all_joint_angles[i] for i in joint_ids]
        #print(joint_angles)
        return joint_angles
    
    
    def get_time_since_reset(self):
        return self._step_counter * self.time_step

    
    def GetHipPositionsInBaseFrame(self):
        return _DEFAULT_HIP_POSITIONS
    
    
    def GetTrueBaseOrientation(self):
        """Get the orientation of the robot

        Returns:
            The orientation is a quaternion in the format [x,y,z,w]
            The orientation can be transformed into Euler angles using the
            function getEulerFromQuaternion (pyBullet)
        """
        _, orientation = self.pybullet_client.getBasePositionAndOrientation(
            self.quadruped)
        return orientation
    
    
    def get_base_position(self):
        """Get the x, y, z coordinates of the robot"""
        pos, _ = self.pybullet_client.getBasePositionAndOrientation(
            self.quadruped)
        return pos
    
    
    def TransformAngularVelocityToLocalFrame(self, angular_velocity, 
                                             orientation):
        """Transform the angular velocity from world frame to robot's frame.

        Args:
            angular_velocity: Angular velocity of the robot in world frame.
            orientation: Orientation of the robot represented as a quaternion.

        Returns:
            angular velocity of based on the given orientation.
        """
        # Treat angular velocity as a position vector, then transform based on 
        # the orientation given by dividing (or multiplying with inverse).
        # Get inverse quaternion assuming the vector is at 0,0,0 origin.
        _, orientation_inversed = self.pybullet_client.invertTransform(
                                                    [0, 0, 0],orientation)
        # Transform the angular_velocity at neutral orientation using a neutral
        # translation and reverse of the given orientation.
        relative_velocity, _ = self.pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self.pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return relative_velocity
    
    
    def get_base_roll_pitch_yaw_rate(self):
        """Get the rate of orientation change of the minitaur's base in 
            euler angle.

        Returns:
            rate of (roll, pitch, yaw) change of the minitaur's base.
        """
        return self.TransformAngularVelocityToLocalFrame(
                    self.pybullet_client.getBaseVelocity(self.quadruped)[1],
                    self.GetTrueBaseOrientation())


    def get_base_roll_pitch(self):
        """Get A1's base orientation in euler angle in the world frame.

            Returns:
                A tuple (roll, pitch, yaw) of the base in world frame.
        """
        return np.array(self.pybullet_client.getEulerFromQuaternion(
            self.GetTrueBaseOrientation())[:2])
        
        
    def get_base_roll_pitch_yaw(self):
        """Get A1's base orientation in euler angle in the world frame.

            Returns:
                A tuple (roll, pitch) of the base in world frame.
        """
        return self.pybullet_client.getEulerFromQuaternion(
            self.GetTrueBaseOrientation())
    
    
    def get_IMU(self):
        """Get the robot's roll, roll rate, pitch, pitch rate and yaw rate"""
        
        return np.array([self.get_base_roll_pitch_yaw(),
                        self.get_base_roll_pitch_yaw_rate()]).reshape(-1)


    def get_foot_contacts(self):
        """Returns a vector with the booleans of the foot state. True indicates
        contact"""
        all_contacts = self.pybullet_client.getContactPoints(bodyA=
                                                             self.quadruped)

        contacts = [False, False, False, False]
        for contact in all_contacts:
            # Ignore self contacts
            if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(
                    contact[_LINK_A_FIELD_NUMBER])
                contacts[toe_link_index] = True
            except ValueError:
                continue
        return contacts
    
    
    def _set_ccd_foot_links(self):
        """The dimension of the sphere of the foot links is small. This command
        is used so that the foot link won't go through the heightfield terrain
        """
        # print("CHANGING CCD")
        for link_id in self._foot_link_ids:
            self.pybullet_client.changeDynamics(self.quadruped, 
                                                link_id, 
                                                ccdSweptSphereRadius=0.01)
            
    
    
    def get_motor_angles(self):
        """Gets the twelve motor angles at the current moment, mapped to 
            [-pi, pi].

        Returns:
            Motor angles, mapped to [-pi, pi].
        """
        return np.asarray([state[0] for state in self._joint_states])
    
    
    def get_motor_velocities(self):
        """Get the velocity of all twelve motors.

        Returns:
            Velocities of all twelve motors.
        """
        return np.asarray([state[1] for state in self._joint_states])
    
    
    def GetPDObservation(self):
        """Get the Proportional Derivative observations of the motors

        Returns:
            A numpy array of the position (angles) and velocities observed
        """
        self.ReceiveObservation()
        observation = []
        observation.extend(self.get_motor_angles())
        observation.extend(self.get_motor_velocities())
        q = observation[0:self.num_motors]
        qdot = observation[self.num_motors:2 * self.num_motors]
        return (np.array(q), np.array(qdot))
    
    
    def get_observation(self):
        """Return the joint angles, joint velocities, pitch, roll, yaw, 
        pitch rate, roll rate, yaw rate"""
        self.ReceiveObservation()
        return np.concatenate((
            self.get_motor_angles(),
            self.get_motor_velocities(),
            self.get_base_roll_pitch(),
            self.get_base_roll_pitch_yaw_rate()
            ))
    
    
    def GetTrueObservation(self):
        """Get all the observations of the robot state. 

        Returns:
            A list of the motor angles, velocities, torques, robot orientation,
             and roll/pitch/yaw rate.
        """
        self.ReceiveObservation()
        observation = []
        observation.extend(self.get_motor_angles())
        observation.extend(self.get_motor_velocities())
        #observation.extend(self.GetTrueMotorTorques())
        observation.extend(self.GetTrueBaseOrientation())
        #observation.extend(self.GetTrueBaseRollPitchYawRate())
        return observation

    
    def ApplyAction(self, motor_commands, motor_control_mode):
        """Apply the motor commands using the motor model.

        Args:
            motor_commands: np.array. Can be motor angles, torques, hybrid 
            commands motor_control_mode: A MotorControlMode enum.
        """
        
        if self._enable_clip_motor_commands:
            motor_commands = self._ClipMotorCommands(motor_commands)
        
        motor_commands = np.asarray(motor_commands)
        q, qdot = self.GetPDObservation()
        qdot_true = self.get_motor_velocities()
        actual_torque, observed_torque = self._motor_model.convert_to_torque(
            motor_commands, q, qdot, qdot_true, motor_control_mode)
        
        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = observed_torque

        # Transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque,
                                                self._motor_direction)
        motor_ids = []
        motor_torques = []

        for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list,
                                                     self._applied_motor_torque,
                                                     self._motor_enabled_list):
            if motor_enabled:
                motor_ids.append(motor_id)
                motor_torques.append(motor_torque)
            else:
                motor_ids.append(motor_id)
                motor_torques.append(0)
                
        self._SetMotorTorqueByIds(motor_ids, motor_torques)
        # return np.sum(np.absolute(motor_torque))
        return motor_torques


    def _ClipMotorCommands(self, motor_commands):
        """Clips motor commands.

        Args:
            motor_commands: np.array. Can be motor angles, torques, hybrid 
            commands

        Returns:
            Clipped motor commands.
        """

        # clamp the motor command by the joint limit, in case weired things 
        # happens
        max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
        current_motor_angles = self.get_motor_angles()
        motor_commands = np.clip(motor_commands,
                                current_motor_angles - max_angle_change,
                                current_motor_angles + max_angle_change)
        return motor_commands
        
        
    def _SetMotorTorqueByIds(self, motor_ids, torques):
        # print(torques)
        self.pybullet_client.setJointMotorControlArray(
            bodyIndex=self.quadruped,
            jointIndices=motor_ids,
            controlMode=self.pybullet_client.TORQUE_CONTROL,
            forces=torques
            )
    
    
    def get_foot_friction(self):
        """Set the lateral friction of the feet.

        Args:
        foot_friction: The lateral friction coefficient of the foot. This value 
            is shared by all four feet.
        """
        for link_id in self._foot_link_ids:
            print(self.pybullet_client.getDynamicsInfo(
                self.quadruped, link_id)[1])
        
        
    def ReceiveObservation(self):    
        self._joint_states = self.pybullet_client.getJointStates(self.quadruped,
                                                            self._motor_id_list)
        

    def _StepInternal(self, action, motor_control_mode):
        self._torques += np.sum(np.absolute(self.ApplyAction(action, 
                                                      motor_control_mode)))
        self.pybullet_client.stepSimulation()
        #self.ReceiveObservation()
        self._state_action_counter += 1
    

    def link_position_in_base_frame(self, link_id ):
        """Computes the link's local position in the robot frame.

        Args:
            robot: A robot instance.
            link_id: The link to calculate its relative position.

        Returns:
            The relative position of the link.
        """
        base_position, base_orientation = \
            self.pybullet_client.getBasePositionAndOrientation(
            self.quadruped)
        inverse_translation, inverse_rotation = \
            self.pybullet_client.invertTransform(
                base_position, base_orientation)

        link_state = self.pybullet_client.getLinkState(self.quadruped, link_id)
        link_position = link_state[0]
        link_local_position, _ = self.pybullet_client.multiplyTransforms(
            inverse_translation, inverse_rotation, link_position, (0, 0, 0, 1))

        return np.array(link_local_position)
    

    def GetFootLinkIDs(self):
        """Get list of IDs for all foot links."""
        return self._foot_link_ids
    
    
    def GetFootPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        assert len(self._foot_link_ids) == self.num_legs
        foot_positions = []
        for foot_id in self.GetFootLinkIDs():
            foot_positions.append(
                self.link_position_in_base_frame(link_id=foot_id)
            )
        return np.array(foot_positions)

    
    def print_ids_information(self):
        """Prints the joint_name_to_id dictionary in a nice format"""
        
        for key, value in self._joint_name_to_id.items():
            print("\t"+str(key)+"\t"+str(value))
            
        print(f"Motor ID list {self._motor_id_list}")
    
    
    def get_applied_motor_torque(self):
        """Returns the motor torques applied to the 12 motors"""
        return self._applied_motor_torque
    
    
    def get_torques(self):
        """Returns the sum of the absolute values of the torques applied to the
        12 motors up until the last time this command was called. It then resets
        the torque term, and start summing again."""
        torque = self._torques
        self._torques = 0
        return torque
        