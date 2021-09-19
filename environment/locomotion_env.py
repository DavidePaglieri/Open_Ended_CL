"""Locomotion environment"""

import numpy as np
import math
import random
from robot import a1
from utils import com_velocity_estimator

dt = 1./240.
ACTION_REPEAT = 4

class LocomotionEnv():
    
    def __init__(self,
                 pybullet_client = None,
                 robot_class = None,
                 pmtg = None,
                 urdf_path = None,
                 scene_class = None,
                 task = None,
                 curriculum = False,
                 simulation_time_step = dt):

        """Initializes the locomotion environment
        
        Args:
            pybullet_client: The instance of a pybullet client
            robot_class: The class of the robot. We prefer to pass the class
                instead of the instance because we might want to hard reset the
                environment by deleting it and creating it anew if it is too
                difficult to reset it.
            pmtg: an instance of the pmtg wrapper
            urdf_path: the path of the URDF of the robot used
            scene_class: class of the surrounding of the robot. If None it 
                simply initializes a normal plane.
            task: An istance of a task class, it contains the reward function 
                and terminal conditions of each episode.
        """
        
        self._pybullet_client = pybullet_client
        self._robot_class = robot_class
        self._pmtg = pmtg
        self._task = task()
        self._sim_time_step = dt
        self._sim_step_counter = 0
        self._env_step_counter = 0
        self._urdf_path = urdf_path
        self._scene_class = scene_class
        self._dt = simulation_time_step
        self._quadruped = None
        self._robot = None
        self._velocity_estimator = None
        self._vel_estimator_class = com_velocity_estimator.COMVelocityEstimator
        self.terrainShape = None
        self.terrain = None
        self._load()
        self._num_action_repeat = ACTION_REPEAT


    def _load_URDF(self):
        """Load the URDF

        Returns:
            _quadruped: the uid to give to the robot's class
        """
        return self._pybullet_client.loadURDF(self._urdf_path, a1.INIT_POSITION,
                                              a1._IDENTITY_ORIENTATION)


    def _load_instance(self):
        """Loads an instance of the robot from the class given

        Returns:
            _robot: the instance of the robot
        """
        # Uncomment the below to load the robot fixed in mid air
        # self._pybullet_client.createConstraint(self._quadruped, -1, -1, -1, 
        #             self._pybullet_client.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
        #                            [0,0,0.5], [0, 0, 0, 1])
        return self._robot_class(self._pybullet_client, 
                                 self._quadruped, self._dt)
        
    def _load_scene(self, heightfield=None):
        """Loads either a simple plane if the scene class is not defined, or the
        scene defined by the class. Be careful that when loading the standard
        plane it may have a different lateral friction coefficient than the scene
        which are normally used. This would cause the robot not to behave as
        expected"""
        if heightfield is None:
            heightfield = np.zeros(128*128)
            terrain_mass = 0
            terrain_visual_shape_id = -1
            terrain_position = [0, 0, 0]
            terrain_orientation = [0, 0, 0, 1]
            self.terrainShape = self._pybullet_client.createCollisionShape(
                                shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
                                meshScale=[1, 1, 1],
                                numHeightfieldRows=128, 
                                numHeightfieldColumns=128,
                                # replaceHeightfieldIndex=self.terrainShape,
                                heightfieldData = heightfield,
                                heightfieldTextureScaling=128)
        
        else:
            size = int(math.sqrt(heightfield.shape[0]))
            self.terrainShape = self._pybullet_client.createCollisionShape(
                                shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
                                meshScale=[0.025, 0.025, 1],
                                # meshScale=[0.05, 0.05, 1],
                                numHeightfieldRows=size, 
                                numHeightfieldColumns=size,
                                # replaceHeightfieldIndex=self.terrainShape,
                                heightfieldData = heightfield,
                                heightfieldTextureScaling=128)
        self.terrain  = self._pybullet_client.createMultiBody(0, self.terrainShape)
        
        # NICER TEXTURE FOR RECORDING
        texUid = self._pybullet_client.loadTexture("/data/grid1.png")
        self._pybullet_client.changeVisualShape(self.terrain, 
                                                -1, 
                                                rgbaColor=[0.9, 0.9, 0.9, 1])
        self._pybullet_client.changeVisualShape(self.terrain, 
                                                -1, 
                                                textureUniqueId=texUid)

        # FRICTION AND COLLISION MARGIN
        self._pybullet_client.resetBasePositionAndOrientation(self.terrain,
                                                            [0,0,0], 
                                                            [0,0,0,1])
        self._pybullet_client.changeDynamics(self.terrainShape, 
                                            -1,
                                            lateralFriction=1)
        self._pybullet_client.changeDynamics(self.terrainShape, 
                                            -1,
                                            collisionMargin=0.01)

        
    def _load(self, heightfield=None):
        """Loads the environment and the robot
        
        Returns the new observation
        """
        
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setTimeStep(self._sim_time_step)
        self._pybullet_client.setGravity(0, 0, -9.81)
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=1,
                                                    allowedCcdPenetration=0.0)
    
        # Disables rendering to load faster
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_RENDERING, 0)

        if (self._quadruped is not None) or (self._robot is not None):
            del self._quadruped
            del self._robot
            del self._velocity_estimator
        self._load_scene(heightfield)
        self._quadruped = self._load_URDF()
        self._robot = self._load_instance()
        self._pmtg.reset()
        self._task.reset(self._robot)
        self._velocity_estimator = self._vel_estimator_class(self._robot)
    
        #Enables rendering after the loading
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_RENDERING, 1)
        
        self._robot.set_foot_friction(0.7)
        # self._robot.get_foot_friction()
        return self.get_state_reward_done(np.zeros(12))
    
    
    def _wait_for_rendering(self):
        """Sleep, otherwise the computation takes less time than real time
        To be finished. This is only used for real time rendering on GUI if the
        simulation is going too fast.
        """
        pass
    
    
    def get_action_space(self):
        return self._pmtg.get_num_actions()
        
        
    def get_state_space(self):
        return self._pmtg.get_num_states()+29+self._task.get_num_commands()+3
        
        
    def hard_reset(self, heightfield = None):
        """Destroys the simulation and resets everything, including the scene,
        the robot and all the objects. All the instances are destroyed and new
        ones are created from the classes. Use the hard reset when you want
        to change the terrain. Otherwise the soft reset if much faster"""
        self._env_step_counter = 0
        self._sim_step_counter = 0
        return self._load(heightfield)
    
    
    def soft_reset(self):
        """Resets the robot pose, position and orientation in the simulation. 
        This is much faster than a hard reset. However, if there are dynamical 
        objects it is better to hard_reset. This includes the heightfield!
        When you also want to change the terrain, do a hard reset, otherwise
        some problems might occur.
        """
        
        self._robot.reset_pose_velocity_control()
        self._pybullet_client.resetBasePositionAndOrientation(
            bodyUniqueId=self._quadruped,
            posObj=self._robot.GetDefaultInitPosition(),
            ornObj=self._robot.GetDeafaultInitOrientation()
        )
        self._sim_step_counter = 0
        self._env_step_counter = 0
        self._pmtg.reset()
        self._task.reset(self._robot)
        self._velocity_estimator.reset()
        
        # Friction randomisation
        # self._robot.set_foot_friction(random.uniform(0.5,0.8))
        
        return self.get_state_reward_done(np.zeros(12))
    
    
    def get_foot_position_in_hip_frame(self):
        """Returns x,y,z position of every foot in hip frame"""
        return self._robot.GetFootPositionsInBaseFrame()
    
    
    def get_foot_desired_position_in_hip_frame(self):
        """Returns desired x,y,z position of every foot in hip frame"""
        return self._pmtg.get_desired_foot_position()
    

    def get_state_reward_done(self, action):
        """Returns the state, the reward and the boolean for the current time
        step"""
        return np.concatenate((
            self._robot.get_observation(),
            self._pmtg.get_phase(),
            self._velocity_estimator.get_x_y_z_velocities(),
            self._task.get_desired_velocities_and_yaw_rate()
            ),axis=0),\
            self._task.get_reward(self._velocity_estimator.get_measurements(),
                                  action),\
            self._task.check_default_terminal_condition()
    
    
    def increase_speed(self, delta):
        """Increases the speed of the TG by a delta factor. Only used when
        testing the trajectory generator alone"""
        self._pmtg.increase_speed(delta)
        
        
    def get_reward_distribution(self):
        return self._task.get_reward_distribution()
    
    
    def spawn_spheres(self):
        """Spawn small spheres to test contacts with the ground"""
        sphereRadius = 0.005
        colSphereId = self._pybullet_client.createCollisionShape(
                                    self._pybullet_client.GEOM_SPHERE, 
                                    radius=sphereRadius)
        colBoxId = self._pybullet_client.createCollisionShape(
                            self._pybullet_client.GEOM_BOX,
                            halfExtents=[sphereRadius, 
                                         sphereRadius, 
                                         sphereRadius])

        mass = 1
        visualShapeId = -1

        for i in range(5):
            for j in range(5):
                for k in range(5):
                    sphereUid = self._pybullet_client.createMultiBody(
                        mass,
                        colSphereId,
                        visualShapeId, [-i * 5 * sphereRadius, 
                                        j * 5 * sphereRadius, 
                                        k * 2 * sphereRadius + 1],
                        useMaximalCoordinates=True)
                    self._pybullet_client.changeDynamics(sphereUid,
                                    -1,
                                    spinningFriction=0.001,
                                    rollingFriction=0.001,
                                    linearDamping=0.0)
                    self._pybullet_client.changeDynamics(sphereUid, 
                                                -1, 
                                                ccdSweptSphereRadius=0.001)

    
    def focus_camera_on_robot(self):
        """Sets the camera to follow the robot. Has to be called at every time
        step. Good for taking videos on GUI connection"""
        # self._pybullet_client.resetDebugVisualizerCamera(0.8, -90, -30, 
        #                             self._robot.get_base_position())
        self._pybullet_client.resetDebugVisualizerCamera(1.0, -30, -20, 
                                    self._robot.get_base_position())
    
    
    def start_recording(self, name):
        """Start recording a video. Only available on GUI connection. It slows
        down the simluation *significantly*. You need to install ffmpeg"""
        self._pybullet_client.startStateLogging(
            loggingType=self._pybullet_client.STATE_LOGGING_VIDEO_MP4,
            fileName="video"+name+".mp4")
    
    
    def stop_recording(self, name):
        """Stop the video recording"""
        self._pybullet_client.stopStateLogging(
            loggingType=self._pybullet_client.STATE_LOGGING_VIDEO_MP4,
            fileName="video"+name+".mp4")
    
    
    def get_key_pressed(self):
        # Found this function on:
        pressed_keys = []
        events = self._pybullet_client.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            pressed_keys.append(key)
        return pressed_keys 
    
    
    def step(self, action=None):
        """Step forward the simulation, given an action of the robot.

        Args:
            action: The x,y,z coordinates of each foot link.
        
        Returns:
            state: a dictionary where the keys are
                the sensor names and the values are the sensor readings.
            reward: the reward obtained by the robot. For the moment it will be
                Null since we are not doing RL but just testing out the env
            done: whether or not the episode terminated.
        """
        
        self._sim_step_counter += self._sim_time_step*self._num_action_repeat
        self._env_step_counter += self._num_action_repeat
        link_positions = self._pmtg.step(time=self._sim_step_counter,
                                         action=action)

        # 60 Hz
        for _ in range(self._num_action_repeat):
            # 240 Hz
            self._robot.inverse_kinematics_action(link_positions)
            
        self._velocity_estimator.update()

        if self._env_step_counter == 120:
            self._pmtg.enable_tg()
            # self._task.enable_command()
        
        # Use to command the command-conditioned policies
        # COMMAND
        # a = self.get_key_pressed()
        # # print(a)
        # if len(a)>0:
        #     if 65295 in a:
        #         self._task.change_desired_yaw_rate(0.02)
        #     if 65296 in a:
        #         self._task.change_desired_yaw_rate(-0.02)
        #     if 65297 in a:
        #         self._task.change_desired_forward_velocity(0.02)
        #     if 65298 in a:
        #         self._task.change_desired_forward_velocity(-0.02)


        # APPLY EXTERNAL FORCES
        
        # STRONG PUSH
        # if self._env_step_counter == 1200:
        #     force = np.array([0., 4000, 0.])
        #     # print(force)
        #     # print("HELLO")
        #     self._pybullet_client.applyExternalForce(self._quadruped,
        #             linkIndex=-1,
        #             forceObj=force,
        #             posObj=self._robot.get_base_position(),
        #             flags=self._pybullet_client.WORLD_FRAME)
        
        # RANDOM PUSHES (training)
        # force = np.random.randn(3)*30
        # # force[2] = 0
        # self._pybullet_client.applyExternalForce(self._quadruped,
        #         linkIndex=-1,
        #         forceObj=force,
        #         posObj=self._robot.get_base_position(),
        #         flags=self._pybullet_client.WORLD_FRAME)
        
        return self.get_state_reward_done(action[:12])