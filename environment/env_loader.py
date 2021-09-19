# Copyright 2021 Davide Paglieri

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Loads a PyBullet environment for the A1 robot locomotion"""

import pybullet as p
import pybullet_utils.bullet_client as bc

from environment import locomotion_env
from robot import a1
from pmtg import pmtg_wrapper
from tasks import control_velocity


def load(connection_mode = "DIRECT",
         task_name = "running_velocity"):
    """Create an instance of the locomotion environment in PyBullet

    Args:
        connection_mode: The type of connection with the PyBullet client.
            Can be DIRECT or GUI. Defaults to DIRECT
        task_name: The task assigned to the robot. It can be "running_velocity"
            where only x velocity impacts the reward or "control_velocity" where
            x_vel, y_vel and yaw_rate make up the reward.

    Returns:
        env: the locomotion environment
    """
    if connection_mode == "DIRECT":
        client = bc.BulletClient(p.DIRECT)
    elif connection_mode == "GUI":
        client = bc.BulletClient(p.GUI)
    else:
        raise("Connection mode not supported")
    
    if task_name == "running_velocity":
        task = running_velocity.Task
    elif task_name == "control_velocity":
        task = control_velocity.Task
    
    client.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    client.setPhysicsEngineParameter(allowedCcdPenetration=0.0)
    
    return locomotion_env.LocomotionEnv(pybullet_client = client, 
                                    robot_class = a1.A1,
                                    urdf_path = a1.URDF_PATH,
                                    pmtg = pmtg_wrapper.PMTG(),
                                    task = task,
                                    curriculum = False)
                                    # scene_class = simple_hallway.Scene)