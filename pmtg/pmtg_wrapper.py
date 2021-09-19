# PMTG wrapper
# Special thanks to Atil Iscen for helping with the implementation
# Part of the implementation is adapted from PyBullet.

import  numpy as np
from pmtg import cyclic_integrator
from pmtg import trajectory_generator
from robot import a1

# These gaits are for FL, RL, FR, RR.
_GAIT_PHASE_MAP = {
    "walk": [0, 0.25, 0.5, 0.75],
    "trot": [0, 0.5, 0.5, 0],
    "bound": [0, 0.5, 0, 0.5],
    "pace": [0, 0, 0.5, 0.5],
    "pronk": [0, 0, 0, 0]
}

residual_ranges = np.array([0.12, 0.10, 0.10,
                            0.12, 0.10, 0.10,
                            0.12, 0.10, 0.10,
                            0.12, 0.10, 0.10])

class PMTG():
    
    """PMTG wrapper's only role is to decompose the actions, 
    call the integrator to progress the phase and call trajectory 
    generator to obtain motor positions based on the new phase. 
    It also serves as the main hub to do decouplings and assigning 
    legs to different integrators if you prefer to have the legs 
    decoupled.
    """
    
    def __init__(self,
                integrator_coupling_mode="all coupled",
                walk_height_coupling_mode="all coupled",
                residual_range=residual_ranges,
                init_leg_phase_offsets=None,
                init_gait=None):
        """Initialzes the wrapped env.

        Args:
        integrator_coupling_mode: How the legs should be coupled for 
            integrators.
        walk_height_coupling_mode: The same coupling mode used for walking 
            walking heights for the legs.
        residual_range: The upper limit for the residual actions that adds to 
            the leg motion. By default it is 0.1 for x,y residuals, and 0.05 for
            z residuals.
        init_leg_phase_offsets: The initial phases of the legs. A list of 4
            variables within [0,1). The order is front-left, rear-left, 
            front-right and rear-right.
        init_gait: The initial gait that sets the starting phase difference
            between the legs. Overrides the arg init_phase_offsets. Has to be
            "walk", "trot", "bound" or "pronk". Used in vizier search.

        Raises:
            ValueError if the controller does not implement get_action and
            get_observation.
        """

        self._num_actions = 12
        self._residual_range = residual_range
        
        if init_gait:
            if init_gait in _GAIT_PHASE_MAP:
                init_leg_phase_offsets = _GAIT_PHASE_MAP[init_gait]
            else:
                raise ValueError("init_gait is not one of the defined gaits.")
        else:
            init_leg_phase_offsets = _GAIT_PHASE_MAP["trot"]
        
        self._trajectory_generator = trajectory_generator.TG(
            init_leg_phase_offsets=init_leg_phase_offsets)
        action_dim = self._extend_action_space()
        self._extend_obs_space()
        
        self.reset()
        
        
    def _extend_obs_space(self):
        """Extend observation space to include pmtg phase variables."""
        pass
    
    
    def _extend_action_space(self):
        """Extend the action space to include pmtg parameters."""
        return 17
    
    
    def get_phase(self):
        """Returns the phase of the trajectory generator"""
        return self._trajectory_generator.get_state()

    
    def _get_observation_bounds(self):
        """Get the bounds of the observation added from the trajectory generator

        Returns:
            lower_bounds: Lower bounds for observations
            upper_bounds: Upper bounds for observations
        """
        lower_bounds = self._trajectory_generator.get_state_lower_bounds()
        upper_bounds = self._trajectory_generator.get_state_upper_bounds()
        return lower_bounds, upper_bounds


    def step(self, time, action=None,):
        """Make a step of the pmtg wrapper

        Args:
            action: a numpy array composed of the policy residuals and the
                time_multiplier, intensity, walking_height, swing_stance_ratio
                used by the Trajectory Generator
            time: time since reset in seconds in the environment

        Returns:
            link_positions: a numpy array composed of the positions of the link
                feet.
        """
        
        delta_real_time = time - self._last_real_time
        self._last_real_time = time
        
        if action is not None:
            residuals = action[0:self._num_actions]*self._residual_range
            # Calculate trajectory generator's output based on the rest of 
            # the actions.
            action_tg = self._trajectory_generator.get_actions(
                delta_real_time, action[self._num_actions:])          
            link_positions = action_tg + residuals         
        
        else:
            link_positions = self._trajectory_generator.get_actions(
                delta_real_time, None)
        
        return np.array([link_positions[6:9],
                           link_positions[0:3],
                           link_positions[9:12],
                           link_positions[3:6]])
        
        
    def reset(self):
        """Reset the Trajectory Generators, PMTG's parameters.
        """

        self._last_real_time = 0
        self._trajectory_generator.reset()
            
    
    def enable_tg(self):
        self._trajectory_generator._stop_command = False
        
        
    def disable_tg(self):
        self._trajectory_generator._stop_command = True

    
    def get_num_actions(self):
        return self._trajectory_generator.get_num_integrators()+12
    
    
    def get_num_states(self):
        return self._trajectory_generator.get_num_integrators()*2

    
    def get_desired_foot_position(self):
        positions = self._trajectory_generator.get_leg_desired_positions()
        return (np.array([positions[6:9],
                           positions[0:3],
                           positions[9:12],
                           positions[3:6]])).reshape(4,3)
        
    
    def increase_speed(self, delta):
        self._trajectory_generator.increase_speed(delta)