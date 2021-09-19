# Trajectory Generator
# Special thanks to Atil Iscen for helping with the implementation
# Part of the implementation is taken from PyBullet.

from pmtg import cyclic_integrator
import numpy as np
import math

#Phases
# [FL RL, FR, RR]
WALK_PHASE = [0, 0.25, 0.5, 0.75]
TROT_PHASE = [0, 0.5, 0.5, 0]
BOUND_PHASE = [0, 0.5, 0, 0.5]
PACE_PHASE = [0, 0, 0.5, 0.5]
PRONK_PHASE = [0, 0, 0, 0]

PHASE_LOWER_BOUND = 0.0
PHASE_UPPER_BOUND = 1.0
SPEED_LOWER_BOUND = 0.0
SPEED_UPPER_BOUND = 2.5
WALK_HEIGHT_LOWER_BOUND = -0.28
WALK_HEIGHT_UPPER_BOUND = -0.16
INTENSITY_LOWER_BOUND = 0.0
INTENSITY_UPPER_BOUND = 1.5
_SWING_STANCE_LOWER_BOUND = 0.2
_SWING_STANCE_UPPER_BOUND = 5.0
_DELTA_SWING_STANCE_CAP = 0.4

_DEFAULT_LEG_AMPLITUDE_SWING = 0.1
_DEFAULT_LEG_AMPLITUDE_LIFT = 0.095
_DEFAULT_LEG_AMPLITUDE_TURN = 0.05
_DEFAULT_WALKING_HEIGHT = -0.25
_DEFAULT_SPEED = 1.25
_DEFAULT_AMPLITUDE = 1

_DELTA_CENTER_EXTENSION_CAP = 0.05
_DELTA_INTENSITY_CAP = 0.1

_LEG_NAMES = ["Front Left Leg",
              "Rear Left Leg",
              "Front Right Leg",
              "Rear Right Leg"]

_LEG_COUPLING_DICT = {
    
    # All the legs are coupled.
    "coupled": [0, 0, 0, 0],
    
    # Front legs and back legs are coupled separately.
    "front back": [0, 1, 0, 1],
    
    # Left legs and right legs are coupled separately.
    "left right": [0, 0, 1, 1],
    
    # Diagonal legs are coupled (i.e. trottting).
    "diagonal": [0, 1, 1, 0],
    
    # Each leg is indepenent.
    "decoupled": [0, 1, 2, 3]
}

class TG():
    
    """The Trajectory generator converts the phase 
        (of the cyclic integrator) to xyz coordinates
        based on the parameters such as amplitude, lift, walking height, etc.
    """
    
    def __init__(self,
                speed_lower_bound = SPEED_LOWER_BOUND,
                speed_upper_bound = SPEED_UPPER_BOUND,
                walk_height_lower_bound=WALK_HEIGHT_LOWER_BOUND,
                walk_height_upper_bound=WALK_HEIGHT_UPPER_BOUND,
                intensity_lower_bound=INTENSITY_LOWER_BOUND,
                intensity_upper_bound=INTENSITY_UPPER_BOUND,
                swing_stance_lower_bound=_SWING_STANCE_LOWER_BOUND,
                swing_stance_upper_bound=_SWING_STANCE_UPPER_BOUND,
                integrator_coupling_mode="decoupled",
                walk_height_coupling_mode="coupled",
                variable_turn_multiplier=False,
                variable_swing_stance_ratio=False,
                swing_stance_ratio=1,
                default_walk_height=_DEFAULT_WALKING_HEIGHT,
                default_speed=_DEFAULT_SPEED,
                default_amplitude=_DEFAULT_AMPLITUDE,
                stop_command = True,
                init_leg_phase_offsets=TROT_PHASE):
        
        """Init of the TG
        
        Args:
            walk_height_lower_bound: Lower bound for walking height which sets 
                the default leg extension of the gait. Unit is rad, -0.5 by 
                default.
            walk_height_upper_bound: Lower bound for walking height which sets 
                the default leg extension of the gait. Unit is rad, 1.0 by 
                default.
            intensity_lower_bound: The upper bound for intensity of the 
                trajectory generator. It can be used to limit the leg movement.
            intensity_upper_bound: The upper bound for intensity of the 
                trajectory generator. It can be used to limit the leg movement.
            swing_stance_lower_bound: Lower bound for the swing vs stance ratio
                parameter. Default value is 0.2.
            swing_stance_upper_bound: Upper bound for the swing vs stance ratio
                parameter. Default value is 5.0.
            integrator_coupling_mode: How the legs should be coupled for 
                integrators.
            walk_height_coupling_mode: The same coupling mode used for walking
                heights for the legs.
            variable_turn_multiplier: A boolean to indicate if the turn
                multiplier of the TG can be changed by the policy.
            variable_swing_stance_ratio: A boolean to indicate if the swing 
                stance ratio can change per time step or not.
            swing_stance_ratio: Time taken by swing phase vs stance phase. This 
                is only relevant if variable_swing_stance_ratio is False.
            init_leg_phase_offsets: The initial phases of the legs. A list of 4
                variables within [0,1). The order is front-left, rear-left, 
                front-right and rear-right.
        """
        
        self._speed_lower_bound = speed_lower_bound
        self._speed_upper_bound = speed_upper_bound
        self._walk_height_lower_bound = walk_height_lower_bound
        self._walk_height_upper_bound = walk_height_upper_bound
        self._intensity_lower_bound = intensity_lower_bound
        self._intensity_upper_bound = intensity_upper_bound
        self._swing_stance_lower_bound = swing_stance_lower_bound
        self._swing_stance_upper_bound = swing_stance_upper_bound
        self._default_walk_height = default_walk_height
        
        self._legs = []
        # Initialize the phase of each leg
        for idx, init_phase in enumerate(init_leg_phase_offsets):
            self._legs.append(LegController(init_phase*2*math.pi,
                                            _LEG_NAMES[idx],
                                            idx))
            
        self._integrator_coupling_mode = integrator_coupling_mode
        self._integrator_id_per_leg = _LEG_COUPLING_DICT[integrator_coupling_mode]
        self._num_integrators = max(
            self._integrator_id_per_leg) + 1 if self._integrator_id_per_leg else 0
        self._legs_per_integrator_id = [[], [], [], []]
        for idx, phase_id in enumerate(self._integrator_id_per_leg):
            self._legs_per_integrator_id[phase_id].append(self._legs[idx])
        
        # For each integrator coupling, create a integrator unit.
        # For each leg controlled by that phase generator, mark the phase 
        # offset.
        self._integrator_units = []
        for legs_per_integrator in self._legs_per_integrator_id:
            if legs_per_integrator:
                _cyclic_integrator = cyclic_integrator.CyclicIntegrator(
                    legs_per_integrator[0].phase)
                self._integrator_units.append(_cyclic_integrator)
                for leg in legs_per_integrator:
                    leg.phase_offset = leg.phase - _cyclic_integrator.phase
                
        # Set the walking heights couplings.
        self._walk_height_id_per_leg = _LEG_COUPLING_DICT[walk_height_coupling_mode]
        self._num_walk_heights = max(
            self._walk_height_id_per_leg) + 1 if self._walk_height_id_per_leg else 0
        self._variable_swing_stance_ratio = variable_swing_stance_ratio
        self._variable_turn_multiplier = variable_turn_multiplier
        self._swing_stance_ratio = swing_stance_ratio
        self._speed = 0
        self._default_speed = default_speed
        self._default_amplitude = default_amplitude
        self._stop_command = stop_command
        
        
    def reset(self):
        """Reset the leg phase offsets to the initial values"""
        self._stop_command = True
        for leg in self._legs:
            leg.reset()
        
        for cyclic_integrator in self._integrator_units:
            cyclic_integrator.reset()
        
        
    def get_parameter_bounds(self):
        """Lower and upper bounds for the parameters generator's parameters.

        Returns:
            2-tuple of:
            - Lower bounds for intensity, walking height and lift.
            - Upper bounds for intensity, walking height and lift.
            If the walking height of the leg is decoupled there will be a lower
            and upper bound for each leg
        """
        lower_bounds = [self._intensity_lower_bound]
        upper_bounds = [self._intensity_upper_bound]
        lower_bounds += [self._walk_height_lower_bound] * self._num_walk_heights
        upper_bounds += [self._walk_height_upper_bound] * self._num_walk_heights
        lower_bounds += [self._swing_stance_lower_bound
                        ] * self._variable_swing_stance_ratio
        upper_bounds += [self._swing_stance_upper_bound
                        ] * self._variable_swing_stance_ratio

        return lower_bounds, upper_bounds
    
    
    def bound_residual_tg_parameters(self, speed, intensity, walking_height):
        """Clamp the TG parameters output by the policy so that they are within
        a reasonable range. In order to make the training easier, the robot
        should not fall immediately while training with the policy. Thus it is
        particularly important to constraint these parameters, especially the
        walking_height. If their values are within the bounds, they are not
        changed, otherwise they are clipped to the closer bound.
        
        Args:
            speed: the residual frequency to be added to the integrator (for the
                moment we only deal with coupled legs, so only 1 integrator)
            intensity: the intensity of swing and lift. They are multiplied to
                the default swing and lift amplitudes.
            walking_height: the z coordinate of the foot links with respect to
                the robot CoM (or the hip_offsets)
                
        Returns:
            speed, intensity, walking_height: the new bounded parameters
        """
        clamp = lambda n, minn, maxn: max(min(maxn,n),minn)
        return clamp(speed,
                     self._speed_lower_bound,
                     self._speed_upper_bound),\
                clamp(intensity, 
                     self._intensity_lower_bound,
                     self._intensity_upper_bound),\
                clamp(walking_height, 
                     self._walk_height_lower_bound,
                     self._walk_height_upper_bound)
    
    
    def increase_speed(self, delta):
        """Increase the speed up until a certain amount

        Args:
            delta ([type]): [description]
        """
        # print(delta)
        self._speed += delta
        self._speed = min(max(self._speed, self._speed_lower_bound), 
                          self._speed_upper_bound)
    
    
    def get_actions(self, delta_real_time, tg_params):
        """Get actions of the motors after increasing the phase delta_time.

        Args:
            delta_real_time: Time in seconds that have actually passed since the 
                last step of the trajectory generator.
            tg_params: An ndarray of the parameters for generating the 
                trajectory. The parameters must be in the correct order:
                (time_scale, intensity_swing, intensity_lift, 
                walking_height, and swing vs stance)
        
        Returns:
            action: a list of the x,y,z coordinates of the foot link of each leg
                in the body frame position. The policy will add the residuals.
        """
        
        # For debugging purposes we add the possibility of tg_params being
        # None, this is used when we want to try the TG alone with no policy!
        if tg_params is None:

            # Progress the phase of the integrators
            for integrator_unit in self._integrator_units:
                integrator_unit.progress_phase(self._speed * delta_real_time,
                                            self._swing_stance_ratio)
            
            # Progress phase of each LegController
            for phase_id, leg_list in enumerate(self._legs_per_integrator_id):
                for leg in leg_list:
                    delta_period = leg.phase_offset / (2.0 * math.pi)
                    leg.phase = self._integrator_units[
                        phase_id].calculate_progressed_phase(
                        delta_period, self._swing_stance_ratio)
            
            actions = []
            for leg in self._legs:
                x, y, z = leg.get_xyz_coordinates()
                actions.extend([x, y, z])
            #print(f"The TG positions are \n{actions}")
            
            return actions
        
        speeds = [0, 0, 0, 0]
        
        if not self._stop_command:
            # COUPLED
            # speeds = [tg_params[0]*0.5+self._default_speed]
            
            # DECOUPLED
            speeds = tg_params[:4]*0.5+self._default_speed

        # Progress all the phase generators based on delta time.
        for idx, integrator_unit in enumerate(self._integrator_units):
            integrator_unit.progress_phase(speeds[idx] * delta_real_time,
                                        self._swing_stance_ratio)

        # Set the phases for the legs based on their offsets with phase 
        # generators.
        for phase_id, leg_list in enumerate(self._legs_per_integrator_id):
            for leg in leg_list:
                # print(leg.phase_offset)
                delta_period = leg.phase_offset / (2.0 * math.pi)
                leg.phase = self._integrator_units[
                    phase_id].calculate_progressed_phase(
                    delta_period, self._swing_stance_ratio)

        # Get the xyz coordinates of the foot link of each leg according to the
        # current phase. Remember the order is FL, RL, FR, RR
        actions = []
        for idx, leg in enumerate(self._legs):
            x, y, z = leg.get_xyz_coordinates()
            actions.extend([x, y, z])
        return np.asarray(actions)
        
        
    def get_state(self):
        """Returns a list of floats representing the phase of the controller.

        The phase of the controller is composed of the phases of the 
        integrators. For each integrator, the phase is composed of 2 floats that 
        represents the sine and cosine of the phase of that integrator.

        Returns:
            List containing sine and cosine of the phases of all the 
            integrators.
        """
        return [x for y in self._integrator_units for x in y.get_state()]
        
        
    def get_state_lower_bounds(self):
        """Lower bounds for the internal state.

        Returns:
            The list containing the lower bounds.
        """
        return [PHASE_LOWER_BOUND] * 2 * self._num_integrators
    
    
    def get_state_upper_bounds(self):
        """Upper bounds for the internal state.

        Returns:
            The list containing the upper bounds.
        """
        return [PHASE_UPPER_BOUND] * 2 * self._num_integrators


    def get_num_integrators(self):
        """Gets the number of integrators used based on coupling mode."""
        return self._num_integrators
    
    
    def get_leg_desired_positions(self):
        """Get the leg desired x,y,z positions"""
        positions = []
        for leg in self._legs:
            x, y, z = leg.get_xyz_coordinates()
            positions.extend([x, y, z])
        return positions
    
    
    
class LegController():
    """Converts the phase into x,y,z coordinates. Has a total of 6 parameters
        amplitude_swing
        amplitude_lift
        walking_height
        intensity_swing
        intensity_lift
        swing_stance_ratio
        
        of which the last 4 are tunable by the policy
    """
    
    def __init__(self, init_phase, leg_name, leg_idx):
    
        self.amplitude_swing = _DEFAULT_LEG_AMPLITUDE_SWING
        self.amplitude_lift = _DEFAULT_LEG_AMPLITUDE_LIFT
        self.amplitude_turn = _DEFAULT_LEG_AMPLITUDE_TURN
        self.walking_height = _DEFAULT_WALKING_HEIGHT
        self._init_phase = init_phase
        self.reset()

        self.leg_name = leg_name
        self.leg_idx = leg_idx
        
        
    def reset(self):
        self.phase = self._init_phase
        self.intensity_lift = 1
        self.intensity_swing = 1
        self.intensity_turn = 0
    
    
    def adjust_walking_height(self, target_center_extension):
        delta = max(-_DELTA_CENTER_EXTENSION_CAP,
                    min(_DELTA_CENTER_EXTENSION_CAP,
                    target_center_extension - self.walking_height))
        self.walking_height += delta


    def adjust_intensity(self, target_intensity_swing, target_intensity_lift,
                         target_intensity_turn):
        """Adjust the intensity_swing and intensity_lift. The change is capped
        to a limit so that it is not too abrupt.

        Args:
            target_intensity_swing: new desired swing intensity factor
            target_intensity_lift: new desired lift intensity factor
        """
        
        # delta_swing = max(-_DELTA_INTENSITY_CAP, 
        #                   min(_DELTA_INTENSITY_CAP, target_intensity_swing - 
        #                   self.intensity_swing))
        delta_lift = max(-_DELTA_INTENSITY_CAP,
                         min(_DELTA_INTENSITY_CAP, target_intensity_lift - 
                          self.intensity_lift))
        # delta_turn = max(-_DELTA_INTENSITY_CAP,
        #                  min(_DELTA_INTENSITY_CAP, target_intensity_turn - 
        #                   self.intensity_turn))
        
        # self.intensity_swing += delta_swing
        self.intensity_lift += delta_lift
        # self.intensity_turn += delta_turn
        
        
    def get_xyz_coordinates(self):
        """Calculates the xyz coordinates of the foot link from phase,
        walking_height, amplitude_lift, intensity_lift, amplitude_swing and
        intensity_swing.

        Returns:
            3 integers: x y z coordinates of the foot link. The y position is 0
                by default, and it is controlled by the policy residuals only.
        """

        if self.phase > math.pi:
            coef = (math.sin(2*self.phase+math.pi/2)-1)/2
            z_coordinate = self.walking_height - (
                self.amplitude_lift*coef*self.intensity_lift)
        
        else:
            z_coordinate = self.walking_height

        return 0, 0, z_coordinate