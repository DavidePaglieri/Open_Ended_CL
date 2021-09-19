# Cyclic integrator
# Special thanks to Atil Iscen for the implementation details

import math
import numpy as np

class CyclicIntegrator():
    """It progresses the phase of the legs.
    If the legs are decoupled, there is one cyclic integrator per leg.
    If all the legs are coupled, there is only one cyclic integrator, that
    progresses the phase for all the legs. (Uses offsets with the leg controller
    to properly move.
    
    All coupled legs is easier to train but less flexible."""
    
    def __init__(self, init_phase=0):
        self._init_phase = init_phase
        self.reset()
        
        
    def reset(self):
        self.phase = self._init_phase
        
        
    def calculate_progressed_phase(self, delta_period, 
                                   swing_stance_speed_ratio):
        """This is used to both calculate the new phase, as well as 
        the current phase of the other legs with a given offset of 
        delta_period.

        Args:
            delta_period: The fraction of the period to add to the current 
                phase of the integrator. If set to 1, the integrator will 
                accomplish one full period and return the same phase. The 
                calculated phase will depend on the current phase (if it is 
                in first half vs second half) and swing vs stance speed ratio.
            swing_stance_speed_ratio: The ratio of the speed of the phase when 
                it is in swing (second half) vs stance (first half). Set to 
                1.0 by default, making it symmetric, same as a classical 
                integrator.
        """

        stance_speed_coef = (
            swing_stance_speed_ratio + 1) * 0.5 / swing_stance_speed_ratio
        swing_speed_coef = (swing_stance_speed_ratio + 1) * 0.5
        delta_left = delta_period
        new_phase = self.phase

        # Remember that delta_period must be positive or 0.
        while delta_left > 0:
            if 0 <= new_phase < math.pi:
                delta_phase_multiplier = stance_speed_coef * math.pi*2.0
                new_phase += delta_left * delta_phase_multiplier
                delta_left = 0
                if new_phase < math.pi:
                    delta_left = 0
                else:
                    delta_left = (new_phase - math.pi) / delta_phase_multiplier
                    new_phase = math.pi
            else:
                delta_phase_multiplier = swing_speed_coef * math.pi*2.0
                new_phase += delta_left * delta_phase_multiplier
                if math.pi <= new_phase < math.pi*2.0:
                    delta_left = 0
                else:
                    delta_left = (new_phase - math.pi*2.0) / delta_phase_multiplier
                    new_phase = 0
                
        return math.fmod(new_phase, math.pi*2.0)


    def progress_phase(self, delta_period, swing_stance_ratio):
        """Updates the phase based on the current phase, delta period and ratio

        Args:
            delta_period: Delta to add to the phase
            swing_stance_ratio: ratio between the swing and stance
        """
        self.phase = self.calculate_progressed_phase(delta_period,
                                                 swing_stance_ratio)


    def get_state(self):
        """
        Returns:
            list of floats: sin and cos of the phase
        """
        return np.array([(math.cos(self.phase) + 1) / 2.0, 
                (math.sin(self.phase) + 1) / 2.0])
        
        
    def get_current_phase(self):
        """
        Returns:
            float: the current phase of the integrator
        """
        return self.phase


    def get_initial_phase(self):
        """
        Returns:
            float: the initial phase of the integrator
        """
        return self._init_phase

        