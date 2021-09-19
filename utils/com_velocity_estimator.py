"""State estimator"""
#File taken from the google repository "motion_imitation"

import numpy as np
from typing import Any, Sequence
import collections


_DEFAULT_WINDOW_SIZE = 20


class MovingWindowFilter(object):
    """A stable O(1) moving filter for incoming data streams.

    We implement the Neumaier's algorithm to calculate the moving window 
    average, which is numerically stable.

    """

    def __init__(self, window_size: int):
        """Initializes the class.

        Args:
        window_size: The moving window size.
        """
        assert window_size > 0
        self._window_size = window_size
        self._value_deque = collections.deque(maxlen=window_size)
        # The moving window sum.
        self._sum = 0
        # The correction term to compensate numerical precision loss during
        # calculation.
        self._correction = 0


    def _neumaier_sum(self, value: float):
        """Update the moving window sum using Neumaier's algorithm.

        For more details please refer to:
        https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

        Args:
        value: The new value to be added to the window.
        """

        new_sum = self._sum + value
        if abs(self._sum) >= abs(value):
            # If self._sum is bigger, low-order digits of value are lost.
            self._correction += (self._sum - new_sum) + value
        else:
            # low-order digits of sum are lost
            self._correction += (value - new_sum) + self._sum

        self._sum = new_sum


    def calculate_average(self, new_value: float) -> float:
        """Computes the moving window average in O(1) time.

        Args:
        new_value: The new value to enter the moving window.

        Returns:
        The average of the values in the window.

        """
        deque_len = len(self._value_deque)
        if deque_len < self._value_deque.maxlen:
            pass
        else:
            # The left most value to be subtracted from the moving sum.
            self._neumaier_sum(-self._value_deque[0])

        self._neumaier_sum(new_value)
        self._value_deque.append(new_value)

        return (self._sum + self._correction) / self._window_size


class COMVelocityEstimator(object):
    """Estimate the CoM velocity"""

    def __init__(
        self,
        robot: Any,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ):
        self._robot = robot
        self._window_size = window_size
        self.reset()


    def get_com_velocity_body_frame(self) -> Sequence[float]:
        """The base velocity projected in the body aligned inertial frame.

        The body aligned frame is a intertia frame that coincides with the body
        frame, but has a zero relative velocity/angular velocity to the world 
        frame.

        Returns:
        The com velocity in body aligned frame.
        """
        return self._com_velocity_body_frame


    def get_roll_pitch_yaw_rate(self):
        return self.roll, self.pitch, self.yaw_rate
        
        
    def get_x_y_velocities(self):
        return (self._com_velocity_body_frame[0],
                self._com_velocity_body_frame[1])
        
    
    def get_x_y_z_velocities(self):
        return self._com_velocity_body_frame
        
    
    def get_x_y_velocities_IMU(self):
        return (self._com_velocity_body_frame[0],
                self._com_velocity_body_frame[1],
                self.roll_rate,
                self.pitch_rate,
                self.yaw_rate)


    def get_measurements(self):
        return (self._com_velocity_body_frame[0],
                self._com_velocity_body_frame[1],
                self.roll_rate,
                self.pitch_rate,
                self.yaw_rate)


    @property
    def com_velocity_world_frame(self) -> Sequence[float]:
        return self._com_velocity_world_frame


    def reset(self):
        self._velocity_filter_x = MovingWindowFilter(
            window_size=self._window_size*3)
        self._velocity_filter_y = MovingWindowFilter(
            window_size=self._window_size*2)
        self._velocity_filter_z = MovingWindowFilter(
            window_size=self._window_size//2)
        self._yaw_rate_filter = MovingWindowFilter(
            window_size=self._window_size*2)
        self._pitch_rate_filter = MovingWindowFilter(
            window_size=self._window_size)
        self._roll_rate_filter = MovingWindowFilter(
            window_size=self._window_size)
        self._com_velocity_world_frame = np.array((0, 0, 0))
        self._com_velocity_body_frame = np.array((0, 0, 0))
        self.yaw_rate = 0
        self.pitch_rate = 0
        self.roll_rate = 0


    def update(self):
        velocity = self._robot.get_base_velocity()
        roll_rate, pitch_rate, yaw_rate = \
            self._robot.get_base_roll_pitch_yaw_rate()
        self.yaw_rate = self._yaw_rate_filter.calculate_average(yaw_rate)
        self.pitch_rate = self._pitch_rate_filter.calculate_average(pitch_rate)
        self.roll_rate = self._roll_rate_filter.calculate_average(roll_rate)


        vx = self._velocity_filter_x.calculate_average(velocity[0])
        vy = self._velocity_filter_y.calculate_average(velocity[1])
        vz = self._velocity_filter_z.calculate_average(velocity[2])
        
        self._com_velocity_world_frame = np.array((vx, vy, vz))

        base_orientation = self._robot.GetTrueBaseOrientation()
        _, inverse_rotation = self._robot.pybullet_client.invertTransform(
            (0, 0, 0), base_orientation)

        self._com_velocity_body_frame, _ = (
            self._robot.pybullet_client.multiplyTransforms(
                (0, 0, 0), inverse_rotation, self._com_velocity_world_frame,
                (0, 0, 0, 1)))
