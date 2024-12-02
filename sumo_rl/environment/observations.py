"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )

class SingleObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        time_since_last_phase_change = [self.ts.time_since_last_phase_change]
        veh_appeared = self.ts.get_lanes_veh_appeared() # ������ ����� �������� �� ����� ����, � ������������ ������� ����
        queue = self.ts.get_lanes_queue() # ������ ����� ����� �� ����� ����
        waiting_time = self.ts.get_lanes_waiting_time() # �������� ��� ���������� ��� ����� �� ����� ����
        observation = np.array(phase_id + time_since_last_phase_change + veh_appeared + [0,0,0,0] + queue + [0,0,0,0] + waiting_time + [0,0,0,0], dtype=np.float32)
        return observation
    
    def observation_space(self) -> spaces.Box:
        """Return the observation space. !!! absolute values - !!!"""
        return spaces.Box(
            low=np.array([
                0,0,
                0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0
            ]),
            high=np.array([
                1,1,
                300,
                100,100,100,100,100,100,100,100,
                25,25,25,25,25,25,25,25,
                10000,10000,10000,10000,10000,10000,10000,10000
            ]),
        )
    
class TwoWayObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        time_since_last_phase_change = [self.ts.time_since_last_phase_change]
        veh_appeared = self.ts.get_lanes_veh_appeared() # ������ ����� �������� �� ����� ����, � ������������ ������� ����
        queue = self.ts.get_lanes_queue() # ������ ����� ����� �� ����� ����
        waiting_time = self.ts.get_lanes_waiting_time() # �������� ��� ���������� ��� ����� �� ����� ����
        observation = np.array(phase_id + time_since_last_phase_change + veh_appeared + queue + waiting_time, dtype=np.float32)
        return observation
    
    def observation_space(self) -> spaces.Box:
        """Return the observation space. !!! absolute values - !!!"""
        return spaces.Box(
            low=np.array([
                0,0,
                0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0
            ]),
            high=np.array([
                1,1,
                300,
                100,100,100,100,100,100,100,100,
                25,25,25,25,25,25,25,25,
                10000,10000,10000,10000,10000,10000,10000,10000
            ]),
        )
# Komarov add - ���� ������� (������ �� ��� ���������� � ����� ������ ���� ��� � ��� ���������� � 2-�� ������� ����)
#   , �� ����� �������������� SingleObservationFunction ��� TwoWayObservationFunction
class DynamicObservationFunction(ObservationFunction):
    max_lines = 8

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        time_since_last_phase_change = [self.ts.time_since_last_phase_change]
        veh_appeared = self.ts.get_lanes_veh_appeared() # ������ ����� �������� �� ����� ����, � ������������ ������� ����
        if len(veh_appeared) < self.max_lines:
            veh_appeared.extend([0] * (self.max_lines - len(veh_appeared)))
        queue = self.ts.get_lanes_queue() # ������ ����� ����� �� ����� ����
        if len(queue) < self.max_lines:
            queue.extend([0] * (self.max_lines - len(queue)))
        waiting_time = self.ts.get_lanes_waiting_time() # �������� ��� ���������� ��� ����� �� ����� ����
        if len(waiting_time) < self.max_lines:
            waiting_time.extend([0] * (self.max_lines - len(waiting_time)))
        observation = np.array(phase_id + time_since_last_phase_change + veh_appeared + queue + waiting_time, dtype=np.float32)
        return observation
    
    def observation_space(self) -> spaces.Box:
        """Return the observation space. !!! absolute values - !!!"""
        return spaces.Box(
            low=np.array([
                0,0,
                0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0
            ]),
            high=np.array([
                1,1,
                600,
                100,100,100,100,100,100,100,100,
                25,25,25,25,25,25,25,25,
                10000,10000,10000,10000,10000,10000,10000,10000
            ]),
        )

class CO2DynamicObservationFunction(ObservationFunction):
    max_lines = 8

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        time_since_last_phase_change = [self.ts.time_since_last_phase_change]

        veh_appeared = self.ts.get_lanes_veh_appeared() # ������ ����� �������� �� ����� ����, � ������������ ������� ����
        if len(veh_appeared) < self.max_lines:
            veh_appeared.extend([0] * (self.max_lines - len(veh_appeared)))

        queue = self.ts.get_lanes_queue() # ������ ����� ����� �� ����� ����
        if len(queue) < self.max_lines:
            queue.extend([0] * (self.max_lines - len(queue)))

        waiting_time = self.ts.get_lanes_waiting_time() # �������� ��� ���������� ��� ����� �� ����� ����
        if len(waiting_time) < self.max_lines:
            waiting_time.extend([0] * (self.max_lines - len(waiting_time)))

        angle = [-10, 0, 20, 0]
        if len(self.ts.lanes) == 4:
            angle = angle + [-10, 0, 20, 0]
        else:
            angle = angle + angle

        co2 = self.ts.get_lanes_co2()
        if len(co2) < self.max_lines:
            co2.extend([0] * (self.max_lines - len(waiting_time)))

        veh_type = self.ts.get_lanes_veh_type()
        if len(veh_type) < self.max_lines:
            for i in (self.max_lines - len(veh_type)):
                veh_type.append([0,0,0])

        veh_type_line = []
        for line in veh_type:
            for type in line:
                veh_type_line += [type]

        observation = np.array(phase_id + time_since_last_phase_change + veh_appeared + queue + waiting_time + angle + co2 + veh_type_line, dtype=np.float32)
        return observation
    
    def observation_space(self) -> spaces.Box:
        """Return the observation space. !!! absolute values - !!!"""
        return spaces.Box(
            low=np.array([
                0,0,
                0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                -60,-60,-60,-60,-60,-60,-60,-60,
                0,0,0,0,0,0,0,0,
                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0
            ]),
            high=np.array([
                1,1,
                600,
                100,100,100,100,100,100,100,100,
                25,25,25,25,25,25,25,25,
                10000,10000,10000,10000,10000,10000,10000,10000,
                60,60,60,60,60,60,60,60,
                60,60,60,60,60,60,60,60,
                25,25,25, 25,25,25, 25,25,25, 25,25,25, 25,25,25, 25,25,25, 25,25,25, 25,25,25
            ]),
        )
