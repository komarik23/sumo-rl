"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
import math
from typing import Callable, List, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5
    PREV_VEH_IDS = []

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        begin_time: int,
        reward_fn: Union[str, Callable],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.sumo = sumo

        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficSignal.reward_fns.keys():
                self.reward_fn = TrafficSignal.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
        # for max green
        #if (self.green_phase == new_phase and self.time_since_last_phase_change < self.max_green + self.yellow_time) or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            # for max green
            #if self.green_phase == new_phase and self.time_since_last_phase_change > self.max_green + self.yellow_time:
                #new_phase = int(not new_phase)

            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self):
        """Computes the reward of the traffic signal."""
        self.last_reward = self.reward_fn(self)
        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward
    
    # Komarov add - наша функція винагороди
    def _khm_reward(self):
        veh_reward = self._khm_count_reward()
        waiting_reward = self._khm_waiting_reward()
        traffic_light_reward = self._khm_traffic_light_reward()

        reward = veh_reward + waiting_reward + traffic_light_reward
        return reward
    
    # Komarov add - наша функція винагороди з CO2
    def _khm_with_co2_reward(self):
        veh_reward = self._khm_count_reward()
        waiting_reward = self._khm_waiting_reward()
        traffic_light_reward = self._khm_traffic_light_reward()
        co2_reward = self._khm_co2_reward()

        reward = veh_reward + waiting_reward + traffic_light_reward + co2_reward
        return reward

    def _khm_traffic_light_reward(self):
        # На скільки кожен з світлофорів порушив задані для нього рамки (від’ємна величина, від 0 до -1 для кожного світлофору)
        phase1 = 40

        k = self.calc_optimal_k(5 * 5) # 5 delta (when worth on 10%) * 5 (random value)
        
        worth_delta = 0
        if self.is_yellow == False:
            spent = self.time_since_last_phase_change

            if spent > phase1 + 5:
                worth_delta = spent - phase1 + 5
            elif spent < phase1 - 5:
                worth_delta = phase1 - 5 - spent

        reward = (1 - math.exp(-k * worth_delta))
        return -reward

    def _khm_count_reward(self):
        # Скільки машин проїхало за період часу (додатня величина, від 0 до 1).
        k = self.calc_optimal_k(int(11 / (5 + self.MIN_GAP) * 600 * 4 / 2))
        # 11m/c, 5m veh length, 2.5m gap, 600c, 4 lanes, 2 it is a half

        combined_list = []
        for step in self.env.PASSED_INTERSECTION_VEHS:
            combined_list += step["vehIDs"]
        uniq_count = len(set(combined_list))
        
        reward = (1 - math.exp(-k * uniq_count))
        return reward

    def _khm_waiting_reward(self):
        # Скільки машин очікує проїзду більше чим 15 сек (від’ємна величина, від 0 до -1) за 600с !!!
        k = self.calc_optimal_k(4 * 5 * 10) # (4 lines * 5 vehs per line) per minute * 10 minutes
        
        combined_list = []
        for step in self.env.WAITING_STEP_VEHS:
            combined_list += step["vehIDs"]
        uniq_count = len(set(combined_list))

        reward = (1 - math.exp(-k * uniq_count))
        return -reward
    
    def _khm_co2_reward(self):
        # викиди CO2 (від’ємна величина, від 0 до -1) за 600с
        k = self.calc_optimal_k(50 * 200 * 10) # (50 vehs * 200 gram) per minute * 10 minutes

        co2 = sum(step["co2"] / 1000 for step in self.env.CO2_PERIOD)
        reward = (1 - math.exp(-k * co2))
        return -reward

    def calc_optimal_k(self, x: int):
        k = math.log(2) / x
        return k

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        )

    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue_percent(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]
    
    def get_lanes_waiting_time(self) -> List[float]:
        # Сумарний час очікування всіх машин на кожній смузі

        lanes_waiting_time = []
        for lane in self.lanes:
            lane_veh_ids = self.sumo.lane.getLastStepVehicleIDs(lane)
            waiting_time = sum(self.sumo.vehicle.getWaitingTime(veh_id) for veh_id in lane_veh_ids)
            lanes_waiting_time.append(waiting_time)

        return lanes_waiting_time

    def get_lanes_queue(self) -> List[float]:
        """Скільки машин очікує на кожній смузі. Тобто черги (кількість для кожної смуги)"""
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            for lane in self.lanes
        ]
        return [queue for queue in lanes_queue]
    
    def get_lanes_veh_appeared(self) -> List[int]:
        """Скільки машин з’явилось по кожній смузі, з попереднього проміжку часу"""
        
        new_cars_per_lane = []
        current_veh_ids = []

        for lane in self.lanes:
            current_lane_ids = self.sumo.lane.getLastStepVehicleIDs(lane)
            new_cars = [car for car in current_lane_ids if car not in self.PREV_VEH_IDS]
            current_veh_ids += new_cars
            new_cars_per_lane.append(len(new_cars))

        self.PREV_VEH_IDS.clear()
        self.PREV_VEH_IDS += current_veh_ids
        return new_cars_per_lane

        def get_lanes_co2(self) -> List[int]:
            # Викиди CO2 на кожній смузі в грамах/с
            lanes_co2 = [
                (int)(self.sumo.lane.getCO2Emission(lane) / 1000)
                for lane in self.lanes
            ]
            
            return [co2 for co2 in lanes_co2]

    def get_lanes_veh_type(self) -> List[List[int]]:
        lane_vehicles = []
        for lane in self.lanes:
            veh_type_eu4 = 0
            veh_type_eu2 = 0
            veh_type_eu0 = 0

            current_lane_ids = self.sumo.lane.getLastStepVehicleIDs(lane)

            for vehID in current_lane_ids:
                vehicle_type = self.sumo.vehicle.getTypeID(vehID)
                if vehicle_type == 'DEFAULT_VEHTYPE':
                    veh_type_eu4 += 1
                elif vehicle_type == 'type_eu2':
                    veh_type_eu2 += 1
                elif vehicle_type == 'type_eu0':
                    veh_type_eu0 +=1

            lane_vehicles.append([veh_type_eu4, veh_type_eu2, veh_type_eu0])

        return lane_vehicles

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
        "kmh": _khm_reward,
        "khm-co2": _khm_with_co2_reward
    }
