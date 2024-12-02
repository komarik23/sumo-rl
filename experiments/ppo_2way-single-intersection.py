import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import PPO


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import DynamicObservationFunction, CO2DynamicObservationFunction


if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou-big-x5-co2.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=150000,
        reward_fn="khm-co2",
        delta_time = 15,
        yellow_time = 2,
        min_green = 15,
        observation_class=CO2DynamicObservationFunction
    )

    #model = PPO.load("outputs/2way-single-intersection/ppo_model", env=env)
    # print('loaded')
    
    model = PPO(
        "MlpPolicy", 
        env=env,
        learning_rate=0.01,
        verbose=1)
    
    model.learn(total_timesteps=150000)
    model.save("outputs/2way-single-intersection/ppo_model")
    print('saved')
