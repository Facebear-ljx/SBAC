import wandb
import argparse
from RL_algos.energy import Energy
import datetime
import random


def main():
    wandb.init(project="energy", entity="facebear")

    seed = random.randint(0, 1000)
    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='hopper-expert-v2', help='choose your mujoco env')
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and SBAC agent
    env_name = args.env_name
    current_time = datetime.datetime.now()
    wandb.run.name = f"{env_name}_{current_time}"

    agent_Energy = Energy(env_name=env_name,
                          device=args.device,
                          ratio=1,
                          seed=args.seed
                          )

    agent_Energy.learn(total_time_step=int(1e+6))
    # agent_TD3_BC.online_exploration(exploration_step=int(1e+3))


if __name__ == '__main__':
    main()
