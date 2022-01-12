import random

import wandb
import argparse
from RL_algos.Macro_algos import Macro
import datetime


def main():
    wandb.init(project="macro", entity="facebear")

    seed = random.randint(0, 1000)
    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='hopper-medium-expert-v2', help='choose your mujoco env')
    parser.add_argument('--skip_steps', type=int, default=7, help='1~4')
    parser.add_argument('--alpha', type=float, default=2.5, help='alpha')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()

    wandb.config.update(args)
    current_time = datetime.datetime.now()

    env_name = args.env_name
    wandb.run.name = f"{args.skip_steps}_{env_name}_{args.alpha}_{current_time}"
    agent_Macro = Macro(env_name=env_name,
                        device=args.device,
                        ratio=1,
                        alpha=args.alpha,
                        lmbda=args.skip_steps,
                        gamma=args.gamma,
                        seed=args.seed
                        )

    agent_Macro.learn(total_time_step=int(1e+6))
    # agent_TD3_BC.online_exploration(exploration_step=int(1e+3))


if __name__ == '__main__':
    main()
