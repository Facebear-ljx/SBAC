import wandb
import argparse
from RL_algos.BEAR_algos import BEAR
import datetime


def main():
    wandb.init(project="BEAR", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with BEAR')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='hopper-medium-replay-v2', help='choose your mujoco env')
    parser.add_argument('--skip_steps', default=1, type=int, help='skip steps of TD update')
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and SBAC agent
    env_name = args.env_name
    current_time = datetime.datetime.now()
    wandb.run.name = f"{env_name}_{current_time}"

    agent_BEAR = BEAR(env_name=env_name,
                      device=args.device
                      )

    agent_BEAR.learn(total_time_step=int(1e+6))


if __name__ == '__main__':
    main()
