import wandb
import argparse
from RL_algos.BCQ_algos import BCQ
import datetime


def main():
    wandb.init(project="BCQ", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with BCQ')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', help='choose your mujoco env')
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and SBAC agent
    env_name = args.env_name
    env_name = "hopper-medium-v2"
    current_time = datetime.datetime.now()
    wandb.run.name = f"{env_name}_{current_time}"

    agent_BCQ = BCQ(env_name=env_name,
                    device=args.device,
                    ratio=1
                    )

    agent_BCQ.learn(total_time_step=int(1e+6))


if __name__ == '__main__':
    main()
