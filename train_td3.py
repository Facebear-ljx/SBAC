import wandb
import argparse
import d4rl
import datetime
from RL_algos.TD3_algos import TD3


def main():
    wandb.init(project="TD3", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='hopper-medium-v2', help='choose your mujoco env')
    parser.add_argument('--start_steps', default=25e3, type=int)
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()
    wandb.config.update(args)

    current_time = datetime.datetime.now()
    env_name = "door-human-v0"
    wandb.run.name = f"{env_name}_{current_time}"
    agent_TD3 = TD3(env_name=env_name,
                    device=args.device,
                    start_steps=args.start_steps,
                    # seed=args.seed
                    )

    agent_TD3.learn(total_time_step=int(1e+6))


if __name__ == '__main__':
    main()
