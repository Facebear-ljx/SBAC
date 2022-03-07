import wandb
import argparse
from RL_algos.TD3_BC_algos import TD3_BC
import datetime


def main():
    wandb.init(project="test_td3_bc", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='antmaze-medium-play-v2', help='choose your mujoco env')
    parser.add_argument('--alpha', default=30, type=float)
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and SBAC agent
    env_name = args.env_name
    current_time = datetime.datetime.now()
    wandb.run.name = f"{env_name}_{current_time}"

    agent_TD3_BC = TD3_BC(env_name=env_name,
                          device=args.device,
                          ratio=1,
                          alpha=args.alpha
                          # seed=args.seed
                          )

    agent_TD3_BC.learn(total_time_step=int(1e+6))
    # agent_TD3_BC.online_exploration(exploration_step=int(1e+3))


if __name__ == '__main__':
    main()
