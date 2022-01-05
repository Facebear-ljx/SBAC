import wandb
import argparse
from RL_algos.TD3_BC_online_algos import TD3_BC_online


def main():
    wandb.init(project="train_td3_bc_online", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--ratio', default=1, type=int)
    parser.add_argument('--env_name', help='choose your mujoco env')
    args = parser.parse_args()
    wandb.config.update(args)

    # env_name = args.env_name
    env_name = "walker2d-medium-expert-v2"

    wandb.run.name = f"{args.ratio}_{env_name}"
    agent_TD3_BC_online = TD3_BC_online(env_name=env_name,
                                        device=args.device,
                                        ratio=args.ratio
                                        )
    # agent_TD3_BC_online.load_parameters()
    # for _ in range(100):
    # agent_TD3_BC_online.learn(total_time_step=int(1e+6))
    # agent_TD3_BC_online.online_exploration(exploration_step=int(1e+6))
    agent_TD3_BC_online.learn(total_time_step=1e+6)


if __name__ == '__main__':
    main()
