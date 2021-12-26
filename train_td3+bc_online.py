import wandb
import argparse
from RL_algos.TD3_BC_online_algos import TD3_BC_online


def main():
    wandb.init(project="train_td3_bc_online", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', help='choose your mujoco env')
    args = parser.parse_args()
    wandb.config.update(args)

    # env_name = args.env_name
    env_name = "hopper-medium-replay-v2"
    agent_TD3_BC_online = TD3_BC_online(env_name=env_name,
                                        device=args.device,
                                        ratio=1
                                        )
    agent_TD3_BC_online.load_parameters()
    # for _ in range(100):
    # agent_TD3_BC_online.learn(total_time_step=int(3e+5))
    agent_TD3_BC_online.online_exploration(exploration_step=int(1e+5))
    agent_TD3_BC_online.learn(total_time_step=3e+5)


if __name__ == '__main__':
    main()
