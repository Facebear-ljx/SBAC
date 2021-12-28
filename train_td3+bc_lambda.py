import wandb
import argparse
from RL_algos.TD3_BC_td_lambda import TD3_BC_td_lambda


def main():
    wandb.init(project="td3_bc_lambda", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', help='choose your mujoco env')
    parser.add_argument('--skip_steps', type=int, default=2, help='1~4')
    args = parser.parse_args()

    wandb.config.update(args)

    # setup mujoco environment and SBAC agent
    # env_name = "halfcheetah-medium-replay-v2"

    env_name = args.env_name
    agent_TD3_BC_lambda = TD3_BC_td_lambda(env_name=env_name,
                                           device=args.device,
                                           ratio=1,
                                           lmbda=args.skip_steps
                                           )

    agent_TD3_BC_lambda.learn(total_time_step=int(1e+6))
    # agent_TD3_BC.online_exploration(exploration_step=int(1e+3))


if __name__ == '__main__':
    main()
