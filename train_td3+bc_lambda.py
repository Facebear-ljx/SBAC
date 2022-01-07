import wandb
import argparse
from RL_algos.TD3_BC_td_lambda import TD3_BC_td_lambda
import datetime



def main():
    wandb.init(project="td3_bc_lambda", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='hopper-medium-v2', help='choose your mujoco env')
    parser.add_argument('--skip_steps', type=int, default=1, help='1~4')
    parser.add_argument('--alpha', type=float, default=2.5, help='alpha')
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()

    wandb.config.update(args)
    current_time = datetime.datetime.now()

    # setup mujoco environment and SBAC agent
    # env_name = "walker2d-medium-replay-v2"

    env_name = args.env_name
    # env_name = 'door-human-v0'
    wandb.run.name = f"{args.skip_steps}_{env_name}_{args.alpha}_{current_time}"
    agent_TD3_BC_lambda = TD3_BC_td_lambda(env_name=env_name,
                                           device=args.device,
                                           ratio=1,
                                           alpha=args.alpha,
                                           lmbda=args.skip_steps,
                                           # seed=seed
                                           )

    agent_TD3_BC_lambda.learn(total_time_step=int(1e+6))
    # agent_TD3_BC.online_exploration(exploration_step=int(1e+3))


if __name__ == '__main__':
    main()
