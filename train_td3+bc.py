import wandb
import argparse
from RL_algos.TD3_BC_algos import TD3_BC


def main():
    wandb.init(project="TD3_BC", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', help='choose your mujoco env')
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and SBAC agent
    env_name = "walker2d-medium-v2"
    agent_TD3_BC = TD3_BC(env_name=env_name,
                          device=args.device,
                          ratio=10
                          )
    for _ in range(100):
        agent_TD3_BC.learn(total_time_step=int(1e+1))
        agent_TD3_BC.online_exploration(exploration_step=int(1e+5))


if __name__ == '__main__':
    main()
