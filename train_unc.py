import wandb
import argparse
from RL_algos.Unc_algos import TD3_BC_Unc
import datetime


def main():
    wandb.init(project="TD3_BC_Unc", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='hopper-medium-v2', help='choose your mujoco env')
    parser.add_argument('--warmup_steps', default=int(3e+4), type=int)
    parser.add_argument('--alpha', default=15, type=float)
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and TD3_Unc agent
    env_name = args.env_name
    current_time = datetime.datetime.now()
    wandb.run.name = f"{args.alpha}_{env_name}_{current_time}"

    agent_TD3_BC = TD3_BC_Unc(env_name=env_name,
                              device=args.device,
                              alpha=args.alpha,
                              warm_up_steps=args.warmup_steps,
                              ratio=1
                              )

    agent_TD3_BC.learn(total_time_step=int(1e+6))
    # agent_TD3_BC.learn_with_warmup(total_time_step=int(1e+6))


if __name__ == '__main__':
    main()
