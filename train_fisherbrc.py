import wandb
import argparse
from RL_algos.FisherBRC_algos import FisherBRC
import datetime
import random
import os


def main():
    wandb.init(project="Fisher-BRC", entity="baiduxinkong")

    seed = random.randint(0, 1000)
    # Parameters
    parser = argparse.ArgumentParser(description='FisherBRC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='hopper-medium-v2', help='choose your mujoco env')
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--warmup_steps', default=int(1e+6), type=int, help='total iteration steps to train EBM')
    parser.add_argument('--scale_state', default=None)
    parser.add_argument('--scale_action', default=False)
    parser.add_argument('--lr_bc', default=3e-4, type=float)
    parser.add_argument('--lr_actor', default=3e-4, type=float)
    parser.add_argument('--lr_critic', default=3e-4, type=float)
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and energy agent
    env_name = args.env_name
    current_time = datetime.datetime.now()
    wandb.run.name = f"{args.lmbda}_{env_name}"

    agent_FisherBRC = FisherBRC(env_name=env_name,
                                device=args.device,
                                ratio=1,
                                seed=args.seed,
                                batch_size=args.batch_size,
                                warmup_steps=args.warmup_steps,
                                lmbda=args.lmbda,
                                scale_state=args.scale_state,
                                scale_action=args.scale_action,
                                lr_bc=args.lr_bc,
                                lr_actor=args.lr_actor,
                                lr_critic=args.lr_critic,
                                gamma=args.gamma,
                                evaluate_freq=5000,
                                evalute_episodes=10
                                )
    # agent_FisherBRC.warm_up(warm_time_step=int(1e+6))
    agent_FisherBRC.learn(total_time_step=int(1e+6))
    # agent_TD3_BC.online_exploration(exploration_step=int(1e+3))


if __name__ == '__main__':
    main()
