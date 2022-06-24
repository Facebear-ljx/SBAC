import wandb
import argparse
from RL_algos.IQL_algos import IQL
import datetime
import random
import os


def main():
    wandb.init(project="iql", entity="baiduxinkong")

    seed = random.randint(0, 1000)
    # Parameters
    parser = argparse.ArgumentParser(description='IQL')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='hopper-medium-v2', help='choose your mujoco env')
    parser.add_argument('--tau', default=0.7, type=float)
    parser.add_argument('--beta', default=3.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--scale_state', default=None)
    parser.add_argument('--scale_action', default=False)
    parser.add_argument('--lr_actor', default=3e-4, type=float)
    parser.add_argument('--lr_critic', default=3e-4, type=float)
    parser.add_argument('--n_steps', default=1e+6, type=int)
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and energy agent
    env_name = args.env_name
    wandb.run.name = f"{args.tau}_{args.beta}_{env_name}"

    iql_agent = IQL(env_name=env_name,
                    device=args.device,
                    ratio=1,
                    seed=args.seed,
                    beta=args.beta,
                    tau=args.tau,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    scale_state=args.scale_state,
                    lr_actor=args.lr_actor,
                    lr_critic=args.lr_critic,
                    gamma=args.gamma,
                    evaluate_freq=5000,
                    evalute_episodes=10,
                    )

    iql_agent.learn(total_time_step=int(1e+6))
    # export WANDB_API_KEY=cfbf81ce9bd7daca9d32f4bd1dbf26e8c93310c3
    # export CUDA_VISIBLE_DEVICES=0

if __name__ == '__main__':
    main()
