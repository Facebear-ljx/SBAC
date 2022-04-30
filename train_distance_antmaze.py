import wandb
import argparse
from RL_algos.energy import Energy
import datetime
import random
import os


def main():
    wandb.init(project="Distance_function_antmaze_toycase_10", entity="facebear")

    seed = random.randint(0, 1000)
    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='antmaze-large-play-v2', help='choose your mujoco env')
    parser.add_argument('--alpha', default=70, type=float)
    parser.add_argument('--gamma', default=0.995, type=float)
    parser.add_argument('--negative_samples', default=20, type=int)
    parser.add_argument('--negative_policy', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--energy_steps', default=int(1e+6), type=int, help='total iteration steps to train EBM')
    parser.add_argument('--strong_contrastive', default=False)
    parser.add_argument('--scale_state', default=None)
    parser.add_argument('--scale_action', default=False)
    parser.add_argument('--lr_ebm', default=1e-4, type=float)
    parser.add_argument('--initial_alpha', default=1., type=float)
    parser.add_argument('--lr_actor', default=3e-4, type=float)
    parser.add_argument('--lr_critic', default=1e-3, type=float)
    parser.add_argument('--lmbda_min', default=1, type=float)
    parser.add_argument('--toycase', default=False)
    parser.add_argument('--sparse', default=False)
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and energy agent
    env_name = args.env_name
    current_time = datetime.datetime.now()
    wandb.run.name = f"{args.alpha}_{env_name}_{args.initial_alpha}_{args.lr_ebm}"

    agent_Energy = Energy(env_name=env_name,
                          device=args.device,
                          ratio=1,
                          seed=args.seed,
                          alpha=args.alpha,
                          negative_samples=args.negative_samples,
                          batch_size=args.batch_size,
                          energy_steps=args.energy_steps,
                          negative_policy=args.negative_policy,
                          strong_contrastive=args.strong_contrastive,
                          lmbda_min=args.lmbda_min,
                          scale_state=args.scale_state,
                          scale_action=args.scale_action,
                          lr_ebm=args.lr_ebm,
                          lr_actor=args.lr_actor,
                          lr_critic=args.lr_critic,
                          initial_alpha=args.initial_alpha,
                          gamma=args.gamma,
                          toycase=args.toycase,
                          sparse=args.sparse,
                          evaluate_freq=100000,
                          evalute_episodes=100
                          )

    agent_Energy.learn(total_time_step=int(1e+6))
    # agent_TD3_BC.online_exploration(exploration_step=int(1e+3))


if __name__ == '__main__':
    main()
