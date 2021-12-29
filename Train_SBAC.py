import wandb
import argparse
from RL_algos.SBAC_algos import SBAC
import datetime


def main():
    wandb.init(project="SBAC", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with SBAC')
    parser.add_argument(
        '--lr_actor', type=float, default=3e-4, metavar='lr_actor', help='discount factor (default: 1e-5)')
    parser.add_argument(
        '--lr_critic', type=float, default=1e-3, metavar='lr_critic', help='discount factor (default: 1e-3)')
    parser.add_argument(
        '--batch_size', type=int, default=256, metavar='BS', help='batch_size (default: 256)')
    parser.add_argument(
        '--warmup_steps', type=int, default=1000000, metavar='warmup_steps', help='warmup_steps (default: 30000)')
    parser.add_argument(
        '--num_hidden', type=int, default=256, metavar='hidden', help='num_hidden (default: 256)')
    parser.add_argument(
        '--threshold', type=float, default=8, metavar='ε', help='threshold (default: 2)')
    parser.add_argument(
        '--alpha', type=float, default=0.01, metavar='alpha', help='alpha (default: 100)')
    parser.add_argument('--skip_steps', default=1, type=int)
    parser.add_argument('--auto_alpha', default=True, help='auto-update the hyper-parameter alpha (default: True)')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--importance', default=False)
    parser.add_argument('--env_name', help='choose your mujoco env')
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and SBAC agent
    env_name = args.env_name
    current_time = datetime.datetime.now()
    wandb.run.name = f"{args.skip_steps}_{env_name}_{args.alpha}_{current_time}"

    agent_SBAC = SBAC(env_name=env_name,
                      num_hidden=args.num_hidden,
                      lr_actor=args.lr_actor,
                      lr_critic=args.lr_critic,
                      batch_size=args.batch_size,
                      warmup_steps=args.warmup_steps,
                      auto_alpha=args.auto_alpha,
                      epsilon=args.threshold,
                      alpha=args.alpha,
                      device=args.device,
                      Use_W=args.importance,
                      lmbda=args.skip_steps
                      )

    agent_SBAC.learn()


if __name__ == '__main__':
    main()
