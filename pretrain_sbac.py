import wandb
import argparse
from RL_algos.SBAC_algos import SBAC


def main():
    wandb.init(project="pretrain", entity="facebear")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with SBAC')
    parser.add_argument(
        '--lr_actor', type=float, default=3e-4, metavar='lr_actor', help='discount factor (default: 1e-5)')
    parser.add_argument(
        '--batch_size', type=int, default=256, metavar='BS', help='batch_size (default: 256)')
    parser.add_argument(
        '--warmup_steps', type=int, default=1000000, metavar='warmup_steps', help='warmup_steps (default: 30000)')
    parser.add_argument(
        '--num_hidden', type=int, default=256, metavar='hidden', help='num_hidden (default: 256)')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--algo', default='SBAC', help='SBAC or others')
    parser.add_argument('--env_name', help='choose your mujoco env')
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and SBAC agent
    env_name = args.env_name
    # env_name = 'hopper-medium-v2'
    agent_SBAC = SBAC(env_name=env_name,
                      num_hidden=args.num_hidden,
                      lr_actor=args.lr_actor,
                      batch_size=args.batch_size,
                      warmup_steps=args.warmup_steps,
                      device=args.device,
                      )

    # Pretrain behavior cloning
    agent_SBAC.pretrain_bc_standard()


if __name__ == '__main__':
    main()
