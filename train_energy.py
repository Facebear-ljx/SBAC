import wandb
import argparse
from RL_algos.energy import Energy
import datetime
import random


def main():
    wandb.init(project="energy", entity="facebear")

    seed = random.randint(0, 1000)
    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_name', default='hopper-expert-v2', help='choose your mujoco env')
    parser.add_argument('--alpha', default=20, type=float)
    parser.add_argument('--negative_samples', default=256, type=int)
    parser.add_argument('--negative_policy', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--energy_steps', default=int(1e+6), type=int, help='total iteration steps to train EBM')
    parser.add_argument('--strong_contrastive', default=True)
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and energy agent
    env_name = args.env_name
    current_time = datetime.datetime.now()
    wandb.run.name = f"{args.alpha}_{env_name}_{args.negative_samples}_{args.strong_contrastive}"

    agent_Energy = Energy(env_name=env_name,
                          device=args.device,
                          ratio=1,
                          seed=args.seed,
                          alpha=args.alpha,
                          negative_samples=args.negative_samples,
                          batch_size=args.batch_size,
                          energy_steps=args.energy_steps,
                          negative_policy=args.negative_policy,
                          strong_contrastive=args.strong_contrastive
                          )

    agent_Energy.learn(total_time_step=int(1e+6))
    # agent_TD3_BC.online_exploration(exploration_step=int(1e+3))


if __name__ == '__main__':
    main()
