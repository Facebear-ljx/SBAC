import argparse
import d3rlpy
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from Sample_Dataset import Sample_from_dataset
import numpy as np
import d4rl
import gym
import wandb


def main():
    wandb.init(project="iql", entity="facebear")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='antmaze-medium-play-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    index = np.where(np.logical_and(np.logical_and(dataset.observations[:, 0] >= 0, dataset.observations[:, 0] <= 10), np.logical_and(dataset.observations[:, 1] >= 3.5, dataset.observations[:, 1] <= 4.5)))


    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0)

    iql = d3rlpy.algos.IQL(actor_learning_rate=3e-4,
                           critic_learning_rate=3e-4,
                           batch_size=256,
                           weight_temp=10.0,
                           max_weight=100.0,
                           expectile=0.9,
                           reward_scaler=reward_scaler,
                           use_gpu=args.gpu)

    # workaround for learning scheduler
    iql.create_impl(dataset.get_observation_shape(), dataset.get_action_size())
    scheduler = CosineAnnealingLR(iql.impl._actor_optim, 1000000)

    def callback(algo, epoch, total_step):
        scheduler.step()

    iql.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=1000000,
            n_steps_per_epoch=1000,
            save_interval=10,
            callback=callback,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"IQL_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    main()