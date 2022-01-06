import d3rlpy

# prepare dataset
dataset, env = d3rlpy.datasets.get_d4rl('walker2d-expert-v0')

# prepare algorithm
cql = d3rlpy.algos.BEAR(use_gpu=True)

# train
cql.fit(dataset,
        eval_episodes=dataset,
        n_epochs=300,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.td_error_scorer,
        },
        tensorboard_dir='CQL'
        )
