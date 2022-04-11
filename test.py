import d3rlpy_bl

# prepare dataset
dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-replay-v2')

# prepare algorithm
cql = d3rlpy.algos.BEAR(use_gpu=True)

# train
cql.fit(dataset,
        eval_episodes=dataset,
        n_epochs=100,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.td_error_scorer
        })