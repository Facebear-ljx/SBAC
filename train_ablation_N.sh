#export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=cfbf81ce9bd7daca9d32f4bd1dbf26e8c93310c3

# hopper
for i in {1..2}
do
   python train_distance_mujoco.py --env_name hopper-medium-replay-v2 --alpha 17.5 --negative_samples 10
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name hopper-medium-replay-v2 --alpha 17.5 --negative_samples 30
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name hopper-medium-v2 --alpha 17.5 --negative_samples 10
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name hopper-medium-v2 --alpha 17.5 --negative_samples 30
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name hopper-medium-expert-v2 --alpha 7.5 --negative_samples 10
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name hopper-medium-expert-v2 --alpha 7.5 --negative_samples 30
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name hopper-random-v2 --alpha --alpha 17.5 --negative_samples 10
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name hopper-random-v2 --alpha --alpha 17.5 --negative_samples 30
done

# halfcheetah
for i in {1..2}
do
   python train_distance_mujoco.py --env_name halfcheetah-medium-expert-v2 --alpha 7.5 --negative_samples 10
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name halfcheetah-medium-expert-v2 --alpha 7.5 --negative_samples 30
done

for i in {1..2}
do 
   python train_distance_mujoco.py --env_name halfcheetah-medium-replay-v2 --alpha 7.5 --negative_samples 10
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name halfcheetah-medium-replay-v2 --alpha 7.5 --negative_samples 30
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name halfcheetah-medium-v2 --alpha 7.5 --negative_samples 10
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name halfcheetah-medium-v2 --alpha 7.5 --negative_samples 30
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name halfcheetah-random-v2 --alpha 17.5 --negative_samples 10
done

for i in {1..2}
do
   python train_distance_mujoco.py --env_name halfcheetah-random-v2 --alpha 17.5 --negative_samples 30
done

#walker2d
for i in {1..2}
do
    python train_distance_mujoco.py --env_name walker2d-medium-expert-v2 --alpha 7.5 --negative_samples 10
done

for i in {1..2}
do
    python train_distance_mujoco.py --env_name walker2d-medium-expert-v2 --alpha 7.5 --negative_samples 30
done

for i in {1..2}
do
    python train_distance_mujoco.py --env_name walker2d-medium-replay-v2 --alpha 7.5 --negative_samples 10
done

for i in {1..2}
do
    python train_distance_mujoco.py --env_name walker2d-medium-replay-v2 --alpha 7.5 --negative_samples 30
done

for i in {1..2}
do
    python train_distance_mujoco.py --env_name walker2d-medium-v2 --alpha 7.5 --negative_samples 10
done

for i in {1..2}
do
    python train_distance_mujoco.py --env_name walker2d-medium-v2 --alpha 7.5 --negative_samples 30
done

for i in {1..2}
do
    python train_distance_mujoco.py --env_name walker2d-random-v2 --alpha 17.5 --negative_samples 10
done

for i in {1..2}
do
    python train_distance_mujoco.py --env_name walker2d-random-v2 --alpha 17.5 --negative_samples 30
done