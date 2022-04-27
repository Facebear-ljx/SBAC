for i in {1..3};
do
  python train_td3+bc.py --env_name antmaze-umaze-diverse-v2 --alpha 2.5
done

for i in {1..3};
do
  python train_td3+bc.py --env_name antmaze-umaze-v2 --alpha 2.5
done

for i in {1..3};
do
  python train_td3+bc.py --env_name antmaze-medium-diverse-v2 --alpha 2.5
done

for i in {1..3};
do
  python train_td3+bc.py --env_name antmaze-medium-play-v2 --alpha 2.5
done

for i in {1..3};
do
  python train_td3+bc.py --env_name antmaze-large-play-v2 --alpha 2.5
done

for i in {1..3};
do
  python train_td3+bc.py --env_name antmaze-large-diverse-v2 --alpha 2.5
done