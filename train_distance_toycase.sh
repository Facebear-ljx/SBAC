#echo "start training"
#for i in {1..4}
#do
#  python train_distance_antmaze.py --env_name antmaze-large-play-v2 --alpha 70
#done

#for i in {1..4}
#do
#  python train_distance_antmaze.py --env_name antmaze-medium-play-v2 --alpha 65
#done
#
#for i in {1..4}
#do
#  python train_distance_antmaze.py --env_name antmaze-medium-diverse-v2 --alpha 65
#done

#for i in {1..3};
#do
#  python train_distance_antmaze.py --env_name antmaze-large-play-v2 --alpha 70 --toycase True
#done


# for i in {1..7};
# do
#   python train_distance_antmaze.py --env_name antmaze-umaze-diverse-v2 --alpha 10
# done

#for i in {1..3};
#do
#  python train_distance_antmaze.py --env_name antmaze-umaze-v2 --alpha 5 --toycase True
#done
# wandb init -p Distance_function_antmaze_toycase_10

# for i in {1..2};
# do
#   python train_distance_antmaze.py --env_name antmaze-medium-diverse-v2 --alpha 70 --toycase True
# done

#for i in {1..7};
#do
#  python train_distance_antmaze.py --env_name antmaze-medium-play-v2 --alpha 70 --toycase True
#done


for i in {1..8};
do
  python train_distance_antmaze.py --env_name antmaze-medium-diverse-v2 --alpha 70 --toycase True
done

# for i in {1..8};
# do
#   python train_distance_antmaze.py --env_name antmaze-meium-diverse-v2 --alpha 70
# done

# for i in {1..8};
# do
#   python train_distance_antmaze.py --env_name antmaze-meium-play-v2 --alpha 70
# done

#
#for i in {1..5}
#do
#  python train_distance_antmaze.py --env_name antmaze-umaze-v2 --alpha 4
#done


#for i in {1..8}
#do
#  python train_distance_antmaze.py --env_name antmaze-large-diverse-v2 --alpha 65
#done

#for i in {1..4}
#do
#  python d3rlpy_bl.py --dataset antmaze-large-play-v2
#done
#
#for i in {1..5}
#do
#  python d3rlpy_bl.py --dataset antmaze-large-diverse-v2
#done

#for i in {1..1}
#do
#  python train_distance_antmaze.py --env_name antmaze-large-play-v2 --alpha 70
#done

#for i in {1..5}
#do
#  python train_distance_antmaze.py --env_name antmaze-medium-diverse-v2 --alpha 65
#done