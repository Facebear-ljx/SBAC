echo "start training"
for i in {1}
do
  python d3rlpy_bl.py --dataset antmaze-large-play-v0
done

for i in {1}
do
  python d3rlpy_bl.py --dataset antmaze-large-diverse-v0
done

#for i in {1..4}
#do
#  python d3rlpy_bl.py --dataset antmaze-large-play-v2
#done
#
#for i in {1..5}
#do
#  python d3rlpy_bl.py --dataset antmaze-large-diverse-v2
#done