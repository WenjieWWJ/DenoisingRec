nohup python -u main.py --dataset=$1 --model=$2 --alpha=$3 --gpu=$4 --eval_freq=2000 --batch_size=1024 > log/$1/$2_$3.log 2>&1 &
