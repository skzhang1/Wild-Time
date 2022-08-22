#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=32G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="mimic-mortality-agem" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ../..

python main.py --dataset=mimic --prediction_type=mortality --method=agem --buffer_size=1000 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=1
python main.py --dataset=mimic --prediction_type=mortality --method=agem --buffer_size=1000 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=2
python main.py --dataset=mimic --prediction_type=mortality --method=agem --buffer_size=1000 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=3