#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=300G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="arxiv-td" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ..

python main.py --dataset=arxiv --method=erm --offline --split_time=2016 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=1 --eval_all_timesteps
# python main.py --dataset=arxiv --method=erm --offline --split_time=2016 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=2 --eval_all_timesteps
# python main.py --dataset=arxiv --method=erm --offline --split_time=2016 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=3 --eval_all_timesteps