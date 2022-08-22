#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="yearbook-td" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ..

python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=1 --split_time=1970
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=2 --split_time=1970
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=3 --split_time=1970

python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=1 --split_time=1975
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=2 --split_time=1975
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=3 --split_time=1975

python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=1 --split_time=1980
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=2 --split_time=1980
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=3 --split_time=1980

python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=1 --difficulty
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=2 --difficulty
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=3 --difficulty
