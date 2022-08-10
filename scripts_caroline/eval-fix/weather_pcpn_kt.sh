#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="weather-pcpn-offline" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ../..

python main.py --dataset=precipitation --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=3 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=1
python main.py --dataset=precipitation --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=3 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=2
python main.py --dataset=precipitation --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=3 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=3

python main.py --dataset=precipitation --method=groupdro --offline --num_groups=3 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=1
python main.py --dataset=precipitation --method=groupdro --offline --num_groups=3 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=2
python main.py --dataset=precipitation --method=groupdro --offline --num_groups=3 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=3

python main.py --dataset=precipitation --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=1
python main.py --dataset=precipitation --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=2
python main.py --dataset=precipitation --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=3

python main.py --dataset=precipitation --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=2 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=1
python main.py --dataset=precipitation --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=2 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=2
python main.py --dataset=precipitation --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=2 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=3

python main.py --dataset=precipitation --method=groupdro --offline --num_groups=2 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=1
python main.py --dataset=precipitation --method=groupdro --offline --num_groups=2 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=2
python main.py --dataset=precipitation --method=groupdro --offline --num_groups=2 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=3

python main.py --dataset=precipitation --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=2 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=1
python main.py --dataset=precipitation --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=2 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=2
python main.py --dataset=precipitation --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=2 --group_size=3 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=3

python main.py --dataset=precipitation --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=1
python main.py --dataset=precipitation --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=2
python main.py --dataset=precipitation --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=3

python main.py --dataset=precipitation --method=groupdro --offline --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=1
python main.py --dataset=precipitation --method=groupdro --offline --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=2
python main.py --dataset=precipitation --method=groupdro --offline --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=3

python main.py --dataset=precipitation --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=1
python main.py --dataset=precipitation --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=2
python main.py --dataset=precipitation --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=3