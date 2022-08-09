#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=32G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="yearbook-eval-fix" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ../..

python main.py --dataset=yearbook --method=coral --coral_lambda=0.9 --momentum=0.99 --num_groups=10 --group_size=5 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=1 --split_time=1970
python main.py --dataset=yearbook --method=coral --coral_lambda=0.9 --momentum=0.99 --num_groups=10 --group_size=5 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=2 --split_time=1970
python main.py --dataset=yearbook --method=coral --coral_lambda=0.9 --momentum=0.99 --num_groups=10 --group_size=5 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=3 --split_time=1970

python main.py --dataset=yearbook --method=groupdro --num_groups=10 --group_size=5 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=1 --split_time=1970
python main.py --dataset=yearbook --method=groupdro --num_groups=10 --group_size=5 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=2 --split_time=1970
python main.py --dataset=yearbook --method=groupdro --num_groups=10 --group_size=5 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=3 --split_time=1970

python main.py --dataset=yearbook --method=irm --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=10 --group_size=5 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=1 --split_time=1970
python main.py --dataset=yearbook --method=irm --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=10 --group_size=5 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=2 --split_time=1970
python main.py --dataset=yearbook --method=irm --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=10 --group_size=5 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=3 --split_time=1970

python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=1 --split_time=1970
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=2 --split_time=1970
python main.py --dataset=yearbook --method=erm --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=3 --split_time=1970

python main.py --dataset=yearbook --method=erm --lisa --mix_alpha=2.0 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001  --offline --random_seed=1 --split_time=1970
python main.py --dataset=yearbook --method=erm --lisa --mix_alpha=2.0 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001  --offline --random_seed=2 --split_time=1970
python main.py --dataset=yearbook --method=erm --lisa --mix_alpha=2.0 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001  --offline --random_seed=3 --split_time=1970

python main.py --dataset=yearbook --method=erm --mixup --mix_alpha=2.0 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=1 --split_time=1970
python main.py --dataset=yearbook --method=erm --mixup --mix_alpha=2.0 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=2 --split_time=1970
python main.py --dataset=yearbook --method=erm --mixup --mix_alpha=2.0 --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --random_seed=3 --split_time=1970

python main.py --dataset=yearbook --method=agem --buffer_size=1000 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=1
python main.py --dataset=yearbook --method=agem --buffer_size=1000 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=2
python main.py --dataset=yearbook --method=agem --buffer_size=1000 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=3

python main.py --dataset=yearbook --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=1
python main.py --dataset=yearbook --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=2
python main.py --dataset=yearbook --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=3

python main.py --dataset=yearbook --method=ft --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=1
python main.py --dataset=yearbook --method=ft --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=2
python main.py --dataset=yearbook --method=ft --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=3

python main.py --dataset=yearbook --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=1
python main.py --dataset=yearbook --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=2
python main.py --dataset=yearbook --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=32 --train_update_iter=100 --lr=0.001 --offline --split_time=1970 --random_seed=3




