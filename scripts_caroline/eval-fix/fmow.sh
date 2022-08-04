#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="fmow-eval-fix" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ../..

python main.py --dataset=fmow --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=3 --group_size=3 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=1
python main.py --dataset=fmow --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=3 --group_size=3 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=2
python main.py --dataset=fmow --method=coral --offline --coral_lambda=0.9 --momentum=0.99 --num_groups=3 --group_size=3 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=3

python main.py --dataset=fmow --method=groupdro --offline --num_groups=3 --group_size=3 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --split_time=10 --num_workers=8 --random_seed=1
python main.py --dataset=fmow --method=groupdro --offline --num_groups=3 --group_size=3 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --split_time=10 --num_workers=8 --random_seed=2
python main.py --dataset=fmow --method=groupdro --offline --num_groups=3 --group_size=3 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --split_time=10 --num_workers=8 --random_seed=3

python main.py --dataset=fmow --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=3 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=1
python main.py --dataset=fmow --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=3 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=2
python main.py --dataset=fmow --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=3 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=3

python main.py --dataset=fmow --method=erm --offline --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=1
python main.py --dataset=fmow --method=erm --offline --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=2
python main.py --dataset=fmow --method=erm --offline --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=3

python main.py --dataset=fmow --method=erm --lisa --offline --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=1
python main.py --dataset=fmow --method=erm --lisa --offline --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=2
python main.py --dataset=fmow --method=erm --lisa --offline --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=3

python main.py --dataset=fmow --method=erm --mixup --offline --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=1
python main.py --dataset=fmow --method=erm --mixup --offline --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=2
python main.py --dataset=fmow --method=erm --mixup --offline --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=3

python main.py --dataset=fmow --method=agem --buffer_size=1000 --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=1
python main.py --dataset=fmow --method=agem --buffer_size=1000 --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=2
python main.py --dataset=fmow --method=agem --buffer_size=1000 --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=3

python main.py --dataset=fmow --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=1
python main.py --dataset=fmow --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=2
python main.py --dataset=fmow --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=3

python main.py --dataset=fmow --method=ft --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=1
python main.py --dataset=fmow --method=ft --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=2
python main.py --dataset=fmow --method=ft --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=3

python main.py --dataset=fmow --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=1
python main.py --dataset=fmow --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=2
python main.py --dataset=fmow --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=64 --train_update_iter=500 --lr=1e-4 --weight_decay=0.0 --offline --split_time=10 --random_seed=3