#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=32G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="mimic-readmit-eval-fix" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ../..

python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=3 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=3 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=3 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --log_dir=./checkpoints_final

python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=3 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=3 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=3 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --log_dir=./checkpoints_final

python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --log_dir=./checkpoints_final

python main.py --dataset=mimic --method=erm --offline --prediction_type=readmission --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=erm --offline --prediction_type=readmission --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=erm --offline --prediction_type=readmission --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --log_dir=./checkpoints_final

python main.py --dataset=mimic --method=erm --offline --lisa --mix_alpha=2.0 --prediction_type=readmission --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=erm --offline --lisa --mix_alpha=2.0 --prediction_type=readmission --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=erm --offline --lisa --mix_alpha=2.0 --prediction_type=readmission --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --log_dir=./checkpoints_final

python main.py --dataset=mimic --method=erm --offline --mixup --mix_alpha=2.0 --prediction_type=readmission --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=erm --offline --mixup --mix_alpha=2.0 --prediction_type=readmission --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --log_dir=./checkpoints_final
python main.py --dataset=mimic --method=erm --offline --mixup --mix_alpha=2.0 --prediction_type=readmission --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --log_dir=./checkpoints_final

python main.py --dataset=mimic --prediction_type=readmission --method=agem --buffer_size=1000 --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=1 --log_dir=./checkpoints_final
python main.py --dataset=mimic --prediction_type=readmission --method=agem --buffer_size=1000 --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=2 --log_dir=./checkpoints_final
python main.py --dataset=mimic --prediction_type=readmission --method=agem --buffer_size=1000 --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=3 --log_dir=./checkpoints_final

python main.py --dataset=mimic --prediction_type=readmission --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=1 --log_dir=./checkpoints_final
python main.py --dataset=mimic --prediction_type=readmission --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=2 --log_dir=./checkpoints_final
python main.py --dataset=mimic --prediction_type=readmission --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=3 --log_dir=./checkpoints_final

python main.py --dataset=mimic --prediction_type=readmission --method=ft --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=1 --log_dir=./checkpoints_final
python main.py --dataset=mimic --prediction_type=readmission --method=ft --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=2 --log_dir=./checkpoints_final
python main.py --dataset=mimic --prediction_type=readmission --method=ft --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=3 --log_dir=./checkpoints_final

python main.py --dataset=mimic --prediction_type=readmission --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=1 --log_dir=./checkpoints_final
python main.py --dataset=mimic --prediction_type=readmission --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=2 --log_dir=./checkpoints_final
python main.py --dataset=mimic --prediction_type=readmission --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=128 --train_update_iter=3000 --lr=1e-1 --offline --num_workers=0 --split_time=2013 --random_seed=3 --log_dir=./checkpoints_final