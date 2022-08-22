#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=32G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="mimic-mortality-reduced-30" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ../..

python main.py --dataset=mimic --method=coral --offline --prediction_type=mortality --coral_lambda=0.9 --num_groups=4 --group_size=3 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=coral --offline --prediction_type=mortality --coral_lambda=0.9 --num_groups=4 --group_size=3 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=coral --offline --prediction_type=mortality --coral_lambda=0.9 --num_groups=4 --group_size=3 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=mimic --method=groupdro --offline --prediction_type=mortality --num_groups=4 --group_size=3 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=groupdro --offline --prediction_type=mortality --num_groups=4 --group_size=3 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=groupdro --offline --prediction_type=mortality --num_groups=4 --group_size=3 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=mimic --method=irm --offline --prediction_type=mortality --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=4 --group_size=3 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=irm --offline --prediction_type=mortality --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=4 --group_size=3 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=irm --offline --prediction_type=mortality --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=4 --group_size=3 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=mimic --method=erm --offline --prediction_type=mortality --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=erm --offline --prediction_type=mortality --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=erm --offline --prediction_type=mortality --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=mimic --method=erm --offline --lisa --mix_alpha=2.0 --prediction_type=mortality --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=erm --offline --lisa --mix_alpha=2.0 --prediction_type=mortality --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=erm --offline --lisa --mix_alpha=2.0 --prediction_type=mortality --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=mimic --method=erm --offline --mixup --mix_alpha=2.0 --prediction_type=mortality --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=erm --offline --mixup --mix_alpha=2.0 --prediction_type=mortality --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=mimic --method=erm --offline --mixup --mix_alpha=2.0 --prediction_type=mortality --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=mimic --prediction_type=mortality --method=agem --buffer_size=1000 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=mimic --prediction_type=mortality --method=agem --buffer_size=1000 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=mimic --prediction_type=mortality --method=agem --buffer_size=1000 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=mimic --prediction_type=mortality --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=mimic --prediction_type=mortality --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=mimic --prediction_type=mortality --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=mimic --prediction_type=mortality --method=ft --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=mimic --prediction_type=mortality --method=ft --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=mimic --prediction_type=mortality --method=ft --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=mimic --prediction_type=mortality --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=mimic --prediction_type=mortality --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=mimic --prediction_type=mortality --method=si --si_c=0.1 --epsilon=0.001 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --offline --num_workers=0 --split_time=2013 --random_seed=3 --reduced_train_prop=0.3