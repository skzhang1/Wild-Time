#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=128G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="arxiv-reduced-30-eval-fix" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ../..

python main.py --dataset=arxiv --method=coral --offline --split_time=2016 --coral_lambda=0.9 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=coral --offline --split_time=2016 --coral_lambda=0.9 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=coral --offline --split_time=2016 --coral_lambda=0.9 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=arxiv --method=groupdro --offline --split_time=2016 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=groupdro --offline --split_time=2016 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=groupdro --offline --split_time=2016 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=arxiv --method=irm --offline --split_time=2016 --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=irm --offline --split_time=2016 --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=irm --offline --split_time=2016 --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=arxiv --method=erm --offline --split_time=2016 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=erm --offline --split_time=2016 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=erm --offline --split_time=2016 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=arxiv --method=erm --lisa --offline --split_time=2016 --mix_alpha=2.0 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=erm --lisa --offline --split_time=2016 --mix_alpha=2.0 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=erm --lisa --offline --split_time=2016 --mix_alpha=2.0 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=arxiv --method=erm --mixup --offline --split_time=2016 --mix_alpha=2.0 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=erm --mixup --offline --split_time=2016 --mix_alpha=2.0 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=erm --mixup --offline --split_time=2016 --mix_alpha=2.0 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=arxiv --method=agem --buffer_size=1000 --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=agem --buffer_size=1000 --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=agem --buffer_size=1000 --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=arxiv --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=arxiv --method=ft --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=ft --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=ft --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=arxiv --method=si --si_c=0.1 --epsilon=1e-4 --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=si --si_c=0.1 --epsilon=1e-4 --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=arxiv --method=si --si_c=0.1 --epsilon=1e-4 --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=3 --reduced_train_prop=0.3