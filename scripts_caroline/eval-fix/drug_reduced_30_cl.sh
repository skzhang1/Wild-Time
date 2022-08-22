#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="drug-reduced-30-si-eval-fix" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ../..

# python main.py --dataset=drug --method=agem --buffer_size=1000 --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=1 --reduced_train_prop=0.3
# python main.py --dataset=drug --method=agem --buffer_size=1000 --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=2 --reduced_train_prop=0.3
# python main.py --dataset=drug --method=agem --buffer_size=1000 --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=3 --reduced_train_prop=0.3
#
# python main.py --dataset=drug --method=ewc --ewc_lambda=0.5 --online --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=1 --reduced_train_prop=0.3
# python main.py --dataset=drug --method=ewc --ewc_lambda=0.5 --online --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=2 --reduced_train_prop=0.3
# python main.py --dataset=drug --method=ewc --ewc_lambda=0.5 --online --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=3 --reduced_train_prop=0.3
#
# python main.py --dataset=drug --method=ft --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=1 --reduced_train_prop=0.3
# python main.py --dataset=drug --method=ft --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=2 --reduced_train_prop=0.3
# python main.py --dataset=drug --method=ft --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=drug --method=si --si_c=0.1 --epsilon=0.001 --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=4 --reduced_train_prop=0.3
python main.py --dataset=drug --method=si --si_c=0.1 --epsilon=0.001 --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=5 --reduced_train_prop=0.3
python main.py --dataset=drug --method=si --si_c=0.1 --epsilon=0.001 --offline --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --data_dir=./Data/Drug-BA --random_seed=7 --reduced_train_prop=0.3