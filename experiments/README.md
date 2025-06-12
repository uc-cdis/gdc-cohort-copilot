python3 ./training/train.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input_csv /path/to/train.csv \
--output_dir /path/to/training_runs/sft \
--baseline_dir /path/to/baselines/sft \
--objective sft \
--batch_size 64 \
--max_epochs 1 \
--lr 5e-5 \
--max_seq_length 1024 \
--run_name users

python3 ./training/train.py \
--model gpt2 \
--input_csv path/to/train.csv \
--output_dir /path/to/training_runs/continual \
--baseline_dir /path/to/baselines/continual \
--objective continual \
--batch_size 32 \
--max_epochs 10 \
--lr 5e-5 \
--max_seq_length 1024 \
--run_name users

python3 ./training/train.py \
--model facebook/bart-base \
--input_csv /path/to/train.csv \
--output_dir /path/to/training_runs/seq2seq \
--baseline_dir /path/to/baselines/seq2seq \
--objective seq2seq \
--batch_size 32 \
--max_epochs 3 \
--lr 5e-5 \
--max_seq_length 1024 \
--run_name users


python3 ./training/train.py \
--model gpt2 \
--input_csv /path/to/train_synthetic_users+100k.csv \
--output_dir /path/to/training_runs/continual \
--baseline_dir /path/to/baselines/continual \
--objective continual \
--batch_size 32 \
--max_epochs 10 \
--lr 5e-5 \
--max_seq_length 1024 \
--run_name 100k


python3 ./training/train.py \
--model gpt2 \
--input_csv /path/to/train_synthetic_users+1M.csv \
--output_dir /path/to/training_runs/continual \
--baseline_dir /path/to/baselines/continual \
--objective continual \
--batch_size 32 \
--max_epochs 10 \
--lr 5e-5 \
--max_seq_length 1024 \
--run_name 1M