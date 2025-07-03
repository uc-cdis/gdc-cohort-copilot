# GDC Cohort LLM

We experiment with multiple variations of model and data mixture for training GDC Cohort LLM. After training, we inference the trained models to evaluate their generations. Our `run.sh` script executes all of our experimental variations; an example workflow is as follows:

### Train model
```bash
python 1-train.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/train.csv \
--log-dir $DATA_DIR/logs/mistral \
--output-dir $DATA_DIR/models/mistral \
--batch-size 64 \
--max-epochs 1 \
--lr 5e-5 \
--max-seq-length 1024
```

### Inference model
```bash
# Mistral-LoRA
python 2-generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--adapter $DATA_DIR/models/mistral \
--input-csv $DATA_DIR/test.csv \
--output-csv $DATA_DIR/generations/mistral-lora-test-generations.csv

# GPT2 or BART
python 2-generate.py \
--model $DATA_DIR/models/gpt2 \
--input-csv $DATA_DIR/test.csv \
--output-csv $DATA_DIR/generations/gpt2-test-generations.csv
```
