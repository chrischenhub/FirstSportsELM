import time

out_dir = '../model'
eval_interval = 5
eval_iters = 100
wandb_log = False # feel free to turn on
wandb_project = 'OmniSportsGPT'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'SportsFineTune'
init_from = 'resume' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 2 batch_size * 8 grad_accum * 1024 tokens = 16384 tokens/iter
# Train has 1,472,751 tokens, so 1 epoch ~= 89 iters

# 
batch_size = 2
gradient_accumulation_steps = 8
max_iters = 30360

# finetune at constant LR
learning_rate = 3e-5
decay_lr = True
