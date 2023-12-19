#%% 
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

#%%
def fine_tune_gpt2(InputDir, EvalDir, OutputDir):
    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load training dataset
    TrainData = TextDataset(tokenizer = tokenizer,
                            file_path = InputDir,
                            block_size = 512)
    
    EvalData = TextDataset(tokenizer = tokenizer, 
                           file_path = EvalDir, 
                           block_size = 512)

    # Create data collator for language modeling
    Collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, 
                                               mlm = False)

    # Set training arguments
    training_args = TrainingArguments(output_dir = OutputDir,
                                      overwrite_output_dir = True,
                                      num_train_epochs = 4,
                                      per_device_train_batch_size = 4,
                                      evaluation_strategy = "steps",
                                      eval_steps = 300)

    # Train the model
    trainer = Trainer(model = model,
                      args = training_args,
                      data_collator = Collator,
                      train_dataset = TrainData,
                      eval_dataset = EvalData)

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(OutputDir)

#%%
if __name__ == '__main__': 
    fine_tune_gpt2("../data/FinetuneDataTrain.txt", "../data/FinetuneDataTest.txt", "../model/output.txt")