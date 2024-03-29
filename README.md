# FirstSportsELM
### The first ever Sports Expert Language Model
  Created by Chris Zexin Chen, Sean Xie, and Chengxi Li.

  Email for questions: zc2404@nyu.edu, xx2033@nyu.edu

### This model is now on Huggingface Space for you to play around!
https://huggingface.co/spaces/Chrisneverdie/SportsDPT

As avid sports enthusiasts, we’ve consistently observed a gap in the market for a dedicated
large language model tailored to the sports domain. This research stems from our intrigue
about the potential of a language model that is exclusively trained and fine-tuned on sports-
related data. We aim to assess its performance against generic language models, thus delving
into the unique nuances and demands of the sports industry

This model structure is built by Andrej Karpathy: https://github.com/karpathy/nanoGPT

Here is an example QA from SportsDPT
![5dc29abdc17ced70ca75e2da6aa5a90](https://github.com/chrischenhub/FirstSportsELM/assets/99419764/db5f6287-8d4f-4c43-9843-de70f726d32b)

## Model Checkpoint File

https://drive.google.com/drive/folders/1PSYYWdUWiM5t0KTtlpwQ1YXBWRwV1JWi?usp=sharing

*Please place the `FineTune_ckpt.pt` file into the model directory located at `finetune/model/` to proceed with the inference process.*

## Pretrain Data 

https://drive.google.com/drive/folders/1bZvWxLnmCDYJhgMDaWumr33KbyDKQUki?usp=sharing
*train.bin ~8.4 Gb/4.5B tokens, val.bin ~4.1 Mb/2M tokens*


## Pretrain
To replicate our model, you need to use train.bin and val.bin in this drive, which is processed and ready to train.
We trained on a 4xA100 40GB node for 30 hrs to get a val loss ~2.36. Once you set up the environment, run the following:

```
$ torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py
```

You can tweak around with the parameters in train_gpt2.py. We had two experiments and the first one failed badly. 

![image](https://github.com/chrischenhub/FirstSportsELM/assets/99419764/e99ee0bd-b49a-421b-808f-796ea90a3f32)

The second trial is a success and the parameters are all stored in pretrain/train_gpt2.py

![image](https://github.com/chrischenhub/FirstSportsELM/assets/99419764/fdd474ef-c11e-4ae9-af58-4c2632bfcd5b)



## Fine Tune

We used thousands of GPT4-generated Sports QA pairs to finetune our model.

1. Generate Tags, Questions and Respones from GPT-4
```
python FineTuneDataGeneration.py YourAPIKey --Numtag 50 --NumQuestion 16 --NumParaphrase 1 --NumAnswer 2
```

2. Convert Json to TXT and Bin for fine-tuning
```
python Json2Bin.py
```
3. Fine Tune OmniSportsGPT
```
python train.py FineTuneConfig.py
```

## Ask Your Question!

1. Inference

```
# Getting an answer from our model!
python Inference.py --Question YourQuestionHere  

# Getting an answer from a GPT-2 model fine-tuned with sports-related Q&A
python DefaultAnswer.py #

# Getting an answer from a general-purpose GPT-2 model fine-tuned by third parties
python RandomGPT2ChatBot.py
```

2. Plot Result
```
python plot.py
```

## Benchmark
  Target: Sports DPT
  
  Default: GPT2 replica finetuned by sports QA
  
  Random: GPT2 size language model finetuned by general QA
  
  Llama2: Llama2 7B finetuned by general QA
  
![Alt text](image-2.png)

![Alt text](image.png)

![Alt text](image-1.png)

## Cost
The entire pretrain and finetune process costs around 250 USD. ~200$ in GPU rentals and ~50$ in OpenAI API usage.
