#%%
import json
from sklearn.model_selection import train_test_split
import tiktoken
import numpy as np

#%%
def PreprocessJson(JsonFile, TestSize = 0.2):
    with open(JsonFile, "r", encoding = "utf-8") as f:
        data = json.load(f)
    
    PrepData = []
    
    for load in data["finetune"]:
        for question in load["questions"]:
            for response in load["responses"]:
                PrepData.append(f"User: {question}\nAssistant: {response}\n [EndOfText]\n \n")
    
    Train, Test = train_test_split(PrepData, 
                                   test_size = TestSize, 
                                   random_state = 777)
    
    return Train, Test


def PrepDataStore(TrainData, TestData, TrainOutputFile, TestOutputFile):
    with open(TrainOutputFile, "w", encoding = "utf-8") as f:
        f.write("".join(TrainData))

    with open(TestOutputFile, "w", encoding = "utf-8") as f:
        f.write("".join(TestData))

def Txt2Bin(InputDir, OutputDir):
    with open(InputDir, 'r', encoding = "utf-8") as f:
        data = f.read()
    n = len(data)

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(data)
    print(f"Data has {len(data):,} tokens")

    ids = np.array(ids, dtype = np.uint16)
    ids.tofile(OutputDir)

#%%
if __name__ == '__main__': 
    Train, Test = PreprocessJson("../data/GPTGenerated.json")
    
    PrepDataStore(Train, Test, "../data/FinetuneDataTrain.txt", "../data/FinetuneDataTest.txt")

    Txt2Bin("../data/FinetuneDataTrain.txt", "../data/train.bin")

    Txt2Bin("../data/FinetuneDataTest.txt", "../data/val.bin")
