from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

def dummy(query, temperature=0.3):
    
    model_path = '../model/output.txt/'  
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    question = query

    input_ids = tokenizer.encode(question, return_tensors='pt')

    max_length = len(input_ids[0]) + 300  
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=temperature)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split('[EndOfText]')[0]

def QuestionList(path = '../data/SportsQuestions.json'):
    question_path = path

    with open(question_path, 'r') as file:
        questions = json.load(file)

    return questions

def IterAns(path = '../data/DefaultAnswer.json'):
    QueryList = QuestionList()
    answers = []

    for query in QueryList:
        answers.append(dummy(query, temperature=0.3))

    output_file_path = path

    with open(output_file_path, 'w') as outfile:
        json.dump(answers, outfile, indent=4)

    return answers

if __name__ == '__main__': 
    IterAns()