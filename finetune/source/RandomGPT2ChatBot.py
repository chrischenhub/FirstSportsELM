from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

def dummy(query, temperature=0.3):
    tokenizer = GPT2Tokenizer.from_pretrained("Arjun-G-Ravi/chat-GPT2")
    model = GPT2LMHeadModel.from_pretrained("Arjun-G-Ravi/chat-GPT2")

    question = f'Question: {query} Answer:'

    input_ids = tokenizer.encode(question, return_tensors='pt')

    max_length = len(input_ids[0]) + 500 
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=temperature)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def QuestionList(path = '../data/SportsQuestions.json'):
    question_path = path

    with open(question_path, 'r') as file:
        questions = json.load(file)

    return questions

def IterAns(path = '../data/RandomChatBotAnswer.json'):
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
    