import openai
from retrying import retry


use_azure = True

if not use_azure:
    api_key = "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"
    #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 gpt-4
    openai.api_key = api_key
else:
    openai.api_key = 'd92659d622a641bfb0dcc93cbfd769a0'
    openai.api_version = "2023-07-01-preview"
    openai.api_type = "azure"
    openai.api_base = 'https://wtf.openai.azure.com/'

#https://platform.openai.com/docs/models/gpt-4
#@retry(stop_max_attempt_number=3, wait_fixed=10000)
def predict_raw(chat, model = "gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=chat
    )

    
    return response

#@retry(stop_max_attempt_number=3, wait_fixed=10000)
def predict_new(chat, model = "gpt-3.5-turbo", ):
    if use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=model,
            messages=chat,
            timeout = 20,
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=chat,
            timeout = 20,
        )

    answer = [c['message']['content'] for c in response['choices']]
    return answer


system_content = "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff September 2021:  Current date: March 2023."

@retry(stop_max_attempt_number=3, wait_fixed=20000,)
def predict_chat(question, history=[], model = "gpt-3.5-turbo"):
    if history is None:
        history = []
    
    # construct a openai chatgpt api request
    #openai.api_key =  "apikey"
    system_message = [
        {"role": "system", "content": system_content}]
    messages = []
    for i in range(len(history)):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": history[i]})
    messages = messages[-10:]
    # insert system content to the beginning of the messages
    messages = system_message + messages
    messages.append({"role": "user", "content": question})
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    history.append(question)
    history.append(response['choices'][0]['message']['content'])
    history_tutple = [(history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)]  # convert to tuples of list
    return history_tutple, history, response
if __name__ == '__main__':
    guess = predict_new([
            {"role": "system", "content": "你是一个中英翻译"},
            {'role':'user', 'content' : '请翻译以下文本:\nGrandpa\'s family has 8 chickens, 3 ducks, and 6 rabbits. So, how many feet do these little animals have?\nThere are 5 students throwing sandbags, if 2 sandbags are given to each of them and there is still 1 sandbag, how many sandbags are there in total?'},
            #{"role": "assistant", "content": "integers"},
            #{'role':'user', 'content' : q + '\nshould we expect the answers of this question to be integer or rational number?'},
        ], model='gpt-4')[0].lower()
    print(guess)
    exit()   
    from datasets import Dataset
    datasets = Dataset.from_csv('data/formula_en.csv')
    q1 = """solve:Grandpa's family has 8 chickens, 3 ducks, and 6 rabbits. So, how many feet do these little animals have?"""
    q2 = """solve:There are 5 students throwing sandbags, if 2 sandbags are given to each of them and there is still 1 sandbag, how many sandbags are there in total?"""
    q3 = """solve:A book has 480 pages, and Xiaohong reads 120 pages, 48 pages, and 144 pages in three days respectively. What percentage of the total number of pages does Xiaohong read in these three days?"""
    q4 = """solve:An isosceles trapezoid, the lower base is 8 cm longer than the upper base, and the sum of the lower base and a waist is 24 cm, what is the perimeter of the trapezoid in cm?"""
    for test in datasets:
        q = test['problem']
        
        guess = predict_new([
            {"role": "system", "content": "你是一个交响乐团的成员"},
            {'role':'user', 'content' : '请写一篇关于贝多芬第五交响乐的鉴赏'},
            #{"role": "assistant", "content": "integers"},
            #{'role':'user', 'content' : q + '\nshould we expect the answers of this question to be integer or rational number?'},
        ], model='gpt-4')[0].lower()
        if 'integer' in guess and 'rational' not in guess:
            print(q)
            print(guess)