

import csv
import json
import glob
from tqdm import tqdm
import re
import os
import random

def norm(q:str):
    q = q.replace('(','（').replace(')','）')
    q = re.sub(r'（\$[^$]*\$）', '（）', q)
    return q

urls = {}
with open('data/formula/formula_urls.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for uid, url in reader:
        urls[uid] = url

with open('data/formula/images.list', 'r', encoding='utf-8') as f:
    images = f.readlines()
images = set([image.strip() for image in images])
questions_ocr = [
    '图中的文字是什么？',
    '识别图中的文字',
    '这张图在问什么问题'
]


questions_answer = [
    '解答图中的问题',
    '图中问题的解答过程和答案是什么',
    '这张图的答案是什么，写出过程'
]
"""
questions_answer = [
    '识别图中的文字并解答图中的问题',
    '这张图在问什么问题？图中问题的解答过程和答案是什么？',
    '图中的文字是什么？题目的答案是什么，写出过程'
]
"""
train_data = []
files = glob.glob('data/formula/raw/*.json')
for i, fname in enumerate(files):
    with open(fname, 'r', encoding='utf-8') as f:
        items = json.load(f)
    for item in tqdm(items, desc= f'{i}/{len(files)} {fname}'):
        is_checked = item['is_checked']
        if not is_checked:
            continue
        titles = item['titles']
        detail_answers = item['detail_answers']
        formula = []
        for detail_answer in detail_answers:
            if detail_answer['formula']:
                formula.append(detail_answer['formula'])
        formula = '\n'.join(formula)
        ocr_values = []
        for title in titles:
            v = title['value']
            if 'http' in v:
                v = re.sub('http://[-=a-zA-Z/?_,0-9.]*','', v)
            ocr_values.append(v)
        ocr_values = norm('\n'.join(ocr_values))

        uid = item['question_uid']
        image = f'{uid[:3]}/{uid}.jpg'
        if f'images/{image}' not in images:
            print('!', flush=True)
            continue
        if not formula:
            continue
        q1 = random.choice(questions_ocr) 
        q2 = random.choice(questions_answer) 
        def add_image(q):
            if random.choice(range(0, 2)) == 0:
                return f'{q}\n<image>'
            else:
                return f'<image>\n{q}'
        

        if not ocr_values:
            continue

        train_data.append({
            "id": uid,
            "image": image,
            "conversations": [
            
                {
                    "from": "human",
                    "value": add_image('图中的文字为：' + ocr_values + ', 解答该问题')
                },
                {
                    "from": "gpt",
                    "value": '解:\n' + formula
                }
            ]
        })
        
        """
        train_data.append({
            "id": uid,
            "image": image,
            "conversations": [
                {
                "from": "human",
                "value": add_image(q1)
            },
            {
                "from": "gpt",
                "value": ocr_values
            },
            ]
        })"""

with open('data/formula/formula.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)