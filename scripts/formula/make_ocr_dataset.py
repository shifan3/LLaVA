

import csv
import json
import glob
from tqdm import tqdm
import re
import os
import random

urls = {}
with open('data/formula/formula_urls.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for uid, url in reader:
        urls[uid] = url

with open('data/formula/images.list', 'r', encoding='utf-8') as f:
    images = f.readlines()
images = set([image.strip() for image in images])

questions = [
    '图中的文字是什么？',
    '识别图中的文字',
]
train_data = []
files = glob.glob('data/formula/raw/*.json')
for i, fname in enumerate(files):
    with open(fname, 'r', encoding='utf-8') as f:
        items = json.load(f)
    for item in tqdm(items, desc= f'{i+1}/{len(files)} {fname}'):

        #is_checked = item['is_checked']
        titles = item['titles']
        value = []
        for title in titles:
            v = title['value']
            if 'http' in v:
                v = re.sub('http://[-=a-zA-Z/?_,0-9.]*','', v)
            value.append(v)
        value = '\n'.join(value)
        if not value:
            continue
        uid = item['question_uid']
        image = f'{uid[:3]}/{uid}.jpg'
        if f'images/{image}' not in images:
            print('!', flush=True)
            continue
        q = random.choice(questions)
        if random.choice(range(0, 2)) == 0:
            q = f'{q}\n<image>'
        else:
            q = f'<image>\n{q}'

        train_data.append({
            "id": uid,
            "image": image,
            "conversations": [
            {
                "from": "human",
                "value": q
            },
            {
                "from": "gpt",
                "value": value
            }
            ]
        })

with open('data/formula/ocr_all.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)