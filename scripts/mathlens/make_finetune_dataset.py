

import csv
import json
import glob
from tqdm import tqdm
import re
import os
import random

with open('data/mathlens/files.list', 'r', encoding='utf-8') as f:
    exists = set([f.strip() for f in f.readlines()])

with open('data/mathlens/files.json', 'r', encoding='utf-8') as f:
    files = json.load(f)


question_ocrs = [
    'what is the text in the picture?',
    'recognize the text in the picture',
]

train_data = []
for item in tqdm(files):
    label:str = item['label']
    path:str = item['path']
    assert 'data/mathlens/images/' in path
    image = path.replace('data/mathlens/images/', '')
    if 'tfrecords_testset' in path:
        continue
    if random.choice(range(10)) != 0:
        continue
    uid = image
    if f'images/{image}' not in exists:
        print('!', image, flush=True)
        continue
    if not label:
        continue
    q = random.choice(question_ocrs)
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
                "value": label
            }
        ]
    })
    

with open('data/mathlens/finetune.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)