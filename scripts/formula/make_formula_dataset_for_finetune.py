

import csv
import json
import glob
from tqdm import tqdm
import re
import os
import random
image_folder = 'data/formula/images'
with open('data/formula/formula.json', 'r', encoding='utf-8') as f:
    formula_data = json.load(f)
print(len(formula_data))
with open('data/LinkSoul/LLaVA-Instruct-150K/llava_instruct_80k.json.modified', 'r', encoding='utf-8') as f:
    chat_data = json.load(f)
print(len(chat_data))
for d in tqdm(chat_data):
    d['image'] = 'train2017/' + d['image']

assert isinstance(chat_data, list) and isinstance(formula_data, list)
random.shuffle(chat_data)
chat_data = chat_data[:len(formula_data)//10]
all_data = formula_data + chat_data
random.shuffle(all_data)
print(len(all_data))

with open('data/formula/formula_finetune.train.json', 'w', encoding='utf-8') as f:
    json.dump(all_data[512:], f, indent=4, ensure_ascii=False)

with open('data/formula/formula_finetune.test.json', 'w', encoding='utf-8') as f:
    json.dump(all_data[:512], f, indent=4, ensure_ascii=False)
