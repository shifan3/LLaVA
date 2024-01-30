

import csv
import json
import glob
from tqdm import tqdm
import re
import os
import random
image_folder = 'data/formula/images'
with open('data/formula/ocr_all.json', 'r', encoding='utf-8') as f:
    ocr_data = json.load(f)
print(len(ocr_data))
with open('data/LinkSoul/LLaVA-Instruct-150K/llava_instruct_80k.json.modified', 'r', encoding='utf-8') as f:
    chat_data = json.load(f)
print(len(chat_data))
for d in tqdm(chat_data):
    d['image'] = 'train2017/' + d['image']

assert isinstance(chat_data, list) and isinstance(ocr_data, list)
random.shuffle(chat_data)
chat_data = chat_data[:len(ocr_data)//5]
all_data = ocr_data + chat_data
random.shuffle(all_data)
print(len(all_data))
with open('data/formula/ocr_finetune.train.json', 'w', encoding='utf-8') as f:
    json.dump(all_data[512:], f, indent=4, ensure_ascii=False)

with open('data/formula/ocr_finetune.test.json', 'w', encoding='utf-8') as f:
    json.dump(all_data[:512], f, indent=4, ensure_ascii=False)