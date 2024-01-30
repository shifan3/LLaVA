

import csv
import json
import glob
from tqdm import tqdm
import re
import os
import random
with open('data/formula/ocr_all.json', 'r', encoding='utf-8') as f:
    ocr_data = json.load(f)
random.shuffle(ocr_data)
ocr_data = ocr_data[:500000]
print(len(ocr_data))
with open('data/LinkSoul/LLaVA-CC3M-Pretrain-595K/chat-translated.json', 'r', encoding='utf-8') as f:
    chat_data = json.load(f)
print(len(chat_data))
for chat in tqdm(chat_data):
    chat['image'] = "LLaVA-CC3M-Pretrain-595K/images/" + chat['image']

assert isinstance(chat_data, list) and isinstance(ocr_data, list)
random.shuffle(chat_data)
chat_data = chat_data[:len(ocr_data)//8]
all_data = ocr_data + chat_data
random.shuffle(all_data)
print(len(all_data))
with open('data/formula/ocr_pretrain.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=4, ensure_ascii=False)
    