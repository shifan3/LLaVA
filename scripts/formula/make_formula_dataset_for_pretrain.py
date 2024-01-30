

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
with open('data/LinkSoul/LLaVA-CC3M-Pretrain-595K/chat-translated.json', 'r', encoding='utf-8') as f:
    chat_data = json.load(f)
print(len(chat_data))
for chat in tqdm(chat_data):
    chat['image'] = "LLaVA-CC3M-Pretrain-595K/images/" + chat['image']


assert isinstance(chat_data, list) and isinstance(formula_data, list)
random.shuffle(chat_data)
chat_data = chat_data[:len(formula_data)//8]
all_data = formula_data + chat_data
random.shuffle(all_data)
print(len(all_data))
with open('data/formula/formula_pretrain.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=4, ensure_ascii=False)