import glob
from tqdm import tqdm
import os

for fname in tqdm(glob.glob('./**/*.sh', recursive=True)):
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('\r', '')
    with open(fname, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)