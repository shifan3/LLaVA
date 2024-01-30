import csv
import glob
from multiprocessing import Pool
import json
from tqdm import tqdm
import os
import requests

def download(x):
    uid, url = x
    to_file = f'data/formula/images/{uid[:3]}/{uid}.jpg'
    if os.path.exists(to_file):
        return
    os.makedirs(os.path.dirname(to_file), exist_ok=True)
    resp =requests.get(url)
    if resp.status_code == 200:
        with open(to_file, 'wb') as f:
            f.write(resp.content)
    else:
        print('!')
    
if __name__ == '__main__':
    urls = {}
    uids = []
    with open('data/formula/formula_urls.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for uid, url in reader:
            urls[uid] = url
            uids.append([uid, url])
    if os.name == 'nt':
        for x in tqdm(uids):
            download(x)
    else:
        pool = Pool(8)
        for _ in tqdm(pool.imap(download, uids), total = len(uids)):
            pass
    
"""
n = 0
for fname in glob.glob('data/formula/*.json'):
    with open(fname, 'r', encoding='utf-8') as f:
        items = json.load(f)
    for item in tqdm(items, desc=fname):
        is_checked = item['is_checked']
        titles = item['titles']
        for title in titles:
            v = title['value']
            if v and 'http' in v and is_checked:
                print(fname, item['question_uid'])
                n += 1

print(n)"""