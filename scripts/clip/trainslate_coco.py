# Imports the Google Cloud Translation library
from google.cloud import translate_v2 as translate
import os
import pickle
import json
import tarfile
from tqdm import tqdm
import glob
import shutil
from chatpgt import predict_new
from multiprocessing.pool import Pool as Pool

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './data/vm-230217-33ffe7e736ab.json'

_cache = {}
_cache_file = "data/translate/translate.json"
if os.path.exists(_cache_file):
    with open(_cache_file, 'r', encoding='utf-8') as f:
        _cache = json.load(f)
    print(f"load {len(_cache)} translate cache")


def save_cache():
    fname = f'{_cache_file}.{len(_cache)}'
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(_cache, f, indent=4,ensure_ascii=False)
    if os.path.exists(_cache_file):
        os.remove(_cache_file)
    shutil.copyfile(fname, _cache_file)
    

# Initialize Translation client
def translate_texts(texts):
    try:
        prompts = '请翻译以下文本:\n' + '\n'.join(texts)
        guesses = predict_new([
                {"role": "system", "content": "你是一个中英翻译"},
                {'role':'user', 'content' : prompts},
                
            ], model='gpt-35-turbo')[0].lower()
        guesses = guesses.split('\n')
        assert len(guesses) == len(texts), '\n'.join(guesses) + "\nvs\n" + '\n'.join(texts)
        return {
            text : guess for text, guess in zip(texts, guesses)
        }
    except Exception as e:
        return {
            'error' : e,
            'input' : texts
        }
    

if __name__ == '__main__':
    
    pool = Pool(4)
    try:
        for tar_file in sorted(glob.glob('data/clip/mscoco/mscoco/*.tar')):
            t = tarfile.open(name=tar_file, mode='r:')
            names = [name for name in t.getnames() if name.endswith('.jpg')]
            
            all_texts = []
            for fname in tqdm(names, desc=f'read {tar_file}'):
                if not fname.endswith('.jpg'):
                    continue
                txt_file = fname.replace('.jpg', '.txt')
                txt_data = t.extractfile(txt_file).read().decode('utf-8')

                if '\n' in txt_data:
                    continue
                if txt_data in _cache:
                    continue
                all_texts.append(txt_data)
            
            if len(all_texts) <= 20:
                batch_size = 1
            elif len(all_texts) <= 100:
                batch_size = 4
            else:
                batch_size = 8

            translate_group = []
            translate_groups = []
            for txt_data in all_texts:
                translate_group.append(txt_data)
                if len(translate_group) == batch_size:
                    translate_groups.append(translate_group)
                    translate_group = []
            with tqdm(total=len(all_texts), desc=f'translate {tar_file}') as pbar:
                for translate_result in pool.imap(translate_texts, translate_groups):
                    if 'error' in translate_result:
                        print(translate_result['error'])
                        pbar.update(len(translate_result['input']))
                        continue
                    for text, translated in translate_result.items():
                        _cache[text] = translated
                        pbar.update(1)
            save_cache()
    finally:
        save_cache()