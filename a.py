import glob
from PIL import Image
from tqdm import tqdm
import os

for fname in tqdm(glob.glob('/mnt/local/knowledge_imgs/**/*.jpg', recursive=True)):
    try:
        Image.open(fname)
    except Exception:
        print('invalid ', fname)
        os.remove(fname.replace('/mnt/local/', '/mnt/data5/'))
        os.remove(fname)