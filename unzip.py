import zipfile
import re
import os

with zipfile.ZipFile('data/coco2017-train/train2017.zip', 'r') as zip_ref:
    for file in zip_ref.namelist():
        if file == 'train2017/':
            continue
        m = re.match(r'^train2017/(\d+)\.jpg$', file)
        if not m:
            print(file)
        assert m

        to_file = f'data/coco2017-train/train2017/{m.group(1)[-3:]}/{m.group(1)}.jpg'
        
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        if os.path.isfile(to_file):
            print(to_file, 'skip')
            continue
        else:
            print(to_file)
        zip_ref.extract(file, to_file)