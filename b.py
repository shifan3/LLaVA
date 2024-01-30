from datasets import load_dataset
import os
import json
from tqdm import tqdm
from datasets.arrow_dataset import Dataset
dataset = load_dataset(
        'ydshieh/coco_dataset_script',
        '2017',
        keep_in_memory=False,
        data_dir=os.path.join(os.getcwd(), "data/clip/coco")
    )
dataset = dataset.remove_columns(['image_id', 'caption_id', 'height', 'width', 'file_name', 'coco_url'])

dataset = dataset.filter(lambda x : x['image_id'] % 2 == 0)

with open('data/formula/ocr.json', 'r', encoding='utf-8') as f:
    ocr_data = json.load(f)

ocr_dataset = {key:[] for key in ['caption', 'image_path']}
for item in ocr_data:
    assert item['conversations'][1]['from'] == 'gpt'
    caption = item['conversations'][1]['value']

    image_path = f"data/formula/images/{item['image']}"
    ocr_dataset['caption'].append(caption)
    ocr_dataset['image_path'].append(image_path)
    ocr_dataset = Dataset.from_dict(ocr_dataset)
ocr_dataset = Dataset.from_dict(ocr_dataset)
ocr_dataset = ocr_dataset.shuffle().train_test_split(test_size=len(dataset['test']))
ocr_dataset1 = ocr_dataset['train'].train_test_split(test_size=len(dataset['validation']))
ocr_dataset['train'] = ocr_dataset1['train']
ocr_dataset['validation'] = ocr_dataset1['test']
ocr_dataset1 = None

def merge_dataset(ds1:Dataset, ds2:Dataset):
    keys = list(ds1[0].keys())
    ret = {key:[] for key in keys}
    for ds in [ds1, ds2]:
        for item in ds:
            for key in keys:
                ret[key].append(item[key])
    return Dataset.from_dict(ret)
