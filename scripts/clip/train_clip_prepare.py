from transformers import (
    CLIPModel,
    CLIPProcessor,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor
)
import random
from datasets import load_dataset, Dataset, DatasetDict
import os
import sys
import pickle

CLIP_MODEL = sys.argv[1]
CLIP_MODEL_NAME = CLIP_MODEL.split('/')[-1]
model = CLIPModel.from_pretrained(
    CLIP_MODEL
)

if not os.path.exists(f"models/{CLIP_MODEL_NAME}"):
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    #tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    #image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    #processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

    model.save_pretrained(f"models/{CLIP_MODEL_NAME}")
    processor.save_pretrained(f"models/{CLIP_MODEL_NAME}")

def merge_dataset(ds1:Dataset, ds2:Dataset):
    keys = list(ds1[0].keys())
    ret = {key:[] for key in keys}
    for ds in [ds1, ds2]:
        for item in ds:
            for key in keys:
                ret[key].append(item[key])
    return Dataset.from_dict(ret)


ds = load_dataset('csv', data_files='data/clip/knowledges.csv', split=['train'])
train_testvalid = ds[0].train_test_split(test_size=0.005)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
dataset1 = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']})
dataset1 = dataset1.filter(lambda x : random.choice([0,1]) == 0)
dataset = load_dataset(
        'ydshieh/coco_dataset_script',
        '2017',
        keep_in_memory=False,
        data_dir=os.path.join(os.getcwd(), "data/clip/coco")
    )
dataset = dataset.filter(lambda x : random.choice([0,1]) == 0)
dataset = dataset.remove_columns(['image_id', 'caption_id', 'height', 'width', 'file_name', 'coco_url'])
for key in dataset.data.keys():
    dataset[key] = merge_dataset(dataset1[key], dataset[key])

with open('data/clip/train_cache.bin', 'wb') as f:
    pickle.dump(dataset, f)