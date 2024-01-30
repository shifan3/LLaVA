import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
import json
import numpy as np
from tqdm import tqdm
import glob
#32 0.838
#28 0.882
#24 0.838
model_name = 'openai/clip-vit-large-patch14-336'
model_name = "checkpoints/clip-vit-large-patch14-336-chn-finetuned/checkpoint-last/"
model:CLIPModel = CLIPModel.from_pretrained(model_name)
#model:CLIPModel = dual_model.vision_model
#print(type(model))
processor:CLIPProcessor = CLIPProcessor.from_pretrained(model_name)

testset = '/mnt/data5/Small-ImageNet-Validation-Dataset-1000-Classes'
with open(f'{testset}/imagenet_simple.json', 'r', encoding='utf-8') as f:
    testset_labels = json.load(f)

testset_labels = [(idx, label) for idx, label in enumerate(testset_labels) if not label.startswith('!')]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
total_yes = 0
total = 0
config:CLIPConfig = model.config
for idx, label in tqdm(testset_labels):

    other_label1 = testset_labels[(idx+50) % len(testset_labels)][1]
    label_image_dir = f'{testset}/ILSVRC2012_img_val_subset/{idx}'
    files = glob.glob(f'{label_image_dir}/*.JPEG')
    yes = 0
    for fname in files:
        image_obj = Image.open(fname)
        inputs = processor(text=[label, other_label1], images=image_obj, return_tensors="pt", max_length=config.text_config.max_position_embeddings, padding="max_length", truncation=True).to(device)
        image_obj.close()
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            print(probs)
            yes += 1 if np.argmax(probs) == 0 else 0
            total += 1


    result = 'OK' if yes / len(files) >= 0.5 else 'BAD'
    total_yes += yes
    print(result, label, yes / len(files), other_label1)  # prints: [[1., 0., 0.]]

print(total_yes/total)

