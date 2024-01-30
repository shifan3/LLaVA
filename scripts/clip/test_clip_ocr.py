import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, VisionTextDualEncoderModel, VisionTextDualEncoderProcessor
import json
import numpy as np
from tqdm import tqdm
import random
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

with open('data/formula/ocr.json', 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
ocr_data_test = ocr_data[:100]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
total_yes = 0
total = 0
config:CLIPConfig = model.config
for ocr in ocr_data_test:
    label = ocr['conversations'][1]['value']
    other_label1 = random.choice(ocr_data)['conversations'][1]['value']
    
    fname = f'data/formula/images/{ocr["image"]}'
    
    image_obj = Image.open(fname)
    inputs = processor(text=[label, other_label1], images=image_obj, return_tensors="pt", max_length=config.text_config.max_position_embeddings, padding="max_length", truncation=True).to(device)
    image_obj.close()
    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    print(probs)
    total_yes += 1 if np.argmax(probs) == 0 else 0
    total += 1


    result = 'OK' if np.argmax(probs) == 0 else 'BAD'
   
print(total_yes/total)

