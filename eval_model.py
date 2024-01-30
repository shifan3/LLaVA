import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
import os
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from PIL import Image
from llava.model import *
import json
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import Conversation
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from optparse import OptionParser
from tqdm import tqdm


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. """,
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)


def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}


    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

        
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    
    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len



def main(args):
    # Model
    disable_torch_init()

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.load_8bit, args.load_4bit, device=args.device)
    
    if args.f16:
        model = model.half()
    elif args.bf16:
        model = model.bfloat16()
    else:
        model = model.to(torch.float32)
    conv_mode = "llava_llama_2"


    with open(args.test_file, 'r', encoding='utf-8') as f:
        testcases = json.load(f)
    yes = 0
    total = 0
    for testcase in tqdm(testcases):
        conv = conv_llama_2.copy()
        if args.image_root:
            image = os.path.join(args.image_root, testcase['image'])
        else:
            image = testcase['image']
        #image = '/mnt/data5/mathlens_2.0/configs/blank.jpg'
        image = load_image(image)
        conversation = testcase['conversations'][0]
        inp = conversation['value']
        expect = testcase['conversations'][1]['value'].split('#')
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        if args.f16:
            image_tensor = image_tensor.half()
        elif args.bf16:
            image_tensor = image_tensor.bfloat16()
        else:
            image_tensor = image_tensor.to(torch.float32)
            print('using float32')
        print(testcase['id'])
        #print(f"{roles[0]}: ", inp)

        #print(f"{roles[1]}: ", end="")

        #if model.config.mm_use_im_start_end:
        #    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        #else:
        #    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                #streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:],skip_special_tokens=True).strip()
        outputs = outputs.replace('解:', '').strip().split('#')
        def match(a1, a2):
            for a11 in a1:
                a11 = a11.split('>')[-1]
                for a22 in a2:
                    a22 = a22.split('>')[-1]
                    if a22 == a11:
                        return True
            return False
        print(outputs)
        if match(outputs, expect):
            yes += 1
            print('YES')
        else:
            print('NO', expect)
        total += 1
        #conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

    print(f'{yes}/{total} {yes/total}')
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--model-path", dest = 'model_path', type=str, default=None)
    #parser.add_argument("--model-base", type=str, default=None)
    parser.add_option("-t", "--test-file", dest = 'test_file', type=str)
    parser.add_option("-i", "--image-root", type=str)
    parser.add_option("-d", "--device", type=str, default="cuda")
    #parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_option("", "--temperature", type=float, default=0.2)
    parser.add_option("", "--max-new-tokens", type=int, default=512)
    parser.add_option("", "--load-8bit", action="store_true")
    parser.add_option("", "--load-4bit", action="store_true")
    parser.add_option("", "--f16", action="store_true")
    parser.add_option("", "--bf16", action="store_true")
    parser.add_option("", "--debug", action="store_true")
    opts, args = parser.parse_args()
    main(opts)
