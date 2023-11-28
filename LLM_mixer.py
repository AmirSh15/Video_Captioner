import argparse
import torch
import glob
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from transformers import BitsAndBytesConfig

import warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="/home/amir/Video_Captioner/data/captions.csv")
    parser.add_argument("--model", default="openchat/openchat_3.5")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--half_precision", action="store_true", default=False, help="use half precision")
    parser.add_argument("--use_4_bit", action="store_true", default=True, help="use 4 bit")
    parser.add_argument("--temperature", default=0.7, type=float)
    args = parser.parse_args()
    
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # check both half precision and 4 bit are not used
    if args.half_precision and args.use_4_bit:
        raise ValueError("Cannot use both half precision and 4 bit")

    # read csv
    df = pd.read_csv(args.csv_path)
    
    # Prepare quantized config
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # load model
    logger.info("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=nf4_config,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=True
    )
    
    openchat = pipeline("text-generation", 
                    model=model, 
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16, 
                    device_map="auto")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Mixing captions"):
        # get caption
        video_caption = row["video_caption"]
        image_caption = row["image_caption"]
        
        # create a template caption
        messages = [
            {
                "role": "system",
                "content": "Take the sentences and combine them in one sentence. Be aware that both senteces describe the same scene and person.",
            },
            {   "role": "user", 
                "content": f"1.{video_caption} 2.{image_caption}"
            },
        ]
        prompt = openchat.tokenizer.apply_chat_template(messages, 
                                                tokenize=False, 
                                                add_generation_prompt=True)
        
        with torch.no_grad(), torch.cuda.amp.autocast():

            # generate caption
            output = openchat(prompt, 
                   max_new_tokens=1024, 
                   do_sample=True, 
                   temperature=args.temperature, 
                   top_k=50, 
                   top_p=1.0,
                   pad_token_id=tokenizer.eos_token_id,
                )
            
            # decode caption
            mixed_caption = output[0]["generated_text"]
        
        # cut off the caption
        mixed_caption = mixed_caption.split("<|end_of_turn|>")[-1].replace("GPT4 Correct Assistant:", "").strip()
        
        # save caption
        df.loc[index, "mixed_caption"] = mixed_caption
        
    # save csv
    df.to_csv(args.csv_path, index=False)