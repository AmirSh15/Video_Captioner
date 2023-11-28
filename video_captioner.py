import open_clip
import argparse
import torch
import glob
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd

from pytorchvideo.data.video import VideoPathHandler
from transformers import Blip2Processor

from video_blip.model import VideoBlipForConditionalGeneration, process

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def video_captioner(
    video_path:str,
    device:str,
    img_model,
    image_processor,
    video_model,
    video_processor,
):
    
    # load data
    video_path_handler = VideoPathHandler()
    clip = video_path_handler.video_from_path(video_path)
    if isinstance(clip.duration, float):
        clip_length = clip.duration
    else:
        clip_length = clip.duration.numerator / clip.duration.denominator
    desired_clip_length = 10
    # only take the middle 10 seconds of the clip
    start_sec = max(0, (clip_length - desired_clip_length) // 2)
    end_sec = min(start_sec + desired_clip_length, clip_length)
    clip_tensor = clip.get_clip(start_sec, end_sec)

    # keep only 10 frames by subsampling
    num_frames = clip_tensor["video"].shape[1]
    frames = clip_tensor["video"][:, ::num_frames//10, ...].unsqueeze(0)

    # select the middle frame
    img_frame = clip_tensor["video"][:, frames.shape[1] // 2, ...]
    # convert to PIL image
    img_frame = Image.fromarray(img_frame.permute(1, 2, 0).numpy().astype("uint8"))

    # construct context
    context = ""

    # process the inputs
    with torch.no_grad(), torch.cuda.amp.autocast():
        # infer video model
        inputs = process(video_processor, video=frames, text=context).to(device)
        generated_ids = video_model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=128,
            temperature=0.1,
            top_k=50, 
            top_p=0.95,
            do_sample=True,
        )
        video_caption = video_processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].strip()
        
        # infer image model
        img_frame = image_processor(img_frame).unsqueeze(0)
        img_frame = img_frame.to(device)
        generated = img_model.generate(
            img_frame,
        )
        img_caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "").strip()
        
    # clear the cuda cache to avoid memory errors
    torch.cuda.empty_cache()
    
    return video_caption, img_caption
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default="", help="path to video root")
    parser.add_argument("--image_model", default="coca_ViT-L-14", help="coca_ViT-L-14 with mscoco_finetuned_laion2B-s13B-b90k")
    parser.add_argument("--video_model", default="kpyu/video-blip-opt-2.7b-ego4d")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--half_precision", action="store_true", default=True, help="use half precision")
    args = parser.parse_args()
    
    # check if the video_root is empty
    if args.video_root == "":
        raise ValueError("Please provide the path to video root")
    
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if args.half_precision:
        torch_precision = torch.float16
    else:
        torch_precision = torch.float32
        
    video_path = glob.glob(os.path.join(args.video_root, "*.mp4"))
    
    # load image model
    logger.info("Loading image model")
    img_model, _, image_processor = open_clip.create_model_and_transforms(
    model_name=args.image_model,
    pretrained="mscoco_finetuned_laion2B-s13B-b90k",
    precision='fp16' if torch_precision == torch.float16 else 'fp32',
    )
    # cast to device
    img_model.to(device)

    # load video model
    logger.info("Loading video model")
    video_processor = Blip2Processor.from_pretrained(args.video_model)
    video_model = VideoBlipForConditionalGeneration.from_pretrained(args.video_model, torch_dtype=torch_precision).to(device)
    
    pbar = tqdm(video_path, total=len(video_path))
    video_captions = []
    image_captions = []
    for video in pbar:
        pbar.set_description(f"Captioning {video}")
        
        video_caption, img_caption = video_captioner(
            video,
            device,
            img_model,
            image_processor,
            video_model,
            video_processor,
        )
        
        # append to list
        video_captions.append(video_caption)
        image_captions.append(img_caption)
        
    # only take the basename
    video_path = [os.path.basename(video) for video in video_path]
    # save captions
    df = pd.DataFrame({"video": video_path, "video_caption": video_captions, "image_caption": image_captions})
    df.to_csv(os.path.join(args.video_root, "captions.csv"), index=False)
        
    
    