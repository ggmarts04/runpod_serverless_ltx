import os
import runpod
import time
import random
import base64
import torch 
from diffusers import AutoencoderKLLTXVideo, LTXPipeline, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

def generate_video_identifier():
    """
    Generate a unique identifier for the video based on the current timestamp
    and a random number.
    """
    timestamp = int(time.time() * 1000)  # Current time in milliseconds
    random_number = random.randint(1000, 9999)  # Generate a 4-digit random number
    return f"{timestamp}-{random_number}"  # Combine timestamp and random number


def generate_video_from_text(input_text, negative_text, output_path):
    """
    Generate a video from input text using the LTX-Video model.

    Args:
        input_text (str): The text to be converted into a video.
        output_path (str): The path to save the generated video.
    """
    video_pipeline = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.float16,
    )

    video_pipeline.to("cuda")
    negative_prompt = negative_text if negative_text else "worst quality, inconsistent motion, blurry, jittery, distorted"

    # Generate video from text
    video = video_pipeline(
        prompt=input_text,
        negative_prompt=negative_prompt,
        width=768,
        height=512,
        num_frames=161,
        num_inference_steps=50,
        max_sequence_length=256,
    ).frames[0]
    export_to_video(video, output_path, fps=24)


def handler(event):
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')
    negative_prompt = input.get('negative_prompt', None)
    video_id = generate_video_identifier()
    print(f"Video ID: {video_id}")
    output_path = f"/tmp/{video_id}.mp4" 
    print(f"Output Path: {output_path}") 

    # Generate the video from the prompt
    generate_video_from_text(prompt, negative_prompt, output_path)

    with open(output_path, "rb") as f:
        encoded_video = base64.b64encode(f.read()).decode('utf-8')

    return {
        "video_id": video_id,
        "video_base64": encoded_video,
        "video_src": f"data:video/mp4;base64,{encoded_video}",
    }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
