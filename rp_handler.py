import os
import runpod
import time
import random
import base64
import torch
import requests
from diffusers import AutoencoderKLLTXVideo, LTXPipeline, LTXVideoTransformer3DModel, AutoModel
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
    model_url = "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-distilled-fp8.safetensors"

    # Attempt to load with fp8 quantization directly if possible,
    # or prepare for it.
    # The model card mentions fp8, and diffusers docs show examples
    # with torch.float8_e4m3fn and torch.bfloat16 compute.
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True, # If this works for safetensors directly
    # ) # This might need to be load_in_4bit or other specific fp8 configs if available

    custom_transformer = AutoModel.from_single_file(
        model_url,
        torch_dtype=torch.bfloat16, # compute dtype
        # quantization_config=bnb_config, # May not be needed if safetensors handles it
                                        # or if LTXPipeline handles it later.
                                        # Start without it, add if errors or performance issues.
    )

    video_pipeline = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video", # Base model identifier
        transformer=custom_transformer,
        torch_dtype=torch.bfloat16, # Use bfloat16 as per newer docs for LTX
    )

    video_pipeline.to("cuda")
    negative_prompt = negative_text if negative_text else "worst quality, inconsistent motion, blurry, jittery, distorted"

    # Apply distilled model parameters:
    recommended_timesteps = [1000, 993, 987, 981, 975, 909, 725, 0.03] # Last value is decode_timestep

    # Generate video from text
    video = video_pipeline(
        prompt=input_text,
        negative_prompt=negative_prompt,
        width=768,
        height=512,
        num_frames=161,
        timesteps=recommended_timesteps[:-1], # Pass all but the last as timesteps
        decode_timestep=recommended_timesteps[-1], # Pass the last as decode_timestep
        guidance_scale=1.0, # Crucial for distilled models
        # num_inference_steps is implicitly defined by len(timesteps) when timesteps are provided
        max_sequence_length=256,
    ).frames[0]
    export_to_video(video, output_path, fps=24)


def upload_to_bunnycdn(file_path: str, unique_key: str) -> bool:
    upload_url = f"https://storage.bunnycdn.com/zockto/video/{unique_key}.mp4"
    access_key = "17e23633-2a7a-4d29-9450be4d6c8e-e01f-45f4"
    headers = {
        "AccessKey": access_key,
        "Content-Type": "video/mp4",
    }
    try:
        with open(file_path, "rb") as f:
            video_data = f.read()
        response = requests.put(upload_url, headers=headers, data=video_data)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        print(f"Successfully uploaded {file_path} to {upload_url}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error uploading to BunnyCDN: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return False
    except FileNotFoundError:
        print(f"Error: Local video file {file_path} not found for upload.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during BunnyCDN upload: {e}")
        return False


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

    # Upload to BunnyCDN
    upload_successful = upload_to_bunnycdn(output_path, video_id)

    if upload_successful:
        # Construct public URL
        public_video_url = f"https://zockto.b-cdn.net/video/{video_id}.mp4"
        response_data = {
            "video_id": video_id,
            "video_url": public_video_url
        }
    else:
        # Fallback or error response if upload failed
        response_data = {
            "error": "Failed to upload video to CDN.",
            "video_id": video_id
        }

    # Clean up the local file, regardless of upload success, as it's now processed or upload failed.
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print(f"Cleaned up temporary file: {output_path}")
        except OSError as e:
            print(f"Error deleting temporary file {output_path}: {e}")
            
    return response_data

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
