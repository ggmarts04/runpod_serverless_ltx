import os
import runpod
import time
import random
import base64
import torch
import requests
from diffusers import LTXConditionPipeline # Removed AutoencoderKLLTXVideo, LTXVideoTransformer3DModel, AutoModel
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video
# Removed BitsAndBytesConfig, T5EncoderModel from transformers import 

# Global Model and Pipeline Loading
MODEL_URL = "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-distilled-fp8.safetensors"
VIDEO_PIPELINE = None # Initialize to None for robust error handling

try:
    print(f"Loading VIDEO_PIPELINE from single file: {MODEL_URL}")
    VIDEO_PIPELINE = LTXConditionPipeline.from_single_file(
        MODEL_URL,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
        # Note: If fp8 specific quantization is needed and not inferred,
        # additional parameters like load_in_8bit or quantization_config might be required.
        # Starting without them to see if from_single_file handles fp8 safetensors correctly.
    )
    VIDEO_PIPELINE.to("cuda")
    print("Successfully loaded VIDEO_PIPELINE using from_single_file and moved to CUDA.")
except Exception as e:
    print(f"FATAL: Error loading VIDEO_PIPELINE with from_single_file: {e}")
    # VIDEO_PIPELINE remains None if an error occurs.
    # Depending on desired behavior, could exit here or let it fail upon first use.
    raise # Reraise the exception to ensure worker startup failure is clear in logs.


# Optional: VAE tiling if it becomes necessary for memory, LTX VAE might not support/need it.
# Ensure VIDEO_PIPELINE is loaded before attempting to access attributes.
# if VIDEO_PIPELINE and hasattr(VIDEO_PIPELINE, 'vae') and hasattr(VIDEO_PIPELINE.vae, 'enable_tiling'):
#     VIDEO_PIPELINE.vae.enable_tiling()
#     print("VAE tiling enabled.")

def generate_video_identifier():
    """
    Generate a unique identifier for the video based on the current timestamp
    and a random number.
    """
    timestamp = int(time.time() * 1000)  # Current time in milliseconds
    random_number = random.randint(1000, 9999)  # Generate a 4-digit random number
    return f"{timestamp}-{random_number}"  # Combine timestamp and random number


def generate_video(prompt_text, negative_prompt_text, output_video_path, image_input=None, video_id_for_temp_files=None):
    """
    Generates a video based on text prompt and optional image input, using the globally loaded VIDEO_PIPELINE.
    """
    global VIDEO_PIPELINE # Ensure we're using the globally loaded pipeline

    conditions = None
    temp_condition_video_path = None # Initialize here to ensure it's defined for the finally block

    if image_input:
        try:
            print(f"Loading image from: {image_input}")
            loaded_img = load_image(image_input)
            
            base_name_from_input = os.path.basename(image_input)
            sanitized_base_name = "".join(c if c.isalnum() or c in ('.', '_') else '_' for c in base_name_from_input)
            
            if video_id_for_temp_files:
                temp_condition_video_path = f"/tmp/condition_{video_id_for_temp_files}_{sanitized_base_name}.mp4"
            else: # Fallback if video_id is not passed
                temp_condition_video_path = f"/tmp/condition_{sanitized_base_name}_{int(time.time())}.mp4"

            print(f"Exporting conditioning image to temporary video: {temp_condition_video_path}")
            export_to_video([loaded_img], temp_condition_video_path, fps=1) # export single image as 1-frame video
            processed_img_for_cond = load_video(temp_condition_video_path) # load it back as tensor
            
            condition = LTXVideoCondition(video=processed_img_for_cond, frame_index=0)
            conditions = [condition]
            print("Image condition prepared.")
        except Exception as e:
            print(f"Error processing image input {image_input}: {e}. Proceeding with text-to-video.")
            conditions = None # Fallback to text-to-video if image processing fails
        finally:
            if temp_condition_video_path and os.path.exists(temp_condition_video_path):
                try:
                    os.remove(temp_condition_video_path)
                    print(f"Cleaned up temporary condition video: {temp_condition_video_path}")
                except Exception as e:
                    print(f"Error deleting temporary condition video {temp_condition_video_path}: {e}")

    # Parameters for distilled model
    recommended_timesteps = [1000, 993, 987, 981, 975, 909, 725, 0.03]
    
    current_negative_prompt = negative_prompt_text if negative_prompt_text else "worst quality, inconsistent motion, blurry, jittery, distorted"

    print(f"Generating video with prompt: '{prompt_text}'")
    if conditions:
        print("Using image conditioning.")
    else:
        print("No image conditioning (text-to-video).")

    video_frames = VIDEO_PIPELINE(
        prompt=prompt_text,
        negative_prompt=current_negative_prompt,
        conditions=conditions,
        width=768, 
        height=512, 
        num_frames=161, 
        timesteps=recommended_timesteps[:-1],
        decode_timestep=recommended_timesteps[-1],
        guidance_scale=1.0,
        max_sequence_length=256, 
    ).frames[0]
    
    export_to_video(video_frames, output_video_path, fps=24)
    print(f"Video generated and saved to {output_video_path}")


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
    input_data = event['input']
    
    prompt = input_data.get('prompt')
    negative_prompt = input_data.get('negative_prompt', None)
    image_url = input_data.get('image_url', None)
    image_base64 = input_data.get('image_base64', None)

    video_id = generate_video_identifier()
    print(f"Video ID: {video_id}")
    output_path = f"/tmp/{video_id}.mp4" 
    print(f"Output Path: {output_path}") 

    image_source_for_pipeline = None
    temp_decoded_image_path = None

    if image_url:
        image_source_for_pipeline = image_url
        print(f"Using image from URL: {image_url}")
    elif image_base64:
        try:
            # Assume PNG format for now if not specified, could be made more robust
            temp_decoded_image_path = f"/tmp/decoded_image_for_video_{video_id}.png" 
            image_data = base64.b64decode(image_base64)
            with open(temp_decoded_image_path, "wb") as f:
                f.write(image_data)
            image_source_for_pipeline = temp_decoded_image_path
            print(f"Decoded base64 image saved to: {temp_decoded_image_path}")
        except Exception as e:
            print(f"Error decoding base64 image: {e}. Proceeding with text-to-video.")
            image_source_for_pipeline = None # Fallback if decoding fails
            if temp_decoded_image_path and os.path.exists(temp_decoded_image_path):
                try:
                    os.remove(temp_decoded_image_path) # Clean up if write failed mid-way or was partial
                except Exception as del_e:
                    print(f"Error deleting partially saved decoded image: {del_e}")
            temp_decoded_image_path = None # Ensure it's None if error occurred

    # Call the unified generate_video function
    try:
        generate_video(prompt, negative_prompt, output_path, image_input=image_source_for_pipeline, video_id_for_temp_files=video_id)
    finally:
        # Clean up the temporary decoded image file if it was created
        if temp_decoded_image_path and os.path.exists(temp_decoded_image_path):
            try:
                os.remove(temp_decoded_image_path)
                print(f"Cleaned up temporary decoded image: {temp_decoded_image_path}")
            except Exception as e:
                print(f"Error deleting temporary decoded image {temp_decoded_image_path}: {e}")

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
