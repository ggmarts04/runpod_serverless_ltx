FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY models/ /models/
COPY download_weights.py /download_weights.py
RUN python /download_weights.py

COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]