FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /

COPY requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

COPY download_weights.py /download_weights.py
RUN python3.11 /download_weights.py

COPY rp_handler.py /

# Start the container
CMD ["python3.11", "-u", "rp_handler.py"]