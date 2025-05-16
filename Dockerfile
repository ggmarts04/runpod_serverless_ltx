FROM python:3.11.9-slim

WORKDIR /

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY download_weights.py /download_weights.py
RUN python /download_weights.py

COPY rp_handler.py /

# Start the container
CMD ["python", "-u", "rp_handler.py"]