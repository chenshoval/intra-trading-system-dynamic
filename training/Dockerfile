FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY utils/ /app/utils/
COPY config/ /app/config/
COPY training/ /app/training/

ENTRYPOINT ["python", "training/train_lightgbm.py"]
