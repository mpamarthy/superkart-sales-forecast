# Lightweight Python base image (same major version family used to train the model)
FROM python:3.12-slim

# Working directory inside the container
WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the serialized model
COPY . .

# The Flask API listens on port 7860
EXPOSE 7860

# Start the API with Gunicorn (2 workers); app:superkart_api = file app.py, object superkart_api
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:superkart_api"]
