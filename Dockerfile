FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY server.py .
COPY index.html .

# Data dir (Railway will download NPZ files here at startup)
RUN mkdir -p /app/data

EXPOSE 8080

CMD ["python3", "server.py", "--data", "/app/data", "--port", "8080"]
