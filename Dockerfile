# StreamShield - Production Container Deployment
# Innovasia 2026 PS-01 Fintech
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and artifacts
COPY . .

# Expose ports:
# 8080: Cyber-Fintech Web Command Center (SPA)
# 8501: Streamlit Operations Dashboard
EXPOSE 8080 8501

# Default launch command runs the Cyber-Fintech Command Center
CMD ["python", "app/server.py", "8080"]

