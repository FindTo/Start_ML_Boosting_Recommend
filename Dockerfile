# Python 3.12
FROM python:3.12

# Working directory
WORKDIR /app/sources

# Copy requirements separetely (for caching)
COPY requirements.txt .

# Upgrade Pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies - no Torch
RUN pip install --no-cache-dir -r requirements.txt

# Copy sources files - scripts and models
COPY sources/ ./

# Command to launch web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]