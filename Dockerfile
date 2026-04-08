FROM python:3.10-slim

# Create a working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files into the container
COPY . .

# Expose Hugging Face's default port
EXPOSE 7860

# Command to run the FastAPI web server continuously
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
