FROM python:3.10-slim

WORKDIR /app

# Copy all files into the container
COPY . .

# Install the package using the pyproject.toml
RUN pip install .

# Expose Hugging Face's default port
EXPOSE 7860

# Run the server command
CMD ["server"]
