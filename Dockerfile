FROM python:3.10-slim

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies and the local package
RUN pip install --no-cache-dir -e .

# Expose the port
EXPOSE 7860

# Use the 'server' script defined in pyproject.toml
CMD ["server"]
