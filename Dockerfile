# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all your files into the container
COPY . .

# Install the necessary libraries
# We include uvicorn to serve the API
RUN pip install pandas openenv-core pydantic openai uvicorn

# Open the port Hugging Face expects
EXPOSE 7860

# Command to run your app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]