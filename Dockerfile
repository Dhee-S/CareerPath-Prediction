# Use official Python 3.11 slim image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first (leverage caching)
COPY setup.txt .

RUN pip install --no-cache-dir -r setup.txt

# Copy the rest of the application code
COPY . .

# Expose port (use 8000 or 5000 depending on your app)
EXPOSE 8000

# Run the web app — update this line based on your framework
# For FastAPI:
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# For Flask:
CMD ["python", "app.py"]
