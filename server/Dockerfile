# Use python as base image
FROM nvcr.io/nvidia/pytorch:21.09-py3
LABEL maintainer="Praneet Solanki prsol@microsoft.com"

# Use working directory /app/model
WORKDIR /app

# Copy and install required packages
COPY requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run flask app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]

