# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

RUN pip install --upgrade pip

# Install Tesseract OCR
RUN apt-get update \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN pip install --no-cache-dir keras-ocr
RUN pip install --no-cache-dir \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

# RUN pip install python-Levenshtein


# Copy the local directory contents into the container.
COPY ImageSimilarityServer.py /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt.
# TensorFlow is already included in the base image, so it doesn't need to be in requirements.txt
RUN pip install  -r requirements.txt

# Make port 5000 available to the world outside this container.
EXPOSE 5000

# Define environment variable.
ENV NAME ImageSimilarityAPI

# Run app.py when the container launches.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "ImageSimilarityServer:app"]
