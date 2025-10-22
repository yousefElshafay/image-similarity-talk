# Image Similarity Demo

A Flask service demonstrating image similarity comparison and OCR engine performance for educational talks.

## Overview

This project compares three OCR engines (Pytesseract, EasyOCR, PaddleOCR) and implements CNN-based image similarity using EfficientNetB0.

## Features

- **Image Similarity**: EfficientNetB0 + SSIM for visual comparison
- **OCR Comparison**: Three engines with performance metrics
- **Text Similarity**: FuzzyWuzzy for extracted text comparison

## Quick Start

### Prerequisites
- Python 
- Tesseract OCR

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install PaddleOCR (required for OCR comparison)
pip install paddlepaddle paddleocr

# Install Pytesseract package
pip install pytesseract
```

**Install Tesseract OCR Engine:**

**Windows:**
```bash
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
# Or using chocolatey:
choco install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract
```

**Start the service:**
```bash
python ImageSimilarityServer.py
```

## API Endpoints

### Image Similarity
```http
POST /similarity
Content-Type: multipart/form-data

Files: img1, img2
```

### OCR Comparison
```http
POST /ocr_similarity
Content-Type: multipart/form-data

Files: img1, img2
Form: ocr_engine=pytesseract|easyocr|paddleocr
```

## OCR Engine Comparison

| Engine | Speed | Memory | Accuracy | Use Case |
|--------|-------|--------|----------|----------|
| Pytesseract | Fast (0.5-2s) | Low (50MB) | Good (80-85%) | Simple docs |
| EasyOCR | Medium (1-3s) | Medium (200MB) | Good (85-90%) | Balanced needs |
| PaddleOCR | Slow (3-8s) | High (500MB+) | Excellent (90-95%) | High accuracy |

## Project Structure

```
├── ImageSimilarityServer.py    # Main Flask service
├── requirements.txt            # Python dependencies
├── Dockerfile                 # Container setup
├── test_images/              # Sample test images
```

## Testing

Basic test example:
```python
import requests

with open('image1.jpg', 'rb') as img1, open('image2.jpg', 'rb') as img2:
    response = requests.post('http://localhost:5000/similarity', 
                           files={'img1': img1, 'img2': img2})
    print(f"Similarity: {response.json()['similarity_score']:.3f}")
```

## Docker

```bash
docker build -t image-similarity .
docker run -p 5000:5000 image-similarity
```
