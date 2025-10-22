from flask import Flask, request, jsonify
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import os
import cv2
import time
import io
from scipy.spatial.distance import cosine
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from difflib import SequenceMatcher
import re
import pytesseract
from paddleocr import PaddleOCR
import gc
import easyocr
from fuzzywuzzy import fuzz


app = Flask(__name__)


base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

model = Model(inputs=base_model.input, outputs=base_model.output)

easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# paddle_reader = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, cpu_threads=4, enable_mkldnn=False)

def resize_image(pil_img, target_size=(224, 224)):
    pil_img = pil_img.resize(target_size, Image.LANCZOS)

    img_array = image.img_to_array(pil_img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(expanded_img_array)


def compute_image_similarity(org1, org2):
    """
    Computes SSIM between two PIL images (org1, org2).
    1) Convert them to NumPy arrays,
    2) Resize to the smaller shape,
    3) Convert to grayscale if needed,
    4) Return SSIM score [0..1].
    """
    img1 = np.array(org1)
    img2 = np.array(org2)

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    print(f"Image 1 - Width: {width1}, Height: {height1}")
    print(f"Image 2 - Width: {width2}, Height: {height2}")

    smaller_height = min(height1, height2)
    smaller_width = min(width1, width2)

    print(f"Smaller Width: {smaller_width}, Smaller Height: {smaller_height}")

    img1 = cv2.resize(img1, (smaller_width, smaller_height))
    img2 = cv2.resize(img2, (smaller_width, smaller_height))

    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = img2

    ssim_score, ssim_array = ssim(gray1, gray2, full=True, channel_axis=True)
    return ssim_score


def convert_transparency_to_white(image):
    """
    Converts any image with transparency (PNG/RGBA) to RGB format (JPG compatible).
    Handles both PNG with alpha channel and RGBA mode images.
    """
    print("Initial image mode:", image.mode)
    
    # Handle both PNG with alpha and RGBA images
    if image.mode in ('RGBA', 'LA') or (image.format == 'PNG' and 'A' in image.mode):
        # Create white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        
        if image.mode == 'RGBA':
            # Use alpha channel for composition
            background.paste(image, mask=image.split()[3])
        else:
            # For other modes, just paste directly
            background.paste(image)
        
        converted_image = background.convert('RGB')
        print("Converted transparent image to RGB")
    else:
        # For non-transparent images, just convert to RGB
        converted_image = image.convert('RGB')
    
    print("Image converted to RGB:", converted_image.mode)
    return converted_image

def resize_for_ocr(image, target_width=1000):
    """
    Resizes a PIL image for OCR, preserving aspect ratio.
    """
    original_width, original_height = image.size
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)
    resized_image = image.resize((target_width, target_height), Image.LANCZOS)
    return resized_image


def normalize_text(text,returnString=False):
    """
    Dynamically normalizes text by handling common OCR errors and variations
    """
    if text is None or not text:
        return None if returnString else []
    
    
    # Dynamic OCR error patterns
    ocr_patterns = {
        r'[|]': 'i',          # Vertical bar to i
        r'[¥]': 'y',          # Yen symbol to y
        r'[0oO]': 'o',        # Any 0 or O to o
        r'[1lI]': 'i',        # 1, l, or I to i
        r'[£]': 'e',          # Pound symbol to e
        r'rn': 'm',           # 'rn' combination to m
        r'vv': 'w',           # 'vv' to w
        r'(?<=[a-z])\.(?=[a-z])': '',  # Remove dots between letters
        r'[^a-z0-9\s]': ' '   # Remove special characters
    }
    
    # Convert to lowercase
    text = text.lower()
    
    # Apply OCR corrections
    for pattern, replacement in ocr_patterns.items():
        text = re.sub(pattern, replacement, text)
    
    if returnString:
        return text
    # Normalize spaces and split into words
    words = [word.strip() for word in text.split() if word.strip()]
    
    # Remove duplicates while maintaining order
    seen = set()
    return [x for x in words if not (x in seen or seen.add(x))]



def remove_White_background_and_crop(img, threshold=230):
    """
    Uses white background detection to find crop box but maintains original image quality.
    
    Args:
        img (PIL.Image): Input PIL Image object
        threshold (int): RGB threshold to identify white pixels (0-255)
    Returns:
        PIL.Image: Cropped original image or original image if no crop box found
    """
    try:
        # Convert to RGB first if needed
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
            
        # Create numpy array for analysis
        img_array = np.array(img)
        
        # Create a mask for non-white pixels using RGB channels
        r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
        white_areas = (r > threshold) & (g > threshold) & (b > threshold)
        
        # Find the bounding box of non-white pixels
        non_white_positions = np.where(~white_areas)
        
        if len(non_white_positions[0]) > 0:  # If we found non-white pixels
            # Get bounding box coordinates
            y_min, y_max = np.min(non_white_positions[0]), np.max(non_white_positions[0])
            x_min, x_max = np.min(non_white_positions[1]), np.max(non_white_positions[1])
            
            # Crop the original image using the bounding box
            return img.crop((x_min, y_min, x_max + 1, y_max + 1))
        
        # If no crop box found, return original image
        return img
        
    except Exception as e:
        print(f"Error in remove_background_and_crop: {str(e)}")
        # Return original image in case of error
        return img

def extract_foreground(image):
    """
    Basic 'foreground extraction' by thresholding white and saturation,
    then cropping to the bounding rectangle of the largest contour.
    """
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    white_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1]
    saturation_mask = cv2.threshold(hsv[:,:,1], 30, 255, cv2.THRESH_BINARY)[1]
    combined_mask = cv2.bitwise_or(white_mask, saturation_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = img.shape[0] * img.shape[1] * 0.001
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not valid_contours:
        return image  # fallback if no major contour found

    padding = 0
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0

    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img.shape[1], x_max + padding)
    y_max = min(img.shape[0], y_max + padding)

    cropped = img[y_min:y_max, x_min:x_max]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


def maintain_aspect_ratio(width, height, target_size):
    """
    Helper that calculates new (w,h) to keep the same aspect ratio
    when you want a certain target size on the smaller side.
    """
    aspect = width / height
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect)
    else:
        new_height = target_size
        new_width = int(target_size * aspect)
    return new_width, new_height

def scale_with_aspect_ratio(img1, img2, min_allowed_size=300):
    """
    Scales both images while preserving aspect ratio, ensuring no dimension is below 300px.
    
    Args:
        img1 (PIL.Image): First image
        img2 (PIL.Image): Second image
        min_allowed_size (int): Minimum allowed dimension size (default: 300)
        
    Returns:
        tuple: (scaled_img1, scaled_img2) - Scaled versions of input images
    """
    w1, h1 = img1.size
    w2, h2 = img2.size

    # Find the smallest dimension among both images
    min_size = min(w1, h1, w2, h2)
    
    # If smallest dimension is less than minimum allowed, use minimum allowed
    target_size = max(min_size, min_allowed_size)
    
    # Calculate new dimensions while maintaining aspect ratio
    new_w1, new_h1 = maintain_aspect_ratio(w1, h1, target_size)
    new_w2, new_h2 = maintain_aspect_ratio(w2, h2, target_size)
    
    # Ensure no dimension is below minimum allowed size
    if new_w1 < min_allowed_size:
        scale_factor = min_allowed_size / new_w1
        new_w1 = min_allowed_size
        new_h1 = int(new_h1 * scale_factor)
    
    if new_h1 < min_allowed_size:
        scale_factor = min_allowed_size / new_h1
        new_h1 = min_allowed_size
        new_w1 = int(new_w1 * scale_factor)
        
    if new_w2 < min_allowed_size:
        scale_factor = min_allowed_size / new_w2
        new_w2 = min_allowed_size
        new_h2 = int(new_h2 * scale_factor)
        
    if new_h2 < min_allowed_size:
        scale_factor = min_allowed_size / new_h2
        new_h2 = min_allowed_size
        new_w2 = int(new_w2 * scale_factor)

    # Resize images using LANCZOS resampling
    img1_scaled = img1.resize((new_w1, new_h1), Image.Resampling.LANCZOS)
    img2_scaled = img2.resize((new_w2, new_h2), Image.Resampling.LANCZOS)
    
    return img1_scaled, img2_scaled


def calculate_image_similarities(original1, original2):
    """
    Calculates both CNN-based similarity and SSIM score for image pairs.
    
    Parameters:
        original1 (PIL.Image): First original image for SSIM comparison
        original2 (PIL.Image): Second original image for SSIM comparison
        originalImg1 (PIL.Image): First original image for CNN comparison
        originalImg2 (PIL.Image): Second original image for CNN comparison
        model: Neural network model for feature extraction
        
    Returns:
        tuple: (similarity_score, similarity_ssim_score)
            - similarity_score (float): CNN-based similarity score [0..1]
            - similarity_ssim_score (float): SSIM-based similarity score [0..1]
    """
    # Prepare images for CNN feature extraction
    imgFeatures1 = resize_image(original1, target_size=(1000, 1000))
    imgFeatures2 = resize_image(original2, target_size=(1000, 1000))

    # Extract features using the model
    features1 = model.predict(imgFeatures1)
    features2 = model.predict(imgFeatures2)

    # Calculate cosine similarity
    similarity_score = 1 - cosine(features1.flatten(), features2.flatten())
    
    # Calculate SSIM score
    similarity_ssim_score = compute_image_similarity(original1, original2)
    
    return similarity_score, similarity_ssim_score


def extract_text_from_image(image, target_width=1000):
    """
    Performs EasyOCR on the given PIL image, returns text ordered by position.
    Returns list of (text, bbox) tuples ordered top-to-bottom, left-to-right.
    """
    image = resize_for_ocr(image, target_width)
    # easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    result = easyocr_reader.readtext(np.array(image))
    
    # Filter results by confidence and get position info
    text_with_positions = []
    for bbox, text, confidence in result:
        if confidence > 0.5:
            # Calculate center point of bounding box
            center_y = sum(point[1] for point in bbox) / 4
            center_x = sum(point[0] for point in bbox) / 4
            
            text_with_positions.append({
                'text': text,
                'bbox': bbox,
                'center': (center_x, center_y)
            })
    
    # Sort first by y-position (top to bottom) with tolerance
    y_tolerance = 20  # Pixels tolerance for same line
    sorted_by_y = sorted(text_with_positions, key=lambda x: x['center'][1])
    
    # Group text elements that are on the same line (within y_tolerance)
    lines = []
    current_line = []
    current_y = float('-inf')
    
    for text_elem in sorted_by_y:
        if abs(text_elem['center'][1] - current_y) > y_tolerance:
            # New line
            if current_line:
                # Sort current line by x position
                current_line.sort(key=lambda x: x['center'][0])
                lines.append(current_line)
            current_line = [text_elem]
            current_y = text_elem['center'][1]
        else:
            # Same line
            current_line.append(text_elem)
    
    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda x: x['center'][0])
        lines.append(current_line)
    
    # Combine text preserving order
    ordered_text = []
    for line in lines:
        line_text = []
        for text_elem in line:
            line_text.append(text_elem['text'])
        ordered_text.append(' '.join(line_text))
    
    return '\n'.join(ordered_text)
def resize_image_with_Blur(pil_img, target_size=(224, 224), blur_radius=2):
    blurred_image = pil_img.resize(target_size, Image.LANCZOS)
    blurred_image = blurred_image.filter(ImageFilter.BoxBlur(radius=blur_radius))
    img_array = image.img_to_array(blurred_image)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(expanded_img_array)
    
def calculate_blur_based_similarity(foreground1, foreground2):
    """
    Calculates similarity score between two images using blur preprocessing.
    
    Parameters:
        foreground1 (PIL.Image): First image to compare
        foreground2 (PIL.Image): Second image to compare
        model: Neural network model for feature extraction
        
    Returns:
        float: Similarity score based on blurred image comparison
    """
    print("similarity score is above 0.80 and below 0.95 => extra blur step")
    
    # Apply blur and resize to both images
    blur1 = resize_image_with_Blur(foreground1, target_size=(1000, 1000))
    blur2 = resize_image_with_Blur(foreground2, target_size=(1000, 1000))

    # Extract features using the model
    featBlur1 = model.predict(blur1)
    featBlur2 = model.predict(blur2)

    # Calculate cosine similarity
    similarity_score_forground = 1 - cosine(featBlur1.flatten(), featBlur2.flatten())
    print("similarity_score blurred:", round(similarity_score_forground, 2))
    
    return similarity_score_forground

@app.route('/health', methods=['GET'])
def health_check():
    return 'healthy', 200

@app.route('/ocr_similarity', methods=['POST'])
def ocr_similarity():
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🔍 OCR SIMILARITY REQUEST STARTED")
    print("="*80)
    
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({'error': 'Missing images'}), 400

    img1 = request.files['img1']
    img2 = request.files['img2']
    
    # Get OCR engine from request (default to pytesseract)
    ocr_engine = request.form.get('ocr_engine', 'pytesseract').lower()
    
    print(f"📋 Selected OCR Engine: {ocr_engine.upper()}")
    print(f"📷 Image 1: {img1.filename}")
    print(f"📷 Image 2: {img2.filename}")
    
    if ocr_engine not in ['pytesseract', 'easyocr', 'paddleocr']:
        return jsonify({'error': 'Invalid OCR engine. Choose from: pytesseract, easyocr, paddleocr'}), 400

    print(f"⏱️  Loading images into memory...")
    Image1 = Image.open(io.BytesIO(img1.read()))
    Image2 = Image.open(io.BytesIO(img2.read()))
    print(f"✅ Images loaded - Image1: {Image1.size}, Image2: {Image2.size}")
    
    # Start OCR timing
    ocr_start_time = time.time()
    
    # Perform OCR based on selected engine
    if ocr_engine == 'pytesseract':
        print("\n🔤 PYTESSERACT ENGINE PROCESSING:")
        print("├── Step 1: Converting PIL images to grayscale internally")
        print("├── Step 2: Page segmentation analysis")
        print("├── Step 3: Character recognition using LSTM")
        print("├── Step 4: Applying language model corrections")
        
        print("├── Processing Image 1...")
        text1 = pytesseract.image_to_string(Image1)
        print("├── Processing Image 2...")
        text2 = pytesseract.image_to_string(Image2)
        
        print("├── Step 5: Cleaning text (removing form feeds, extra spaces)")
        text1 = ' '.join(text1.replace('\f', ' ').split())
        text2 = ' '.join(text2.replace('\f', ' ').split())
        print("└── ✅ Pytesseract processing complete")
        
    elif ocr_engine == 'easyocr':
        print("\n🧠 EASYOCR ENGINE PROCESSING:")
        print("├── Step 1: Converting PIL images to NumPy arrays")
        Image1 = resize_for_ocr(Image1, 700)
        Image2 = resize_for_ocr(Image2, 700)

        img1_array = np.array(Image1)
        img2_array = np.array(Image2)
        print(f"├── Array shapes - Image1: {img1_array.shape}, Image2: {img2_array.shape}")
        
        print("├── Step 2: Initializing EasyOCR Reader (CRAFT + CRNN models)")
        print("├── Parameters: languages=['en'], gpu=False, verbose=False")
        # easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        print("├── Step 3: CRAFT text detection (finding text regions)")
        print("├── Step 4: CRNN text recognition (reading characters)")
        print("├── Step 5: Confidence filtering (threshold: 0.5)")

        print("├── Processing Image 1...")
        result1 = easyocr_reader.readtext(img1_array)
        print(f"├── Found {len(result1)} text regions in Image 1")
        
        print("├── Processing Image 2...")
        result2 = easyocr_reader.readtext(img2_array)
        print(f"├── Found {len(result2)} text regions in Image 2")

        print("├── Step 6: Extracting high-confidence text")
        text1 = ' '.join([text for (bbox, text, confidence) in result1 if confidence > 0.5])
        text2 = ' '.join([text for (bbox, text, confidence) in result2 if confidence > 0.5])
        
        high_conf_1 = len([1 for (bbox, text, confidence) in result1 if confidence > 0.5])
        high_conf_2 = len([1 for (bbox, text, confidence) in result2 if confidence > 0.5])
        print(f"├── High confidence regions - Image1: {high_conf_1}, Image2: {high_conf_2}")
        
       
    elif ocr_engine == 'paddleocr':
        print("\n🚀 PADDLEOCR ENGINE PROCESSING:")
        print("├── Step 1: Converting PIL images to NumPy arrays")
        img1_array = np.array(Image1)
        img2_array = np.array(Image2)
        print(f"├── Array shapes - Image1: {img1_array.shape}, Image2: {img2_array.shape}")

        print("├── Step 2: Initializing PaddleOCR (DB detector + SVTR recognizer)")
        print("├── Parameters: use_angle_cls=True, lang=en, use_gpu=False")
        paddle_reader = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, cpu_threads=4, enable_mkldnn=False)
        
        try:
            print("├── Step 3: Text detection using DB (Differentiable Binarization)")
            print("├── Step 4: Text angle classification")
            print("├── Step 5: Text recognition using SVTR/CRNN")
            
            print("├── Processing Image 1...")
            result1 = paddle_reader.ocr(img1_array, cls=True)
            
            print("├── Processing Image 2...")
            result2 = paddle_reader.ocr(img2_array, cls=True)
            
            print("├── Step 6: Confidence filtering (threshold: 0.5)")
            text1 = ' '.join([line[1][0] for line in result1[0] if line[1][1] > 0.5]) if result1[0] else ""
            text2 = ' '.join([line[1][0] for line in result2[0] if line[1][1] > 0.5]) if result2[0] else ""
            
            regions_1 = len(result1[0]) if result1[0] else 0
            regions_2 = len(result2[0]) if result2[0] else 0
            high_conf_1 = len([1 for line in result1[0] if line[1][1] > 0.5]) if result1[0] else 0
            high_conf_2 = len([1 for line in result2[0] if line[1][1] > 0.5]) if result2[0] else 0
            
            print(f"├── Total regions - Image1: {regions_1}, Image2: {regions_2}")
            print(f"├── High confidence regions - Image1: {high_conf_1}, Image2: {high_conf_2}")
            
        finally:
            print("├── Step 7: Cleaning up PaddleOCR resources")
            print("├── Deleting paddle_reader instance")
            print("├── Deleting NumPy arrays")
            print("├── Running garbage collection")
            del paddle_reader
            del img1_array, img2_array
            gc.collect()
            print("└── ✅ PaddleOCR processing complete")
    
    # End OCR timing
    ocr_end_time = time.time()
    ocr_processing_time = ocr_end_time - ocr_start_time
    
    print(f"\n📊 EXTRACTED TEXT RESULTS:")
    print(f"├── Image 1 Text: '{text1[:100]}{'...' if len(text1) > 100 else ''}'")
    print(f"├── Image 2 Text: '{text2[:100]}{'...' if len(text2) > 100 else ''}'")
    print(f"├── Text 1 Length: {len(text1)} characters")
    print(f"└── Text 2 Length: {len(text2)} characters")
    
    print(f"\n🔄 CALCULATING SIMILARITY SCORES:")
    print("├── Using FuzzyWuzzy for token set ratio")
    print("├── Using FuzzyWuzzy for strict ratio")
    
    # Calculate similarity
    fuzzy_text_score = fuzz.token_set_ratio(text1, text2) / 100
    strict_text_score = fuzz.ratio(text1, text2) / 100

    # Calculate total processing time
    total_processing_time = time.time() - start_time
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"├── Fuzzy Text Similarity: {fuzzy_text_score:.3f}")
    print(f"├── Strict Text Similarity: {strict_text_score:.3f}")
    print(f"├── OCR Processing Time: {ocr_processing_time:.3f}s")
    print(f"└── Total Processing Time: {total_processing_time:.3f}s")
    print("="*80)

    return jsonify({
        'ocr_engine_used': ocr_engine,
        'img1Text': text1,
        'img2Text': text2,
        'fuzzy_text_similarity_score': fuzzy_text_score,
        'strict_text_similarity_score': strict_text_score,
        'ocr_processing_time_seconds': round(ocr_processing_time, 3),
        'total_processing_time_seconds': round(total_processing_time, 3)
    })



@app.route('/similarity', methods=['POST'])
def similarity():
    try:
        provided_api_key = request.args.get('api_key')
        api_key = os.getenv('API_KEY')
        if api_key and provided_api_key != api_key:
            return jsonify({'error': 'Invalid or missing API key'}), 401

        if 'img1' not in request.files or 'img2' not in request.files:
            return jsonify({'error': 'Missing images'}), 400

        img1 = request.files['img1']
        img2 = request.files['img2']
        similarity_ssim_score = None
        similarity_seq = None
        try:
            Image1 = Image.open(io.BytesIO(img1.read()))
            Image2 = Image.open(io.BytesIO(img2.read()))

          
            # Convert RGBA to RGB if present
            print("├── Step 1: Converting if present PNG or transparent backgrounds to white background")

            Image1 = convert_transparency_to_white(Image1)
            Image2 = convert_transparency_to_white(Image2)
           
            # Extract "foreground" (optional)
            print("├── Step 2: Extract foreground Object and scale with aspect ratio")
            Image1Foreground = extract_foreground(Image1)
            Image2Foreground = extract_foreground(Image2)
            Image1Foreground, Image2Foreground = scale_with_aspect_ratio(Image1Foreground, Image2Foreground)

        except IOError:
            return jsonify({'error': 'Invalid image format'}), 400

        # SSIM only
        # SSIM is very senstive to size and alignments, which wont work on e-commerce images
        # so maybe a deep learning based approach is better suited, but we need it to be efficient so we use EfficientNetB0 family
        # ok what about text similarity using OCR?

        print("├── Step 3: perform similarity calculations")

        # ============ CNN Embedding Similarity (EfficientNet) ============
        similarity_score, similarity_ssim_score = calculate_image_similarities(
                                                Image1Foreground, 
                                                Image2Foreground
                                             
        )

        print("├── Step 4: Check for low resolution images and apply blur-based similarity if needed")

        Lowerquality = False
        if 0.80 < similarity_score < 0.95 :
            print("├── Step 4.1: Low resolution detected, applying blur-based similarity")
            Lowerquality = True
            similarity_score = calculate_blur_based_similarity(
                                    Image1Foreground,
                                    Image2Foreground,
                                )
          
            print("similarity_score: after blur step", similarity_score)

        # similarity_ssim_score = compute_image_similarity(Image1, Image2)
        # print("similarity_ssim_score:", similarity_ssim_score)
        # response = {
        #     'ssim_index_score': similarity_ssim_score,          # SSIM result
        # }
        print("├── Step 5: Check for textual similarity if image similarity is high enough")

        image_similarity_score = similarity_score    
        fuzzy_text_score  = None
        strict_text_score = None
        if similarity_score >  0.97:
            print("├── Step 5.1: High image similarity detected, performing OCR-based text extraction and similarity")    
            text1  = extract_text_from_image(Image1, 700)
            text2 =  extract_text_from_image(Image2, 700)
       
            text1, text2 = normalize_text(text1, True), normalize_text(text2, True)

            if  text1 is  None and text2 is  None:
                print("No text found in either image.")
            else:
                print("Extracted Text 1:", text1)
                print("Extracted Text 2:", text2)
        
                strict_text_score = SequenceMatcher(None, text1, text2).ratio()
                fuzzy_text_score = fuzz.token_set_ratio(text1, text2) / 100
    
                print("strict_text_score:", strict_text_score)
                print("fuzzy_text_score:", fuzzy_text_score)
                similarity_score = (similarity_score * 0.5 + strict_text_score * 0.5)

        response = {
            'image_similarity_score': image_similarity_score,   # CNN-based
            'strict_text_similarity_score': strict_text_score,  # OCR-based Strict textual similarity
            'fuzzy_text_similarity_score': fuzzy_text_score,    # OCR-based Fuzzy textual similarity
            'ssim_index_score': similarity_ssim_score,          # SSIM result
            'similarity_score': similarity_score,               # Possibly updated
            }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True, use_reloader=True, reloader_type='stat')
