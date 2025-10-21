import requests
import datetime
import os
from io import BytesIO
import time
from PIL import Image


# Define your image pairs using the images in the /images folder
image_pairs = [
    {
        'ref_image_path': 'TestImages/ref_53.jpg',
        'uploaded_image_path': 'TestImages/upload_53.jpg',
        'expected_result': 'Different',
        'threshold': 0.95
    },
    {
        'ref_image_path': 'TestImages/ref_82.jpg',
        'uploaded_image_path': 'TestImages/upload_82.jpg',
        'expected_result': 'Different',
        'threshold':  0.95
    },
    {
        'ref_image_path': 'TestImages/ref_137.jpg',
        'uploaded_image_path': 'TestImages/upload_137.jpg',
        'expected_result': 'Different',
        'threshold':  0.95
    },
    {
        'ref_image_path': 'TestImages/e451.jpeg',
        'uploaded_image_path': 'TestImages/e452.jpeg',
        'expected_result': 'Different',
        'threshold':  0.95
    }
]

# API configuration
api_url = os.getenv('API_URL', 'http://127.0.0.1:5000/similarity')
# api_url = os.getenv('API_URL', 'ourstagingendpoint.com/similarity')
# api_url = os.getenv('API_URL', 'http://productionendpoint.com/similarity')

# Test tracking
passed_tests = []
failed_tests = []

def read_local_image(image_path):
    """Read a local image file and return its content"""
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Error: Image file not found: {image_path}")
        return None

print("🚀 Starting Image Similarity Tests")
print("=" * 50)

for i, pair in enumerate(image_pairs):
    try:
        ref_image_path = pair['ref_image_path']
        uploaded_image_path = pair['uploaded_image_path']
        expected_result = pair['expected_result']
        threshold = pair['threshold']

        print(f"\n[{i+1}] Testing: {os.path.basename(ref_image_path)} vs {os.path.basename(uploaded_image_path)}")

        # Read local images
        img1 = read_local_image(ref_image_path)
        img2 = read_local_image(uploaded_image_path)

        if img1 is None or img2 is None:
            print(f"❌ FAILED - Could not load images")
            failed_tests.append({
                'pair': pair,
                'reason': 'Image loading failed',
                'expected': expected_result,
                'actual': 'N/A',
                'score': 'N/A'
            })
            continue

        # Send request to Flask API
        try:
            response = requests.post(api_url, files={'img1': BytesIO(img1), 'img2': BytesIO(img2)}, timeout=60)

            if response.status_code == 200:
                data = response.json()
                similarity_score = data.get("similarity_score", 0)
                actual_result = "Equal" if similarity_score >= threshold else "Different"

                # Check if result matches expectation
                if actual_result == expected_result:
                    print(f"✅ PASSED - Expected: {expected_result}, Got: {actual_result} (Score: {similarity_score:.3f})")
                    passed_tests.append({
                        'pair': pair,
                        'expected': expected_result,
                        'actual': actual_result,
                        'score': similarity_score
                    })
                else:
                    print(f"❌ FAILED - Expected: {expected_result}, Got: {actual_result} (Score: {similarity_score:.3f})")
                    failed_tests.append({
                        'pair': pair,
                        'reason': 'Result mismatch',
                        'expected': expected_result,
                        'actual': actual_result,
                        'score': similarity_score
                    })

            else:
                print(f"❌ FAILED - API Error: {response.status_code}")
                failed_tests.append({
                    'pair': pair,
                    'reason': f'API Error {response.status_code}',
                    'expected': expected_result,
                    'actual': 'N/A',
                    'score': 'N/A'
                })

        except requests.exceptions.ConnectionError as e:
            print(f"❌ FAILED - Connection error: {e}")
            failed_tests.append({
                'pair': pair,
                'reason': 'Connection error',
                'expected': expected_result,
                'actual': 'N/A',
                'score': 'N/A'
            })

    except Exception as e:
        print(f"❌ FAILED - Unexpected error: {e}")
        failed_tests.append({
            'pair': pair,
            'reason': f'Unexpected error: {e}',
            'expected': expected_result,
            'actual': 'N/A',
            'score': 'N/A'
        })

# Print final results
print("\n" + "=" * 50)
print("📊 TEST RESULTS SUMMARY")
print("=" * 50)

total_tests = len(passed_tests) + len(failed_tests)
print(f"Total Tests: {total_tests}")
print(f"✅ Passed: {len(passed_tests)} ({len(passed_tests)/total_tests*100:.1f}%)")
print(f"❌ Failed: {len(failed_tests)} ({len(failed_tests)/total_tests*100:.1f}%)")

if passed_tests:
    print("\n🎉 PASSED TESTS:")
    for i, test in enumerate(passed_tests, 1):
        ref_name = os.path.basename(test['pair']['ref_image_path'])
        upload_name = os.path.basename(test['pair']['uploaded_image_path'])
        print(f"  {i}. {ref_name} vs {upload_name} - Score: {test['score']:.3f}")

if failed_tests:
    print("\n💥 FAILED TESTS:")
    for i, test in enumerate(failed_tests, 1):
        ref_name = os.path.basename(test['pair']['ref_image_path'])
        upload_name = os.path.basename(test['pair']['uploaded_image_path'])
        print(f"  {i}. {ref_name} vs {upload_name}")
        print(f"     Expected: {test['expected']}, Got: {test['actual']}")
        print(f"     Reason: {test['reason']}")
        if test['score'] != 'N/A':
            print(f"     Score: {test['score']:.3f}")

print("\n🏁 Testing complete!")