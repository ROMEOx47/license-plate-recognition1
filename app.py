from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pytesseract

app = Flask(__name__)

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    # Get the uploaded image file
    image = request.files['image']

    # Check if an image file was uploaded
    if image:
        try:
            # Read the uploaded image
            img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to the image
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Perform edge detection
            edges = cv2.Canny(blur, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by area (Assuming the license plate will be one of the largest contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            # Iterate over the contours to find the license plate
            for contour in contours:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                crop_img = img[min(box[:, 1]):max(box[:, 1]), min(box[:, 0]):max(box[:, 0])]
                gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

                # Use pytesseract to extract text from the image
                text = pytesseract.image_to_string(gray_crop)
                if text:
                    return jsonify({"result": text})

            # If no license plate is found
            return jsonify({"result": "No license plate found in the image."})

        except Exception as e:
            return jsonify({"error": str(e)})

    else:
        return jsonify({"error": "No image file uploaded."})


if __name__ == '__main__':
    app.run()
