import cv2
import numpy as np
import os
import uuid
from flask import Flask, render_template, request, send_file, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def convert_to_sketch(image_path):
    # Load the image and convert to grayscale
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image to prepare for sketch effect
    inverted_image = cv2.bitwise_not(gray_image)
    
    # Apply a Gaussian blur to the inverted image to simulate softness
    kernel_size = (35, 35) if min(image.shape[:2]) > 1000 else (21, 21)
    blurred = cv2.GaussianBlur(inverted_image, kernel_size, sigmaX=0, sigmaY=0)
    
    # Invert the blurred image and create the base sketch
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch_base = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    
    # Apply adaptive thresholding to add detail and texture
    detailed_sketch = cv2.adaptiveThreshold(sketch_base, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, blockSize=9, C=2)
    
    # Enhance edges with Canny edge detection and blend with the detailed sketch
    edges = cv2.Canny(gray_image, 50, 150)
    final_sketch = cv2.addWeighted(detailed_sketch, 0.8, edges, 0.2, 0)
    
    # Save the sketch with a unique filename to prevent overwriting
    sketch_filename = f"{uuid.uuid4().hex}_sketch.png"
    sketch_path = os.path.join(UPLOAD_FOLDER, sketch_filename)
    cv2.imwrite(sketch_path, final_sketch)
    
    return sketch_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)
            
            # Convert the uploaded image to a sketch
            sketch_path = convert_to_sketch(image_path)
            
            # Redirect to a result page to show the sketch preview
            return redirect(url_for('show_sketch', filename=os.path.basename(sketch_path)))
    
    return render_template('index.html')

@app.route('/sketch/<filename>')
def show_sketch(filename):
    sketch_url = url_for('static', filename=f'uploads/{filename}')
    return render_template('sketch_result.html', sketch_url=sketch_url, filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
