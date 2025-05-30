from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from PIL import Image
from databaseClass import Database
import mysql.connector
import os
import cv2
import logging

# Suppress TensorFlow logs (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/search": {"origins": "http://127.0.0.1:5500"}})

# Initialize the database
db = Database()

# Load the CLIP model and processor for image feature extraction
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define a list of valid colors and their synonyms with approximate HSV ranges
color_map = {
    'red': ['red', 'maroon', 'crimson', 'scarlet'],
    'green': ['green', 'olive', 'army-green', 'emerald'],
    'black': ['black'],
    'white': ['white'],
    'purple': ['purple', 'dark purple', 'violet'],
    'teal': ['teal'],
    'beige': ['beige'],
    'yellow': ['yellow', 'gold'],
    'blue': ['blue', 'dark blue', 'ice blue', 'navy'],
    'pink': ['pink', 'baby pink', 'light pink'],
    'grey': ['grey', 'mint gray', 'gray'],
    'orange': ['orange']
}

def get_dominant_color(image):
    """
    Detect the dominant color of the image using HSV color space.
    """
    try:
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Convert to HSV color space
        image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        # Reshape to a list of pixels
        pixels = image_hsv.reshape(-1, 3)
        # Calculate the dominant color using k-means clustering (k=1 for dominant color)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        _, labels, centers = cv2.kmeans(np.float32(pixels), 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_hsv = centers[0].astype(int)
        h, s, v = dominant_hsv
        logger.debug(f"Dominant color HSV: {h}, {s}, {v}")

        # Define HSV ranges for colors (H: 0-179, S: 0-255, V: 0-255)
        if v < 50 and s < 50:  # Low saturation and value indicate black
            return 'black'
        elif v > 200 and s < 50:  # High value and low saturation indicate white
            return 'white'
        elif 0 <= h <= 10 or 160 <= h <= 179:  # Red range (considering the wraparound)
            return 'red'
        elif 11 <= h <= 25:  # Orange range
            return 'orange'
        elif 26 <= h <= 45:  # Yellow range
            return 'yellow'
        elif 46 <= h <= 65:  # Green range
            return 'green'
        elif 66 <= h <= 85:  # Teal range
            return 'teal'
        elif 86 <= h <= 105:  # Blue range
            return 'blue'
        elif 106 <= h <= 135:  # Purple range
            return 'purple'
        elif 136 <= h <= 165:  # Pink range
            return 'pink'
        elif 0 <= s < 50 and 50 <= v <= 200:  # Low saturation with moderate value indicates grey/beige
            return 'grey' if v < 150 else 'beige'
        else:
            return None  # Default if no clear match
    except Exception as e:
        logger.error(f"Error in get_dominant_color: {e}")
        return None

def get_product_details_from_image(image_file, num_results=5):
    """
    Function to get product details from the database by comparing an input image with product descriptions,
    with color filtering.
    """
    try:
        # Open and process the uploaded image for color detection
        image = Image.open(image_file).convert("RGB")
        dominant_color = get_dominant_color(image)
        logger.debug(f"Dominant color detected: {dominant_color}")

        # Fetch products from the database, filtering by dominant color if detected
        query = """
        SELECT product_detail.Product_id, 
               product_detail.product_name, 
               product_description.product_description, 
               product_detail.product_price, 
               product_detail.product_size, 
               product_detail.product_image, 
               product_detail.product_link,
               product_detail.color
        FROM product_detail
        JOIN product_description ON product_detail.Product_link = product_description.Product_link
        """
        if dominant_color:
            # Filter by color synonyms
            color_synonyms = color_map.get(dominant_color.lower(), [])
            placeholders = ', '.join(['%s'] * len(color_synonyms))
            query += f" WHERE product_detail.color IN ({placeholders})"
            db.cursor.execute(query, color_synonyms)
        else:
            db.cursor.execute(query)

        products = db.cursor.fetchall()

        # Handle case where no products exist in the database
        if not products:
            logger.error("No products found in the database")
            return {"error": "No products found in the database"}

        # Extract product descriptions and details
        product_descriptions = [product[2] for product in products]
        product_details = [product for product in products]

        # Preprocess the image
        image_input = processor(images=image, return_tensors="pt")

        # Extract features from the image using the CLIP model
        with torch.no_grad():
            image_features = model.get_image_features(**image_input).numpy()

        # Preprocess the text with truncation enabled
        inputs = processor(
            text=product_descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True  # Truncate descriptions longer than 77 tokens
        )
        with torch.no_grad():
            text_features = model.get_text_features(**inputs).numpy()

        # Compute cosine similarity between the image and text embeddings
        similarities = np.dot(image_features, text_features.T) / (
            np.linalg.norm(image_features) * np.linalg.norm(text_features, axis=1)
        )

        # Get indices of the most similar products
        sorted_idx = similarities.argsort()[0][-num_results:][::-1]

        # Prepare the response with product details
        result = []
        for idx in sorted_idx:
            product = product_details[idx]
            result.append({
                "product_id": product[0],
                "product_name": product[1],
                "product_description": product[2],
                "product_price": product[3],
                "product_size": product[4],
                "product_image": product[5],
                "product_link": product[6],
                "color": product[7],  # Include color for debugging
                "similarity_score": float(similarities[0][idx])  # Add similarity score for transparency
            })
        logger.debug(f"Returning product details: {result}")
        return result
    except mysql.connector.Error as e:
        logger.error(f"Database error: {e}")
        return {"error": f"Database error: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}"}

@app.route('/search', methods=['POST'])
def search_product():
    """
    API endpoint to receive an input image and return matched product details.
    """
    try:
        logger.debug("Request received!")

        # Validate and fetch the uploaded image
        if 'image' not in request.files or request.files['image'].filename == '':
            return jsonify({"error": "No valid image file provided"}), 400

        image_file = request.files['image']
        logger.debug(f"Received image file: {image_file.filename}")

        # Get product details based on the uploaded image
        product_details = get_product_details_from_image(image_file)

        # Check for errors in the result
        if 'error' in product_details:
            logger.error(f"Error in product details: {product_details['error']}")
            return jsonify(product_details), 500

        logger.debug(f"Returning product details: {product_details}")
        return jsonify(product_details), 200
    except Exception as e:
        # Log any unexpected errors
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
