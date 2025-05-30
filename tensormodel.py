from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io

app = Flask(__name__)
CORS(app, resources={r"/search": {"origins": "http://127.0.0.1:5500"}})

# Initialize the database
db = Database()

# Load the pre-trained ResNet50 model for feature extraction
model = ResNet50(weights="imagenet")

def extract_image_features(img):
    """Extract features from the image using a pre-trained ResNet50 model"""
    img = img.resize((224, 224))  # Resize image to fit ResNet50 input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()  # Flatten the features into a 1D array

def get_product_details_from_description(description_input):
    """Fetch the most relevant product based on description using cosine similarity"""
    try:
        query = """
        SELECT product_detail.Product_id, 
               product_detail.product_name, 
               product_description.product_description, 
               product_detail.product_price, 
               product_detail.product_size, 
               product_detail.product_image, 
               product_detail.product_link
        FROM product_detail
        JOIN product_description ON product_detail.Product_link = product_description.Product_link
        """
        db.cursor.execute(query)
        products = db.cursor.fetchall()

        if not products:
            return {"error": "No products found in the database"}

        # Extract product descriptions for comparison
        product_descriptions = [product[2] for product in products]
        product_ids = [product[0] for product in products]
        product_names = [product[1] for product in products]
        product_prices = [product[3] for product in products]
        product_sizes = [product[4] for product in products]
        product_images = [product[5] for product in products]
        product_links = [product[6] for product in products]

        # Use TfidfVectorizer for description comparison
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(product_descriptions + [description_input])
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

        most_similar_idx = cosine_similarities.argmax()
        
        # Prepare the result with matched product details
        result = {
            "product_id": product_ids[most_similar_idx],
            "product_name": product_names[most_similar_idx],
            "product_description": product_descriptions[most_similar_idx],
            "product_price": product_prices[most_similar_idx],
            "product_size": product_sizes[most_similar_idx],
            "product_image": product_images[most_similar_idx],
            "product_link": product_links[most_similar_idx]
        }

        return result

    except mysql.connector.Error as e:
        return {"error": f"Error retrieving products from the database: {e}"}

@app.route('/search', methods=['POST'])
def search_product():
    """API endpoint to receive both image and description, and return matched product details"""
    try:
        # Extract image and description from the request
        image_file = request.files.get('image')
        description_input = request.form.get('description')

        if not image_file or not description_input:
            return jsonify({"error": "Image or description missing"}), 400

        # Process image
        img = Image.open(io.BytesIO(image_file.read()))
        image_features = extract_image_features(img)

        # Process description
        product_details_from_description = get_product_details_from_description(description_input)

        # For simplicity, we'll return both results here; adjust based on your needs
        result = {
            "image_features": image_features.tolist(),  # Send image features as a list (for simplicity)
            "description_match": product_details_from_description
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
