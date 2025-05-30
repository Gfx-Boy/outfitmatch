from flask import Flask, request, jsonify, render_template, redirect, session, url_for, send_from_directory, flash
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
from flask_mysqldb import MySQL
import os
import re
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import logging
import traceback
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define the upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# MySQL configuration
app.config['MYSQL_HOST'] = '193.203.166.181'
app.config['MYSQL_USER'] = 'u557851335_cloth_db'
app.config['MYSQL_PASSWORD'] = 'Newton 12'  # Update with your MySQL password
app.config['MYSQL_DB'] = 'u557851335_cloth_design_d'
app.config['SECRET_KEY'] = '123'  # Change to a secure key in production

# Initialize MySQL
mysql = MySQL(app)
# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define a list of valid colors and their synonyms with approximate HSV ranges
color_map = {
    'red': ['red', 'maroon', 'crimson', 'scarlet', 'burgundy', 'ruby', 'wine', 'cherry', 'garnet'],
    'green': ['green', 'olive', 'army-green', 'emerald', 'mint', 'lime', 'sage', 'forest', 'moss'],
    'black': ['black', 'jet', 'ebony', 'charcoal'],
    'white': ['white', 'ivory', 'cream', 'snow', 'pearl'],
    'purple': ['purple', 'dark purple', 'violet', 'lavender', 'mauve', 'plum', 'orchid'],
    'teal': ['teal', 'aqua', 'cyan', 'turquoise'],
    'beige': ['beige', 'tan', 'camel', 'sand', 'khaki'],
    'yellow': ['yellow', 'gold', 'mustard', 'lemon', 'amber'],
    'blue': ['blue', 'dark blue', 'ice blue', 'navy', 'sky', 'azure', 'cobalt', 'sapphire', 'denim'],
    'pink': ['pink', 'baby pink', 'light pink', 'rose', 'fuchsia', 'magenta', 'blush', 'coral'],
    'grey': ['grey', 'mint gray', 'gray', 'slate', 'ash', 'silver'],
    'orange': ['orange', 'peach', 'apricot', 'rust', 'tangerine', 'coral'],
    'brown': ['brown', 'chocolate', 'coffee', 'mocha', 'walnut', 'hazel', 'taupe', 'umber'],
    'cream': ['cream', 'off-white', 'eggshell'],
    'silver': ['silver', 'metallic'],
    'gold': ['gold', 'golden', 'champagne'],
}

# Reverse mapping: color synonym -> primary color
color_lookup = {}
for primary_color, synonyms in color_map.items():
    for synonym in synonyms:
        color_lookup[synonym] = primary_color

type_list = [
    # Pakistani/Middle Eastern dress types and common variations
    'embroidered', 'organza', 'lawn', 'khaddar', 'caftan', 'suit', 'shalwar',
    'kameez', 'dupatta', 'shirt', 'trouser', 'pakistani', 'marina', 'saree',
    'kurta', 'kurti', 'chiffon', 'cotton', 'silk', 'velvet',
    'abaya', 'kaftan', 'maxi', 'lehenga', 'anarkali', 'frock', 'gown',
    'shalwar kameez', 'salwar', 'salwar kameez', 'hijab', 'burqa', 'jilbab',
    'pishwas', 'angrakha', 'sari', 'saree', 'peshwas', 'cape', 'cape dress',
    'palazzo', 'palazzo pants', 'tunic', 'tunic dress', 'long shirt', 'long dress',
    'middle eastern', 'arabic', 'pakistani dress', 'pakistani suit', 'pakistani gown',
    'pakistani frock', 'pakistani kurta', 'pakistani kameez', 'pakistani shalwar',
    'pakistani dupatta', 'pakistani saree', 'pakistani abaya', 'pakistani kaftan',
    'pakistani maxi', 'pakistani lehenga', 'pakistani anarkali', 'pakistani pishwas',
    'pakistani angrakha', 'pakistani sari', 'pakistani peshwas', 'pakistani cape',
    'pakistani palazzo', 'pakistani tunic', 'pakistani long shirt', 'pakistani long dress',
    'middle eastern dress', 'middle eastern suit', 'middle eastern gown',
    'middle eastern frock', 'middle eastern kurta', 'middle eastern kameez',
    'middle eastern shalwar', 'middle eastern dupatta', 'middle eastern saree',
    'middle eastern abaya', 'middle eastern kaftan', 'middle eastern maxi',
    'middle eastern lehenga', 'middle eastern anarkali', 'middle eastern pishwas',
    'middle eastern angrakha', 'middle eastern sari', 'middle eastern peshwas',
    'middle eastern cape', 'middle eastern palazzo', 'middle eastern tunic',
    'middle eastern long shirt', 'middle eastern long dress'
]

def get_dominant_color(image):
    try:
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Resize image to reduce noise and computation
        image_cv = cv2.resize(image_cv, (100, 100), interpolation=cv2.INTER_AREA)

        # Convert to HSV color space
        image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

        # Compute overall mean value and saturation to check for black/white
        v_mean = np.mean(image_hsv[:, :, 2])
        s_mean = np.mean(image_hsv[:, :, 1])
        logger.debug(f"Overall HSV - Value: {v_mean}, Saturation: {s_mean}")

        # Check for black or white without masking initially
        if v_mean < 30 and s_mean < 50:  # Low value and saturation indicate black
            return 'black'
        elif v_mean > 180 and s_mean < 50:  # High value and low saturation indicate white
            return 'white'

        # Create a wider mask to include more colored pixels
        mask = cv2.inRange(image_hsv, np.array([0, 5, 5]), np.array([179, 255, 250]))
        logger.debug(f"Mask pixel count: {np.sum(mask > 0)}")
        # Apply the mask to the HSV image
        masked_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

        # Compute histogram of the hue channel (0-179 in OpenCV)
        hist = cv2.calcHist([masked_hsv], [0], mask, [180], [0, 180])
        # Normalize histogram to avoid zero division
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros_like(hist)
        # Find the hue with the highest frequency
        dominant_hue = np.argmax(hist)
        # Get average saturation and value for the dominant hue
        hue_mask = cv2.inRange(image_hsv, np.array([dominant_hue-5, 5, 5]), np.array([dominant_hue+5, 255, 250]))
        s_mean_hue = np.mean(image_hsv[:, :, 1][hue_mask > 0]) if np.sum(hue_mask) > 0 else 0
        v_mean_hue = np.mean(image_hsv[:, :, 2][hue_mask > 0]) if np.sum(hue_mask) > 0 else 0

        logger.debug(f"Dominant hue: {dominant_hue}, Saturation: {s_mean_hue}, Value: {v_mean_hue}")

        # Define HSV-based color classification
        if s_mean_hue < 40 and 40 <= v_mean_hue <= 180:  # Low saturation with moderate value indicates grey/beige
            return 'grey' if v_mean_hue < 120 else 'beige'
        elif 0 <= dominant_hue <= 10 or 160 <= dominant_hue <= 179:  # Red range
            return 'red'
        elif 11 <= dominant_hue <= 25:  # Orange range
            return 'orange'
        elif 26 <= dominant_hue <= 40:  # Yellow range
            return 'yellow'
        elif 41 <= dominant_hue <= 65:  # Green range
            return 'green'
        elif 66 <= dominant_hue <= 85:  # Teal range
            return 'teal'
        elif 86 <= dominant_hue <= 105:  # Blue range
            return 'blue'
        elif 106 <= dominant_hue <= 135:  # Purple range
            return 'purple'
        elif 136 <= dominant_hue <= 159:  # Pink range
            return 'pink'
        else:
            return None  # Default if no clear match
    except Exception as e:
        logger.error(f"Error in get_dominant_color: {traceback.format_exc()}")
        # Fallback with full image analysis
        image_hsv_full = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        v_mean_full = np.mean(image_hsv_full[:, :, 2])
        s_mean_full = np.mean(image_hsv_full[:, :, 1])
        # Compute histogram without mask
        hist_full = cv2.calcHist([image_hsv_full], [0], None, [180], [0, 180])
        hist_full = hist_full / np.sum(hist_full) if np.sum(hist_full) > 0 else np.zeros_like(hist_full)
        dominant_hue_full = np.argmax(hist_full)
        hue_mask_full = cv2.inRange(image_hsv_full, np.array([dominant_hue_full-5, 5, 5]), np.array([dominant_hue_full+5, 255, 250]))
        s_mean_hue_full = np.mean(image_hsv_full[:, :, 1][hue_mask_full > 0]) if np.sum(hue_mask_full) > 0 else 0
        v_mean_hue_full = np.mean(image_hsv_full[:, :, 2][hue_mask_full > 0]) if np.sum(hue_mask_full) > 0 else 0

        logger.debug(f"Fallback - Dominant hue: {dominant_hue_full}, Saturation: {s_mean_hue_full}, Value: {v_mean_hue_full}")
        if v_mean_full < 30 and s_mean_full < 50:
            return 'black'
        elif v_mean_full > 180 and s_mean_full < 50 and (0 <= dominant_hue_full <= 10 or 160 <= dominant_hue_full <= 179):
            return 'white'
        elif s_mean_hue_full < 40 and 40 <= v_mean_hue_full <= 180:
            return 'grey' if v_mean_hue_full < 120 else 'beige'
        elif 0 <= dominant_hue_full <= 10 or 160 <= dominant_hue_full <= 179:
            return 'red'
        elif 11 <= dominant_hue_full <= 25:
            return 'orange'
        elif 26 <= dominant_hue_full <= 40:
            return 'yellow'
        elif 41 <= dominant_hue_full <= 65:
            return 'green'
        elif 66 <= dominant_hue_full <= 85:
            return 'teal'
        elif 86 <= dominant_hue_full <= 105:
            return 'blue'
        elif 106 <= dominant_hue_full <= 135:
            return 'purple'
        elif 136 <= dominant_hue_full <= 159:
            return 'pink'
        else:
            return None

def extract_keywords_from_image(image_path):
    """
    Calls Meta Llama Vision API to extract keywords from the uploaded image.
    """
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "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"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    import base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data = {
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe the clothing in this image. List the type, color, style, and any notable features as keywords, separated by commas."},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
            ]}
        ]
    }
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        # Extract keywords from the response
        text = result['choices'][0]['message']['content']
        # Split by commas and clean up
        keywords = [kw.strip().lower() for kw in text.split(',') if kw.strip()]
        # Use only the API's output for further processing (main color/type extraction happens later)
        return keywords
    except Exception as e:
        logger.error(f"Meta Llama Vision API error: {e}")
        return []

def extract_relevant_keywords(meta_llama_output, color_map):
    """
    Extracts and normalizes relevant keywords (colors, types, materials) from Meta Llama output.
    Returns a set of keywords for searching.
    Optimized for Pakistani/Middle Eastern dress types and robust color mapping.
    """
    import re
    color_lookup = {synonym: main for main, synonyms in color_map.items() for synonym in synonyms}
    text = ' '.join(meta_llama_output).lower()
    words = re.findall(r'\b\w+\b', text)
    keywords = set()
    for word in words:
        # Normalize color (robust for red and synonyms)
        if word in color_lookup:
            keywords.add(color_lookup[word])
        # Add all expanded dress types
        elif word in type_list:
            keywords.add(word)
    return list(keywords)

def extract_main_color_and_type(meta_llama_output, color_map, type_list):
    """
    Extracts the main color (normalized) and main product type from Meta Llama output.
    Returns a tuple: (color, type) or (None, None) if not found.
    Optimized for Pakistani/Middle Eastern dress types and robust color mapping.
    """
    import re
    color_lookup = {synonym: main for main, synonyms in color_map.items() for synonym in synonyms}
    text = ' '.join(meta_llama_output).lower()
    words = re.findall(r'\b\w+\b', text)
    found_color = None
    found_type = None
    for word in words:
        if not found_color and word in color_lookup:
            found_color = color_lookup[word]
        if not found_type and word in type_list:
            found_type = word
        if found_color and found_type:
            break
    return found_color, found_type

def search_products_by_keywords(keywords):
    """
    Searches the product_detail table for products matching any of the keywords in name, size, price, or also matches the product_description table for richer results.
    """
    if not keywords:
        return []
    cursor = mysql.connection.cursor()
    # Build a dynamic WHERE clause for flexible matching
    like_clauses = []
    params = []
    for kw in keywords:
        like_clauses.append("(product_detail.Product_name LIKE %s OR product_detail.Product_size LIKE %s OR product_detail.Product_price LIKE %s OR product_description.Product_description LIKE %s)")
        params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%"])
    where_clause = " OR ".join(like_clauses)
    query = f"""
        SELECT product_detail.Product_name, product_detail.Product_price, product_detail.Product_size, product_detail.Product_link, product_detail.Product_image, product_description.Product_description
        FROM product_detail
        LEFT JOIN product_description ON product_detail.Product_link = product_description.Product_link
        WHERE {where_clause}
        LIMIT 20
    """
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    products = []
    for row in rows:
        products.append({
            "product_name": row[0],
            "product_price": row[1],
            "product_size": row[2],
            "product_link": row[3],
            "product_image": row[4],
            "product_description": row[5] if len(row) > 5 else None
        })
    return products

def search_products_by_color_and_type(color, type_, mysql):
    """
    Searches for products that match BOTH the main color and the main type in product_name or product_description.
    """
    if not color or not type_:
        return []
    cursor = mysql.connection.cursor()
    query = f"""
        SELECT product_detail.Product_name, product_detail.Product_price, product_detail.Product_size, product_detail.Product_link, product_detail.Product_image, product_description.Product_description
        FROM product_detail
        LEFT JOIN product_description ON product_detail.Product_link = product_description.Product_link
        WHERE (
            (product_detail.Product_name LIKE %s OR product_description.Product_description LIKE %s)
            AND
            (product_detail.Product_name LIKE %s OR product_description.Product_description LIKE %s)
        )
        LIMIT 20
    """
    params = [f"%{color}%", f"%{color}%", f"%{type_}%", f"%{type_}%"]
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    products = []
    for row in rows:
        products.append({
            "product_name": row[0],
            "product_price": row[1],
            "product_size": row[2],
            "product_link": row[3],
            "product_image": row[4],
            "product_description": row[5] if len(row) > 5 else None
        })
    return products

CORS(app)  # Allow CORS for all origins and all routes

@app.route('/search/', methods=['POST'])
def search_producttext():
    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")

        description_input = data.get('description', '')

        if not description_input:
            return jsonify({"error": "Description input is missing"}), 400

        logger.debug(f"Searching for product details matching: {description_input}")

        product_details = get_product_details_from_description(description_input)

        if 'error' in product_details:
            logger.error(f"Error in product search: {product_details['error']}")
            return jsonify(product_details), 500

        if 'loggedin' in session:
            user_id = session['id']
            cursor = mysql.connection.cursor()
            for product in product_details:
                query = """
                INSERT INTO search_history (user_id, search_query, product_name, product_description, regular_price, search_type)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                values = (
                    user_id,
                    description_input,
                    product['product_name'],
                    product['product_description'],
                    product['product_price'],
                    "text"
                )
                cursor.execute(query, values)
            mysql.connection.commit()
            cursor.close()
            logger.debug(f"Search history stored for user_id: {user_id} with query: {description_input}")

        logger.debug(f"Returning product details: {product_details}")
        return jsonify(product_details), 200
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        cursor.close()
        if account:
            session['loggedin'] = True
            session['id'] = account[0]
            session['username'] = account[1]
            return redirect(url_for('index'))
        else:
            flash("Incorrect username/password!", "danger")
    return render_template('auth/login.html', title="Login")

@app.route('/auth/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        account = cursor.fetchone()
        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email:
            flash("Please fill out the form!", "danger")
        else:
            cursor.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)', (username, email, password))
            mysql.connection.commit()
            flash("You have successfully registered!", "success")
            return redirect(url_for('login'))
        cursor.close()
    return render_template('auth/register.html', title="Register")

@app.route('/search_history')
def search_history():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    try:
        user_id = session['id']
        cursor = mysql.connection.cursor()
        query = """
        SELECT id, user_id, search_query, product_name, product_description, 
               regular_price, search_time, search_type, image_filename 
        FROM search_history 
        WHERE user_id = %s 
        ORDER BY search_time DESC
        """
        cursor.execute(query, (user_id,))
        history = cursor.fetchall()
        cursor.close()

        history_list = []
        for row in history:
            image_filename = row[8]
            history_list.append({
                'id': row[0],
                'user_id': row[1],
                'search_query': row[2],
                'product_name': row[3],
                'product_description': row[4],
                'regular_price': row[5],
                'search_time': row[6].strftime('%Y-%m-%d %H:%M:%S') if row[6] else None,
                'search_type': row[7],
                'image_filename': image_filename
            })
        return render_template('search_history2.html', history=history_list)
    except Exception as e:
        logger.error(f"Error in search_history: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/search/img', methods=['POST'], strict_slashes=False)
def search_product():
    try:
        if 'image' not in request.files or request.files['image'].filename == '':
            return jsonify({"error": "No valid image file provided"}), 400

        image_file = request.files['image']
        image_filename = image_file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)
        logger.debug(f"Image saved at: {image_path}")

        # 1. Always use Meta Llama API to extract keywords from image
        keywords = extract_keywords_from_image(image_path)
        logger.debug(f"Extracted keywords from Meta Llama API: {keywords}")
        if not keywords:
            # Check if the log contains a 429 error (rate limit)
            import traceback
            last_error = traceback.format_exc()
            if '429' in last_error or 'Too Many Requests' in last_error:
                return jsonify({"error": "Image analysis service is temporarily unavailable due to rate limiting. Please try again later."}), 503
            return jsonify({"error": "Image analysis service is temporarily unavailable due to rate limiting. Please try again later."}), 500

        # 2. Extract main color and type from Meta Llama API output
        main_color, main_type = extract_main_color_and_type(keywords, color_map, type_list)
        logger.debug(f"Main color: {main_color}, Main type: {main_type}")

        # 3. Only use the main color for searching (ignore type for color search)
        if main_color:
            cursor = mysql.connection.cursor()
            query = f"""
                SELECT product_detail.Product_name, product_detail.Product_price, product_detail.Product_size, product_detail.Product_link, product_detail.Product_image, product_description.Product_description
                FROM product_detail
                LEFT JOIN product_description ON product_detail.Product_link = product_description.Product_link
                WHERE (product_detail.Product_name LIKE %s OR product_description.Product_description LIKE %s)
                LIMIT 20
            """
            params = [f"%{main_color}%", f"%{main_color}%"]
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
            products = []
            for row in rows:
                products.append({
                    "product_name": row[0],
                    "product_price": row[1],
                    "product_size": row[2],
                    "product_link": row[3],
                    "product_image": row[4],
                    "product_description": row[5] if len(row) > 5 else None
                })
        else:
            products = []

        # 4. If no products found, return a message
        if not products:
            return jsonify({"message": "No product found"}), 200

        # 5. Log search in search_history for logged-in users
        search_source = 'database'
        if 'loggedin' in session:
            user_id = session['id']
            cursor = mysql.connection.cursor()
            for product in products:
                query = """
                INSERT INTO search_history (user_id, search_query, product_name, product_description, regular_price, search_type, image_filename)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    user_id,
                    ', '.join(keywords),
                    product.get('product_name', ''),
                    product.get('product_description', ''),
                    product.get('product_price', ''),
                    search_source,
                    image_filename
                )
                cursor.execute(query, values)
            mysql.connection.commit()
            cursor.close()
            logger.debug(f"Search history stored for user_id: {user_id} with image: {image_filename} and keywords: {keywords}")

        return jsonify(products), 200
    except Exception as e:
        logger.error(f"Error in search_product: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/products', methods=['GET'])
def get_products():
    try:
        # Optionally, you can add query params for filtering/searching
        cursor = mysql.connection.cursor()
        query = '''
            SELECT product_detail.Product_name, product_description.Product_description, product_detail.Product_price, product_detail.Product_link, product_detail.Product_image
            FROM product_detail
            JOIN product_description ON product_detail.Product_link = product_description.Product_link
            LIMIT 50
        '''
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        products = []
        for row in rows:
            products.append({
                "product_name": row[0],
                "product_description": row[1],
                "product_price": row[2],
                "product_link": row[3],
                "product_image": row[4]
            })
        return jsonify(products), 200
    except Exception as e:
        logger.error(f"Error in get_products: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

# --- TEST: Search products using Meta Llama API keywords ---
from databaseClass import Database

def test_search_with_meta_llama_keywords():
    # Simulate keywords returned by Meta Llama API (replace with actual API call if needed)
    meta_llama_keywords = ["embroidered", "maroon", "organza"]
    db = Database()
    results = db.search_products_by_keywords(meta_llama_keywords, num_results=5)
    print("Search results for Meta Llama keywords:")
    for product in results:
        print(product)
    db.close()

if __name__ == "__main__":
    test_search_with_meta_llama_keywords()

