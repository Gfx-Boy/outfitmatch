from flask import Flask, request, jsonify, render_template, redirect, session, url_for
from flask_cors import CORS
from databaseClass import Database
import mysql.connector
from flask import flash
import re
import MySQLdb.cursors
from flask_mysqldb import MySQL
import mysql.connector

# --- FAISS, CLIP, E5, and helper imports ---
import faiss
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import re
import json
import ast

app = Flask(_name_)

# Initialize the database
db = Database()
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''  # Set your MySQL password here
app.config['MYSQL_DB'] = 'cloth_design_database'
app.config['SECRET_KEY'] = '123'

# ✅ Initialize MySQL
mysql = MySQL(app)

# Load models (adjust paths as needed)
clip_model = CLIPModel.from_pretrained("C:\\Users\\Ahmer\\Downloads\\models\\openaiclip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("C:\\Users\\Ahmer\\Downloads\\models\\openaiclip-vit-base-patch32")
e5_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
e5_model = AutoModel.from_pretrained('intfloat/e5-large-v2')


# FAISS Index Class
class FaissIndex:
    def _init_(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)
        self.product_data = []

    def add_vectors(self, vectors, product_data):
        vectors = np.array(vectors).astype('float32')
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.product_data.extend(product_data)

    def search(self, query_vector, k=10, filters=None):
        query_vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            product = self.product_data[idx]
            product['similarity'] = float(distances[0][i])
            results.append(product)
        return results


# Helper functions for feature extraction and filtering
def extract_clip_text_features(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
    return features / features.norm(p=2, dim=-1, keepdim=True)


def extract_e5_text_features(text):
    text = "query: " + text if not text.startswith("query: ") else text
    inputs = e5_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512,
                          return_attention_mask=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    e5_model.to(device)
    with torch.no_grad():
        outputs = e5_model(**inputs)
    token_embeddings = outputs.last_hidden_state
    input_mask = inputs['attention_mask']
    input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu()


def json_to_numpy(features_json):
    if isinstance(features_json, np.ndarray):
        return features_json.astype(np.float32)
    if isinstance(features_json, (list, tuple)):
        return np.array(features_json, dtype=np.float32)
    if isinstance(features_json, str):
        try:
            parsed = json.loads(features_json)
            if isinstance(parsed, dict):
                features = parsed['features']
            else:
                features = parsed
        except json.JSONDecodeError:
            features = ast.literal_eval(features_json)
        return np.array(features, dtype=np.float32)
    if isinstance(features_json, dict):
        return np.array(features_json['features'], dtype=np.float32)
    raise ValueError(f"Unsupported input type: {type(features_json)}")


def is_gibberish(text):
    if len(text) < 3:
        return False
    if len(set(text.lower())) < (len(text) / 2):
        return True
    adjacent_pairs = [
        ('q', 'w'), ('w', 'e'), ('e', 'r'), ('r', 't'), ('t', 'y'), ('y', 'u'), ('u', 'i'), ('i', 'o'), ('o', 'p'),
        ('a', 's'), ('s', 'd'), ('d', 'f'), ('f', 'g'), ('g', 'h'), ('h', 'j'), ('j', 'k'), ('k', 'l'),
        ('z', 'x'), ('x', 'c'), ('c', 'v'), ('v', 'b'), ('b', 'n'), ('n', 'm')
    ]
    text_lower = text.lower()
    adjacent_count = 0
    for i in range(len(text_lower) - 1):
        if (text_lower[i], text_lower[i + 1]) in adjacent_pairs:
            adjacent_count += 1
    if adjacent_count / (len(text) - 1) > 0.5:
        return True
    return False


def extract_colors(query):
    EASTERN_COLOR_PALETTE = {
        'red': ['crimson', 'maroon', 'ruby', 'burgundy', 'scarlet'],
        'blue': ['navy', 'peacock', 'sapphire', 'azure', 'cobalt'],
        'green': ['emerald', 'jade', 'olive', 'forest', 'mint'],
        'gold': ['metallic', 'mustard', 'brass', 'golden', 'amber'],
        'white': ['ivory', 'off-white', 'cream', 'pearl', 'snow'],
        'black': ['ebony', 'jet', 'onyx', 'charcoal', 'raven'],
        'pink': ['rose', 'blush', 'fuchsia', 'magenta', 'coral'],
        'purple': ['violet', 'lavender', 'lilac', 'plum', 'amethyst']
    }
    colors = []
    query = query.lower()
    for color in EASTERN_COLOR_PALETTE:
        if re.search(rf'\\b{color}\\b', query):
            colors.append(color)
    for color, variants in EASTERN_COLOR_PALETTE.items():
        if any(re.search(rf'\\b{v}\\b', query) for v in variants):
            colors.append(color)
    return list(set(colors))


def enhance_eastern_results(results):
    # Dummy: just return results for now
    return results


# --- Initialize FAISS indices and load product data ---
clip_faiss = FaissIndex(512)
e5_faiss = FaissIndex(1024)


def initialize_faiss_indices():
    cursor = mysql.connection.cursor()
    cursor.execute("""
                   SELECT f.id,
                          f.product_link,
                          f.clip_features,
                          d.product_name,
                          d.product_price,
                          d.product_size,
                          d.product_image,
                          pd.product_description,
                          pd.e5_features
                   FROM image_features_table f
                            JOIN product_detail d ON f.product_link = d.product_link
                            JOIN product_description pd ON f.product_link = pd.product_link
                   """)
    products = cursor.fetchall()
    cursor.close()
    clip_vectors = []
    e5_vectors = []
    product_data = []
    for product in products:
        try:
            product_info = {
                "product_id": product[0],
                "product_link": product[1],
                "product_name": product[3],
                "product_price": product[4],
                "product_size": product[5],
                "product_image": product[6],
                "product_description": product[7],
            }
            clip_features = json_to_numpy(product[2])
            e5_features = json_to_numpy(product[8])
            clip_vectors.append(clip_features)
            e5_vectors.append(e5_features)
            product_data.append(product_info)
        except Exception as e:
            print(f"Skipping product {product[0]}: {str(e)}")
            continue
    clip_faiss.add_vectors(clip_vectors, product_data)
    e5_faiss.add_vectors(e5_vectors, product_data)
    print(f"FAISS indices initialized with {len(product_data)} products")


# Call this once at startup
def setup_app():
    initialize_faiss_indices()


setup_app()


# --- New text search function ---
def get_product_details_from_description(description_input, num_results=5):
    try:
        if is_gibberish(description_input):
            return {"error": "Couldn't understand your search",
                    "suggestion": "Try searching for items like 'blue dress' or 'cotton shirt'"}
        e5_features = extract_e5_text_features(description_input).numpy()[0]
        clip_features = extract_clip_text_features(description_input).numpy()[0]
        filters = {}
        colors = extract_colors(description_input)
        if colors:
            filters['colors'] = colors
        # Search both indices
        e5_results = e5_faiss.search(e5_features, num_results * 2, filters)
        clip_results = clip_faiss.search(clip_features, num_results * 2, filters)
        # Combine and re-rank results
        combined_results = {}
        for result in e5_results + clip_results:
            pid = result['product_id']
            if pid not in combined_results:
                combined_results[pid] = dict(result)
                combined_results[pid]['combined_similarity'] = result['similarity']
            else:
                combined_results[pid]['combined_similarity'] += result['similarity']
        final_results = sorted(combined_results.values(), key=lambda x: x['combined_similarity'], reverse=True)[
                        :num_results]
        SIMILARITY_THRESHOLD = 0.6
        final_results = [r for r in final_results if r['combined_similarity'] >= SIMILARITY_THRESHOLD]
        if not final_results:
            return {"error": "No matching products found"}
        return enhance_eastern_results(final_results)
    except Exception as e:
        print(f"Text search error: {str(e)}")
        return {"error": str(e)}


# @app.route('/search', methods=['POST'])
# def search_product():
#     """
#     API endpoint to receive input text and return matched product details
#     """
#     try:
#         # Get the input text (description) from the POST request
#         data = request.get_json()
#         print(f"Received data: {data}")

#         # Extract description input from the received data
#         description_input = data.get('description', '')

#         if not description_input:
#             return jsonify({"error": "Description input is missing"}), 400

#         print(f"Searching for product details matching: {description_input}")

#         # Get the most similar product based on description
#         product_details = get_product_details_from_description(description_input)

#         if 'error' in product_details:
#             print(f"Error in product search: {product_details['error']}")
#             return jsonify(product_details), 500

#         # Return the product details as JSON
#         print(f"Returning product details: {product_details}")
#         return jsonify(product_details), 200

#     except Exception as e:
#         print(f"Error_textttt: {str(e)}")
#         return jsonify({"error": str(e)}), 500


CORS(app, resources={r"/search": {"origins": "http://127.0.0.1:5500"}})


@app.route('/search', methods=['POST'])
def search_product():
    """
    API endpoint to receive input text, return matched product details, and store search history with query
    """
    try:
        # Get the input text (description) from the POST request
        data = request.get_json()
        print(f"Received data: {data}")

        # Extract description input from the received data
        description_input = data.get('description', '')

        if not description_input:
            return jsonify({"error": "Description input is missing"}), 400

        print(f"Searching for product details matching: {description_input}")

        # Get the most similar product based on description
        product_details = get_product_details_from_description(description_input)

        if 'error' in product_details:
            print(f"Error in product search: {product_details['error']}")
            return jsonify(product_details), 500

        # If user is logged in, store the search history
        if 'loggedin' in session:
            user_id = session['id']
            cursor = mysql.connection.cursor()

            # Insert each product result into the search_history table with the search query
            for product in product_details:
                query = """
                        INSERT INTO search_history (user_id, search_query, product_name, product_description, \
                                                    regular_price)
                        VALUES (%s, %s, %s, %s, %s) \
                        """
                values = (
                    user_id,
                    description_input,  # Store the user's input query
                    product['product_name'],
                    product['product_description'],
                    product['product_price']
                )
                cursor.execute(query, values)

            # Commit the transaction
            mysql.connection.commit()
            cursor.close()
            print(f"Search history stored for user_id: {user_id} with query: {description_input}")

        # Return the product details as JSON
        print(f"Returning product details: {product_details}")
        return jsonify(product_details), 200

    except Exception as e:
        print(f"Error_textttt: {str(e)}")
        return jsonify({"error": str(e)}), 500


# updated code for signup and login
@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/login/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('index'))
        else:
            # Account doesnt exist or username/password incorrect
            flash("Incorrect username/password!", "danger")
    return render_template('auth/login.html', title="Login")


@app.route('/auth/register', methods=['GET', 'POST'])
def register():
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('SELECT * FROM users WHERE username = %s', (username))
        cursor.execute("SELECT * FROM users WHERE username LIKE %s", [username])
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email:
            flash("Incorrect username/password!", "danger")
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO users VALUES (NULL, %s, %s, %s)', (username, email, password))
            mysql.connection.commit()
            flash("You have successfully registered!", "success")
            return redirect(url_for('login'))

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash("Please fill out the form!", "danger")
    # Show registration form with message (if any)
    return render_template('./auth/register.html', title="Register")


@app.route('/search_history', methods=['GET'])
def search_history():
    """
    Retrieve and display the user's search history.
    """
    if 'loggedin' not in session:
        flash("You must be logged in to view search history!", "danger")
        return redirect(url_for('login'))

    user_id = session['id']
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Fetch all relevant columns, including search_query
    cursor.execute(
        'SELECT search_query, product_name, product_description, regular_price, search_time FROM search_history WHERE user_id = %s ORDER BY search_time DESC',
        (user_id,))
    history = cursor.fetchall()
    cursor.close()

    return render_template('search_history2.html', history=history)


if _name_ == '_main_':
    app.run(debug=True, port = 5000)