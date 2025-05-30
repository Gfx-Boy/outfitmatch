from flask import Flask, request, jsonify, render_template, redirect, session, url_for
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from databaseClass import Database
import mysql.connector
from flask import flash
import re
import MySQLdb.cursors
from flask_mysqldb import MySQL
import mysql.connector

import json



app = Flask(__name__)

# Initialize the database
db = Database()
app.config['MYSQL_HOST'] = '193.203.166.181'
app.config['MYSQL_USER'] = 'u557851335_cloth_db'
app.config['MYSQL_PASSWORD'] = 'Newton 12'
app.config['MYSQL_DB'] = 'u557851335_cloth_design_d'
app.config['SECRET_KEY'] = '123'

# ✅ Initialize MySQL
mysql = MySQL(app)


def get_product_details_from_description(description_input, num_results=5):
    """
    Function to get product details from the database by comparing the input text with product descriptions
    and return multiple similar products.
    """
    try:
        # Retrieve all descriptions and their corresponding product details from the database
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

        # If no products were returned
        if not products:
            print("No products found in the database.")  # Log when no products are found
            return {"error": "No products found in the database"}

        # Extract product descriptions and product IDs for comparison
        product_descriptions = [product[2] for product in products]
        product_ids = [product[0] for product in products]
        product_names = [product[1] for product in products]
        product_prices = [product[3] for product in products]
        product_sizes = [product[4] for product in products]
        product_images = [product[5] for product in products]
        product_links = [product[6] for product in products]

        # Use TfidfVectorizer to convert text into vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(product_descriptions + [description_input])

        # Calculate cosine similarity between the input description and all product descriptions
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

        # Get the indices of the most similar products
        similarity_scores = cosine_similarities.flatten()  # Flatten the array to get all similarity scores
        sorted_idx = similarity_scores.argsort()[-num_results:][::-1]  # Get top N most similar products, sorted by score

        # Prepare the response with multiple product details
        result = []
        for idx in sorted_idx:
            product = {
                "product_id": product_ids[idx],
                "product_name": product_names[idx],
                "product_description": product_descriptions[idx],
                "product_price": product_prices[idx],
                "product_size": product_sizes[idx],
                "product_image": product_images[idx],
                "product_link": product_links[idx]
            }
            result.append(product)

        print(f"Returning top {num_results} product details: {result}")  # Log the returned result
        return result
    except mysql.connector.Error as e:
        print(f"Error retrieving products from the database: {e}")  # Log the error
        return {"error": f"Error retrieving products from the database: {e}"}



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
                INSERT INTO search_history (user_id, search_query, product_name, product_description, regular_price)
                VALUES (%s, %s, %s, %s, %s)
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
    return render_template('auth/login.html',title="Login")




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
        cursor.execute( "SELECT * FROM users WHERE username LIKE %s", [username] )
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
            cursor.execute('INSERT INTO users VALUES (NULL, %s, %s, %s)', (username,email, password))
            mysql.connection.commit()
            flash("You have successfully registered!", "success")
            return redirect(url_for('login'))

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash("Please fill out the form!", "danger")
    # Show registration form with message (if any)
    return render_template('./auth/register.html',title="Register")



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
    cursor.execute('SELECT search_query, product_name, product_description, regular_price, search_time FROM search_history WHERE user_id = %s ORDER BY search_time DESC', (user_id,))
    history = cursor.fetchall()
    cursor.close()

    return render_template('search_history2.html', history=history)



if __name__ == '__main__':
    app.run(debug=True, port=5000)
