import mysql.connector
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection configuration
db_config = {
    'host': '193.203.166.181',
    'user': 'u557851335_cloth_db',
    'password': 'Newton 12',  # Update with your MySQL password
    'database': 'u557851335_cloth_design_d'
}

# Define a list of known colors and their synonyms
color_map = {
    'red': ['red', 'crimson', 'scarlet', 'brick red', 'berry', 'hot pink'],
    'maroon': ['maroon'],
    'green': ['green', 'olive', 'army-green', 'emerald', 'bottle green', 'mehndi green', 'mint green'],
    'white': ['white'],
    'purple': ['purple', 'dark purple', 'violet', 'eggplant', 'lilac'],
    'teal': ['teal', 'teal blue'],
    'beige': ['beige'],
    'yellow': ['yellow', 'gold', 'mustard'],
    'blue': ['blue', 'dark blue', 'ice blue', 'navy', 'midnight blue', 'navy blue', 'powder blue'],
    'grey': ['grey', 'mint gray', 'gray'],
    'orange': ['orange', 'rust', 'peach'],
    'black': ['black'],
    'pink': ['pink', 'baby pink', 'tea pink', 'blush pink']
}

# Reverse mapping: color synonym -> primary color
color_lookup = {}
for primary_color, synonyms in color_map.items():
    for synonym in synonyms:
        color_lookup[synonym.lower()] = primary_color

def check_and_add_color_column(cursor):
    try:
        # Check if 'color' column exists
        cursor.execute("SHOW COLUMNS FROM product_detail LIKE 'color'")
        if not cursor.fetchone():
            logger.info("Adding 'color' column to product_detail table")
            cursor.execute("ALTER TABLE product_detail ADD color VARCHAR(50)")
            logger.info("'color' column added successfully")
        else:
            logger.info("'color' column already exists")
    except mysql.connector.Error as e:
        logger.error(f"Error checking/adding color column: {e}")
        raise

def extract_primary_color(product_name, description):
    product_name_lower = product_name.lower()
    description_lower = description.lower()

    # Step 1: Check the description for explicit "Color: <color>" pattern
    if "color:" in description_lower:
        parts = description_lower.split("color:")[1].split()
        if parts:
            first_word = parts[0].strip()
            if first_word in color_lookup:
                logger.debug(f"Color found in 'Color:' pattern: {first_word} -> {color_lookup[first_word]}")
                return color_lookup[first_word]

    # Step 2: Check the product name for a color
    words = product_name_lower.split()
    for word in words:
        word = word.strip()
        if word in color_lookup:
            logger.debug(f"Color found in product name: {word} -> {color_lookup[word]}")
            return color_lookup[word]

    # Step 3: Check the description for the first known color
    words = description_lower.split()
    for word in words:
        word = word.strip()
        if word in color_lookup:
            logger.debug(f"Color found in description: {word} -> {color_lookup[word]}")
            return color_lookup[word]

    logger.warning(f"No color found for product name: {product_name}")
    return None

def populate_colors():
    conn = None
    cursor = None
    try:
        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Add color column if it doesn't exist
        check_and_add_color_column(cursor)
        conn.commit()

        # Fetch total number of products
        cursor.execute("SELECT COUNT(*) FROM product_detail")
        total_products = cursor.fetchone()[0]
        batch_size = 1000  # Process 1000 records at a time
        logger.info(f"Total products to process: {total_products}")

        # Open file to log unmatched products
        with open('unmatched_products.txt', 'w', encoding='utf-8') as unmatched_file:
            # Fetch products in batches
            offset = 0
            while offset < total_products:
                query = """
                SELECT product_detail.Product_link, product_detail.product_name, product_description.product_description
                FROM product_detail
                JOIN product_description ON product_detail.Product_link = product_description.Product_link
                LIMIT %s OFFSET %s
                """
                cursor.execute(query, (batch_size, offset))
                products = cursor.fetchall()

                if not products:
                    logger.info("No more products to process.")
                    break

                # Process each product in the batch
                for product in products:
                    product_link = product[0]
                    product_name = product[1]
                    description = product[2]
                    primary_color = extract_primary_color(product_name, description)
                    if primary_color:
                        # Update the color column
                        update_query = """
                        UPDATE product_detail
                        SET color = %s
                        WHERE Product_link = %s
                        """
                        cursor.execute(update_query, (primary_color, product_link))
                        logger.info(f"Updated product {product_link} with color {primary_color}")
                    else:
                        # Log unmatched products
                        unmatched_file.write(f"{product_link} | {product_name} | {description[:100]}...\n")
                        logger.warning(f"No color found for product {product_link}")

                # Commit the batch
                conn.commit()
                logger.info(f"Processed batch of {len(products)} products, offset: {offset}")
                offset += batch_size

        logger.info("Color column populated successfully. Check unmatched_products.txt for products without detected colors.")

    except mysql.connector.Error as e:
        logger.error(f"Error updating database: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            logger.info("Database connection closed.")

if __name__ == '__main__':
    populate_colors()

