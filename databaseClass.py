import mysql.connector
from mysql.connector import Error

class Database:
    def __init__(self, host="193.203.166.181", user="u557851335_cloth_db", password="Newton 12", database="u557851335_cloth_design_d"):
        """Initialize the database connection."""

        try:
            self.conn = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            self.cursor = self.conn.cursor()

            # Ensure the tables are created
            # self.create_tables()
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            self.conn = None
            self.cursor = None

    def create_tables(self):
        """Create the tables if they don't exist."""
        if self.conn and self.cursor:
            # Create product_details table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS product_detail (
                    Product_id INT AUTO_INCREMENT PRIMARY KEY,
                    Product_image Text,
                    Product_link VARCHAR(255) UNIQUE NOT NULL PRIMARY KEY,
                    Product_price VARCHAR(255),
                    Product_size TEXT,
                    Product_name VARCHAR(255)
                )
            """)

    def is_link_existing(self, link):
        """Check if the product link already exists in both tables."""
        if not self.conn or not self.cursor:
            return False

        # Check in product_description table
        self.cursor.execute("SELECT 1 FROM product_description WHERE Product_link = %s", (link,))
        if self.cursor.fetchone():
            print("true")
            return True
            

        # Check in product_detail table
        self.cursor.execute("SELECT 1 FROM product_detail WHERE Product_link = %s", (link,))
        if self.cursor.fetchone():
            return True

        return False

    def save_product_links(self, product_links):
        """Save product links to both tables."""
        if not self.conn or not self.cursor:
            return

        for link in product_links:
            # Check if the link already exists
            if not self.is_link_existing(link):
                # Insert the product link into the product_description table
                self.cursor.execute("INSERT INTO product_description (Product_link) VALUES (%s)", (link,))
                # Insert the product link into the product_detail table
                self.cursor.execute("INSERT INTO product_detail (Product_link) VALUES (%s)", (link,))
                print(f"Saved new link: {link}")
            else:
                print(f"Link already exists: {link}")

        self.conn.commit()

    
    def save_product_details(self, product_details):
        if not self.conn or not self.cursor:
            print("Database connection not initialized.")
            return

        for detail in product_details:
            try:
                #inserting in the database tablename(product-detail)
                self.cursor.execute("""
                INSERT INTO product_detail(product_name, product_price, product_size, product_image, product_link)
                values("%s", %s, %s, %s,%s); """, (detail["name"], (detail["price"]), ','.join(detail["size"]), detail["image"], detail["url"]))
                #inserting in the database tablename(product-description)
                self.cursor.execute("""
                INSERT INTO product_description(product_description, product_link)
                values("%s", %s); """, (detail["description"], (detail["url"])))
                self.conn.commit()

            except Error as e:
                print(f"SQL Error: {e}")

        print("All product details saved.")

    def get_all_product_links(self):
        """Retrieve all product links from both tables."""
        if not self.conn or not self.cursor:
            return []

        self.cursor.execute("SELECT Product_link FROM product_description")
        description_links = self.cursor.fetchall()

        self.cursor.execute("SELECT Product_link FROM product_detail")
        details_links = self.cursor.fetchall()

        # Combine the links from both tables
        all_links = set(link[0] for link in description_links + details_links)
        return all_links

    def search_products_by_keywords(self, keywords, num_results=10):
        """
        Search for products where any of the keywords appear in product_name, product_size, product_price, or product_description.
        Returns a list of matching product details.
        """
        if not self.conn or not self.cursor:
            return []

        # Prepare the WHERE clause for each keyword in all relevant fields
        like_clauses = []
        params = []
        for kw in keywords:
            like_clauses.append("(pd.product_name LIKE %s OR pd.product_size LIKE %s OR pd.product_price LIKE %s OR pdesc.product_description LIKE %s)")
            for _ in range(4):
                params.append(f"%{kw}%")
        where_clause = " OR ".join(like_clauses)

        query = f"""
            SELECT pd.Product_id, pd.product_name, pd.product_price, pd.product_size, pd.Product_image, pd.Product_link, pdesc.product_description
            FROM product_detail pd
            LEFT JOIN product_description pdesc ON pd.Product_link = pdesc.Product_link
            WHERE {where_clause}
            LIMIT %s
        """
        params.append(num_results)
        self.cursor.execute(query, tuple(params))
        results = self.cursor.fetchall()
        # Return as list of dicts for easier use
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in results]

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

