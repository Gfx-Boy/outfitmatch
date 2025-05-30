from bs4 import BeautifulSoup
import pandas as pd
import requests
from databaseClass import Database

class NewWebScraper:
    def __init__(self, db):
        """Initialize the WebScraper class with the database object."""
        self.db = db

        print("asdasdada")

    def fetch_page(self, url):
        """Fetch the HTML content of a page."""
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching page {url}: {e}")
            return None

    def extract_links(self, raw_html, link_class):
        """Extract all links with a specific class from the HTML content."""
        soup = BeautifulSoup(raw_html, "html.parser")
        links = []

        # Find all <a> tags with the given class
        link_elements = soup.find_all("a", class_=link_class)
        for link in link_elements:
            href = link.get("href")
            if href:
                links.append(href)
        return links

    def scrape_product_details(self, product_url, website_config):
        """Scrape detailed information about a product."""
        raw_html = self.fetch_page(product_url)
        if not raw_html:
            print(f"Failed to fetch product page: {product_url}")
            return None

        soup = BeautifulSoup(raw_html, "html.parser")

        # Use website-specific class names from config
        price_class = website_config["price_class"]
        size_class = website_config["size_class"]
        availability_class = website_config["availability_class"]
        description_class = website_config["description_class"]
        image_class = website_config["image_class"]

        try:
            price = soup.find(class_=price_class).text.strip()
        except AttributeError:
            price = None

        try:
            size = [size.text.strip() for size in soup.find_all(class_=size_class)]
        except AttributeError:
            size = None

        try:
            availability = soup.find(class_=availability_class).text.strip()
        except AttributeError:
            availability = None

        try:
            description = soup.find(class_=description_class).text.strip()
        except AttributeError:
            description = None

        try:
            image = soup.find(class_=image_class).get("src")
        except AttributeError:
            image = None

        return {
            "url": product_url,
            "price": price,
            "size": size,
            "availability": availability,
            "description": description,
            "image": image
        }

    def scrape_collections(self, base_url, collection_links, website_config):
        """Scrape each collection link to extract product links and details."""
        all_product_details = []
        for collection in collection_links:
            collection_url = base_url + collection if not collection.startswith("http") else collection
            print(f"Scraping collection: {collection_url}")
            raw_html = self.fetch_page(collection_url)
            if raw_html:
                product_links = self.extract_links(raw_html, website_config["product_link_class"])
                print(f"Found {len(product_links)} product links in collection: {collection_url}")
                for product_link in product_links:
                    product_url = base_url + product_link if not product_link.startswith("http") else product_link
                    product_details = self.scrape_product_details(product_url, website_config)
                    if product_details:
                        all_product_details.append(product_details)
        return all_product_details

    def scrape_website(self, config):
        """Scrape a single website using the given configuration."""
        base_url = config["base_url"]

        print(f"Scraping base URL: {base_url}")
        raw_html = self.fetch_page(base_url)
        if not raw_html:
            print(f"Failed to fetch base URL: {base_url}")
            return

        # Step 1: Check if base URL has both collections and products
        collection_links = self.extract_links(raw_html, config["collection_link_class"])
        product_links = self.extract_links(raw_html, config["product_link_class"])

        all_product_details = []

        if collection_links and not product_links:
            print(f"Found {len(collection_links)} collection links. Scraping them for product links...")
            all_product_details = self.scrape_collections(base_url, collection_links, config)
        elif product_links:
            print(f"Found {len(product_links)} product links directly at base URL.")
            for product_link in product_links:
                product_url = base_url + product_link if not product_link.startswith("http") else product_link
                product_details = self.scrape_product_details(product_url, config)
                if product_details:
                    all_product_details.append(product_details)
        else:
            print("No valid collections or product links found. Skipping...")

        # Step 2: Save all product details to Excel
        if all_product_details:
            # self.save_to_excel(all_product_details, f"{config['base_url'].replace('https://', '').replace('/', '_')}_products.xlsx")
            self.save_to_excel(all_product_details,f"D:\\WebScrapedData\\_products.xlsx")
            print()

    def save_to_excel(self, data, file_name):
        """Save product details to an Excel file in sorted form."""
        try:
            # Convert the list of dictionaries to a pandas DataFrame
            df = pd.DataFrame(data)
            
            # Sorting by price (if it exists)
            if "price" in df.columns:
                df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert price to numeric, handle errors
                df = df.sort_values(by='price', ascending=True)

            # Save to Excel
            df.to_excel(file_name, index=False)
            print(f"Product details saved to Excel file: {file_name}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")

    def scrape_multiple_websites(self, website_configs):
        """Scrape multiple websites based on their configurations."""
        for config in website_configs:
            print(f"Starting scrape for website: {config['base_url']}")
            self.scrape_website(config)
            print(f"Completed scrape for website: {config['base_url']}")
