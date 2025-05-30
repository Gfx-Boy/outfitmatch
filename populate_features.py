from flask import Flask
from flask_mysqldb import MySQL
import os
import mysql.connector
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoTokenizer
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MySQL configuration
app.config['MYSQL_HOST'] = '193.203.166.181'
app.config['MYSQL_USER'] = 'u557851335_cloth_db'
app.config['MYSQL_PASSWORD'] = 'Newton 12'  # Update with your MySQL password
app.config['MYSQL_DB'] = 'u557851335_cloth_design_d'

# Initialize MySQL
mysql = MySQL(app)

# Load models
logger.info("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

logger.info("Loading DINOv2 model...")
dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast=True)
dino_model = AutoModel.from_pretrained('facebook/dinov2-small')

logger.info("Loading E5 text embedding model...")
e5_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
e5_model = AutoModel.from_pretrained('intfloat/e5-large-v2')

def extract_clip_features(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features / features.norm(p=2, dim=-1, keepdim=True)

def extract_dino_features(image):
    inputs = dino_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = dino_model(**inputs).last_hidden_state.mean(dim=1)
    return features / features.norm(p=2, dim=-1, keepdim=True)

def extract_e5_text_features(text):
    text = "query: " + text if not text.startswith("query: ") else text
    inputs = e5_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )
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

def populate_features():
    with app.app_context():
        cursor = mysql.connection.cursor()
        
        # Fetch products from product_detail and product_description
        query = """
        SELECT d.product_link, d.product_image, pd.product_description
        FROM product_detail d
        JOIN product_description pd ON d.product_link = pd.product_link
        """
        cursor.execute(query)
        products = cursor.fetchall()
        
        for product in products:
            product_link, product_image, product_description = product
            logger.info(f"Processing product: {product_link}")
            
            # Load image
            try:
                image_path = os.path.join('path/to/your/product/images', product_image)  # Update with actual path
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to load image for {product_link}: {str(e)}")
                continue
            
            # Extract features
            try:
                clip_features = extract_clip_features(image).numpy()[0].tolist()
                dino_features = extract_dino_features(image).numpy()[0].tolist()
                e5_features = extract_e5_text_features(product_description).numpy()[0].tolist()
                
                # Store features in image_features_table
                insert_query = """
                INSERT INTO image_features_table (product_link, clip_features, dinov2_features, e5_features)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    clip_features = VALUES(clip_features),
                    dinov2_features = VALUES(dinov2_features),
                    e5_features = VALUES(e5_features)
                """
                cursor.execute(insert_query, (
                    product_link,
                    json.dumps(clip_features),
                    json.dumps(dino_features),
                    json.dumps(e5_features)
                ))
                mysql.connection.commit()
                logger.info(f"Features stored for {product_link}")
            except Exception as e:
                logger.error(f"Failed to process features for {product_link}: {str(e)}")
                continue
        
        cursor.close()

if __name__ == '__main__':
    populate_features()