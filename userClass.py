import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from UserDatabase import Database

class User:
    def __init__(self):
        # Load the pre-trained CLIP model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.database = Database()

    def process_image_and_find_matches(self, image_path):
        # Load and preprocess the image
        inputs = self.processor(images=image_path, return_tensors="pt", padding=True)
        
        # Extract image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_np = image_features.cpu().numpy()
        
        # Fetch text descriptions and their embeddings from the database
        descriptions, text_features = self.database.get_text_features()
        
        # Compute cosine similarity between image and text features
        similarities = cosine_similarity(image_features_np, text_features)
        
        # Find the top matches
        top_indices = np.argsort(similarities[0])[::-1][:5]
        matches = [descriptions[i] for i in top_indices]
        
        return matches
