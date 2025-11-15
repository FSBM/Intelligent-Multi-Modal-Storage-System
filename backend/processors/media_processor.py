import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import asyncio
from typing import Dict
import logging
import json
from urllib.request import urlopen

logger = logging.getLogger(__name__)

class MediaProcessor:
    """Processes media files (images/videos) to generate embeddings and classify using ResNet50 (ImageNet)"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing MediaProcessor on device: {self.device}")
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.eval()
        self.resnet.to(self.device)
        
        # Embedding model: everything except final classification layer
        self.embedding_model = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.classifier = self.resnet  # full model for predictions
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Load ImageNet class labels
        try:
            url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            with urlopen(url) as f:
                self.imagenet_labels = json.load(f)
            logger.info("Loaded ImageNet labels successfully")
        except Exception as e:
            logger.warning(f"Could not load ImageNet labels: {e}")
            self.imagenet_labels = [f"class_{i}" for i in range(1000)]
    
    async def process(self, file_path: Path, mime_type: str) -> Dict:
        """Process media file and return embeddings, category, and metadata"""
        try:
            if mime_type.startswith("image/"):
                return await self._process_image(file_path, mime_type)
            elif mime_type.startswith("video/"):
                return await self._process_video(file_path, mime_type)
            else:
                raise ValueError(f"Unsupported media type: {mime_type}")
        except Exception as e:
            logger.error(f"Error processing media {file_path}: {e}", exc_info=True)
            return {
                "embeddings": None,
                "category": "uncategorized",
                "metadata": {"error": str(e)},
            }
    
    async def _process_image(self, file_path: Path, mime_type: str) -> Dict:
        """Process image using ResNet embeddings and ImageNet classification"""
        loop = asyncio.get_event_loop()
        
        # Load image
        image = Image.open(file_path).convert("RGB")
        
        # Metadata
        metadata = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "mime_type": mime_type,
        }
        
        # Preprocess image
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            features = self.embedding_model(image_tensor)
            embeddings = features.cpu().numpy().flatten().tolist()
        
        # Predict category using ResNet classifier
        def predict_category():
            with torch.no_grad():
                outputs = self.classifier(image_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_idx = torch.argmax(probs, dim=1).item()
                return self.imagenet_labels[top_idx]
        
        category = await loop.run_in_executor(None, predict_category)
        
        return {
            "embeddings": embeddings,
            "category": category,
            "metadata": metadata,
        }
    
    async def _process_video(self, file_path: Path, mime_type: str) -> Dict:
        """Video placeholder (extracting frame not implemented)"""
        logger.info(f"Video processing for {file_path} - placeholder")
        metadata = {
            "mime_type": mime_type,
            "type": "video",
            "note": "Video frame extraction not implemented",
        }
        return {
            "embeddings": None,
            "category": "uncategorized",
            "metadata": metadata,
        }
    
    def get_embeddings_dim(self) -> int:
        """Embedding dimension from ResNet output"""
        return 2048
