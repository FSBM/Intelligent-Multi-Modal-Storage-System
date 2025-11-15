<<<<<<< HEAD
import torch
from torchvision import models, transforms
=======
"""
Media Processor
Generates embeddings via CLIP/ViT and performs semantic classification
"""
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
from PIL import Image
import numpy as np
from pathlib import Path
import asyncio
<<<<<<< HEAD
from typing import Dict
import logging
import json
from urllib.request import urlopen

logger = logging.getLogger(__name__)

class MediaProcessor:
    """Processes media files (images/videos) to generate embeddings and classify using ResNet50 (ImageNet)"""
=======
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MediaProcessor:
    """Processes media files (images/videos) to generate embeddings and classifications"""
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing MediaProcessor on device: {self.device}")
        
<<<<<<< HEAD
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.eval()
        self.resnet.to(self.device)
        
        # Embedding model: everything except final classification layer
        self.embedding_model = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.classifier = self.resnet  # full model for predictions
=======
        # Load CLIP model for image embeddings
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.clip_model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load CLIP model: {e}. Using fallback.")
            self.clip_model = None
            self.clip_processor = None
        
        # Predefined categories for classification
        self.categories = [
            "nature", "animals", "people", "architecture", "food",
            "vehicles", "technology", "art", "sports", "travel",
            "business", "medical", "education", "entertainment", "other"
        ]
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
<<<<<<< HEAD
        
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
=======
    
    async def process(self, file_path: Path, mime_type: str) -> Dict:
        """
        Process media file and return embeddings, category, and metadata
        
        Args:
            file_path: Path to media file
            mime_type: MIME type of the file
            
        Returns:
            Dictionary with embeddings, category, and metadata
        """
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
        try:
            if mime_type.startswith("image/"):
                return await self._process_image(file_path, mime_type)
            elif mime_type.startswith("video/"):
                return await self._process_video(file_path, mime_type)
            else:
                raise ValueError(f"Unsupported media type: {mime_type}")
        except Exception as e:
            logger.error(f"Error processing media {file_path}: {e}", exc_info=True)
<<<<<<< HEAD
=======
            # Return default values on error
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
            return {
                "embeddings": None,
                "category": "uncategorized",
                "metadata": {"error": str(e)},
            }
    
    async def _process_image(self, file_path: Path, mime_type: str) -> Dict:
<<<<<<< HEAD
        """Process image using ResNet embeddings and ImageNet classification"""
=======
        """Process image file"""
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
        loop = asyncio.get_event_loop()
        
        # Load image
        image = Image.open(file_path).convert("RGB")
        
<<<<<<< HEAD
        # Metadata
=======
        # Extract basic metadata
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
        metadata = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "mime_type": mime_type,
        }
        
<<<<<<< HEAD
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
=======
        # Generate embeddings
        embeddings = None
        category = "uncategorized"
        
        if self.clip_model and self.clip_processor:
            try:
                # Generate CLIP embeddings
                inputs = self.clip_processor(
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    embeddings = image_features.cpu().numpy().flatten().tolist()
                
                # Classify using text-image similarity
                category = await loop.run_in_executor(
                    None, self._classify_image, image, embeddings
                )
                
            except Exception as e:
                logger.warning(f"CLIP processing failed: {e}")
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
        
        return {
            "embeddings": embeddings,
            "category": category,
            "metadata": metadata,
        }
    
<<<<<<< HEAD
    async def _process_video(self, file_path: Path, mime_type: str) -> Dict:
        """Video placeholder (extracting frame not implemented)"""
        logger.info(f"Video processing for {file_path} - placeholder")
        metadata = {
            "mime_type": mime_type,
            "type": "video",
            "note": "Video frame extraction not implemented",
        }
=======
    def _classify_image(self, image: Image.Image, embeddings: List[float]) -> str:
        """Classify image into category using CLIP"""
        if not self.clip_model or not embeddings:
            return "uncategorized"
        
        try:
            # Create text prompts for categories
            category_texts = [f"a photo of {cat}" for cat in self.categories]
            
            # Get text embeddings
            text_inputs = self.clip_processor(
                text=category_texts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            image_emb = torch.tensor(embeddings).unsqueeze(0).to(self.device)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            
            similarity = (image_emb @ text_features.T).cpu().numpy().flatten()
            best_category_idx = np.argmax(similarity)
            
            # Only return category if similarity is above threshold
            if similarity[best_category_idx] > 0.2:
                return self.categories[best_category_idx]
            else:
                return "uncategorized"
        
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return "uncategorized"
    
    async def _process_video(self, file_path: Path, mime_type: str) -> Dict:
        """Process video file (extract frame and process as image)"""
        # For now, we'll extract a frame from the video
        # In production, use ffmpeg or similar
        logger.info(f"Video processing for {file_path} - using placeholder")
        
        metadata = {
            "mime_type": mime_type,
            "type": "video",
            "note": "Full video processing requires ffmpeg",
        }
        
        # TODO: Implement video frame extraction
        # For now, return basic metadata
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
        return {
            "embeddings": None,
            "category": "uncategorized",
            "metadata": metadata,
        }
    
    def get_embeddings_dim(self) -> int:
<<<<<<< HEAD
        """Embedding dimension from ResNet output"""
        return 2048
=======
        """Get dimension of embeddings"""
        if self.clip_model:
            return 512  # CLIP ViT-B/32 produces 512-dim embeddings
        return 0

>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
