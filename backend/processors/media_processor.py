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
        
        # Map ImageNet labels to meaningful categories
        self._init_category_mapping()
    
    def _init_category_mapping(self):
        """Initialize mapping from ImageNet labels to meaningful categories"""
        # Map ImageNet labels to broader, more meaningful categories
        self.category_mapping = {
            # Animals
            "animals": ["dog", "cat", "bird", "horse", "cow", "elephant", "bear", "zebra", "giraffe", 
                       "sheep", "goat", "pig", "chicken", "duck", "rabbit", "mouse", "rat", "tiger",
                       "lion", "leopard", "fox", "wolf", "deer", "monkey", "ape", "panda", "kangaroo"],
            # Food
            "food": ["pizza", "burger", "sandwich", "hot dog", "ice cream", "cake", "bread", "pasta",
                    "soup", "salad", "fruit", "apple", "banana", "orange", "strawberry", "grape",
                    "coffee", "tea", "wine", "beer", "chocolate", "cookie", "donut"],
            # Vehicles
            "vehicles": ["car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane", "helicopter",
                        "boat", "ship", "submarine", "tank", "ambulance", "fire truck", "police car"],
            # Nature
            "nature": ["tree", "flower", "plant", "grass", "forest", "mountain", "beach", "ocean",
                      "river", "lake", "sunset", "sunrise", "cloud", "sky", "rainbow"],
            # People/Activities
            "people": ["person", "man", "woman", "child", "baby", "face", "hand", "foot"],
            "sports": ["soccer", "football", "basketball", "tennis", "baseball", "golf", "swimming",
                       "running", "cycling", "skiing", "surfing", "diving"],
            # Technology
            "technology": ["computer", "laptop", "phone", "tablet", "screen", "keyboard", "mouse",
                          "monitor", "television", "camera", "radio"],
            # Architecture
            "architecture": ["building", "house", "church", "tower", "bridge", "castle", "palace",
                           "skyscraper", "temple", "mosque", "library", "museum"],
            # Objects
            "furniture": ["chair", "table", "sofa", "bed", "desk", "cabinet", "shelf"],
            "clothing": ["shirt", "pants", "dress", "shoe", "hat", "jacket", "coat", "suit"],
        }
    
    def _map_to_category(self, imagenet_label: str) -> str:
        """Map ImageNet label to a meaningful category"""
        label_lower = imagenet_label.lower()
        
        # Check each category for matches
        for category, keywords in self.category_mapping.items():
            for keyword in keywords:
                if keyword in label_lower:
                    return category
        
        # If no match, try to extract meaningful word from label
        # ImageNet labels are often like "n02084071 dog" or "tabby cat"
        words = label_lower.split()
        if len(words) > 1:
            # Return the last meaningful word (usually the object name)
            return words[-1].replace("_", " ").title()
        
        # Fallback: return cleaned label
        return label_lower.replace("_", " ").title()
    
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
                imagenet_label = self.imagenet_labels[top_idx]
                # Map to meaningful category
                return self._map_to_category(imagenet_label)
        
        category = await loop.run_in_executor(None, predict_category)
        
        return {
            "embeddings": embeddings,
            "category": category,
            "metadata": metadata,
        }
    
    async def _process_video(self, file_path: Path, mime_type: str) -> Dict:
        """Process video file by extracting frames and analyzing them"""
        logger.info(f"Video processing for {file_path}")
        loop = asyncio.get_event_loop()
        
        def extract_video_frame():
            """Extract a frame from video for analysis"""
            try:
                # Try using imageio-ffmpeg or opencv
                try:
                    import imageio
                    import imageio_ffmpeg
                    reader = imageio.get_reader(file_path)
                    # Get frame from middle of video
                    num_frames = reader.count_frames()
                    if num_frames > 0:
                        frame_idx = num_frames // 2
                        frame = reader.get_data(frame_idx)
                        reader.close()
                        # Convert numpy array to PIL Image
                        frame_image = Image.fromarray(frame)
                        return frame_image, num_frames
                except ImportError:
                    pass
                
                # Fallback: try opencv
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(file_path))
                    if not cap.isOpened():
                        raise ValueError("Could not open video file")
                    
                    # Get video properties
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Extract frame from middle
                    if num_frames > 0:
                        frame_idx = num_frames // 2
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_image = Image.fromarray(frame_rgb)
                            return frame_image, num_frames
                except ImportError:
                    pass
                
                # If no video library available
                return None, 0
            except Exception as e:
                logger.warning(f"Error extracting video frame: {e}")
                return None, 0
        
        frame_image, num_frames = await loop.run_in_executor(None, extract_video_frame)
        
        metadata = {
            "mime_type": mime_type,
            "type": "video",
            "num_frames": num_frames,
            "file_size": file_path.stat().st_size,
        }
        
        # If we successfully extracted a frame, process it like an image
        if frame_image:
            try:
                # Get frame dimensions
                metadata["width"] = frame_image.width
                metadata["height"] = frame_image.height
                
                # Preprocess frame for ResNet
                image_tensor = self.image_transform(frame_image).unsqueeze(0).to(self.device)
                
                # Generate embeddings from frame
                with torch.no_grad():
                    features = self.embedding_model(image_tensor)
                    embeddings = features.cpu().numpy().flatten().tolist()
                
                # Predict category using ResNet classifier
                def predict_category():
                    with torch.no_grad():
                        outputs = self.classifier(image_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        top_idx = torch.argmax(probs, dim=1).item()
                        imagenet_label = self.imagenet_labels[top_idx]
                        # Map to meaningful category
                        return self._map_to_category(imagenet_label)
                
                category = await loop.run_in_executor(None, predict_category)
                
                return {
                    "embeddings": embeddings,
                    "category": category,
                    "metadata": metadata,
                }
            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
                return {
                    "embeddings": None,
                    "category": "uncategorized",
                    "metadata": {**metadata, "error": str(e)},
                }
        else:
            # No frame extracted - try to categorize from filename
            logger.warning(f"Could not extract frame from video {file_path}")
            category = self._categorize_from_filename(file_path)
            return {
                "embeddings": None,
                "category": category,
                "metadata": {**metadata, "note": "Frame extraction failed - install imageio-ffmpeg or opencv-python"},
            }
    
    def _categorize_from_filename(self, file_path: Path) -> str:
        """Categorize media file based on filename when other methods fail"""
        filename_lower = file_path.stem.lower()
        
        # Check filename for common patterns
        filename_keywords = {
            "video": ["video", "movie", "film", "clip", "recording", "footage"],
            "nature": ["nature", "landscape", "outdoor", "mountain", "beach", "forest"],
            "people": ["person", "people", "portrait", "selfie", "family", "friends"],
            "animals": ["pet", "dog", "cat", "animal", "wildlife"],
            "food": ["food", "meal", "cooking", "recipe", "restaurant"],
            "sports": ["sport", "game", "match", "training", "exercise"],
            "travel": ["travel", "trip", "vacation", "holiday", "tour"],
        }
        
        for category, keywords in filename_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return category
        
        return "video"
    
    def get_embeddings_dim(self) -> int:
        """Embedding dimension from ResNet output"""
        return 2048
