# """
# Media Processor
# Generates embeddings via CLIP/ViT and performs semantic classification
# """
# import torch
# from torchvision import transforms
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import numpy as np
# from pathlib import Path
# import asyncio
# from typing import Dict, List, Optional
# import logging

# logger = logging.getLogger(__name__)

# class MediaProcessor:
#     """Processes media files (images/videos) to generate embeddings and classifications"""
    
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Initializing MediaProcessor on device: {self.device}")
        
#         # Load CLIP model for image embeddings
#         try:
#             self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#             self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#             self.clip_model.to(self.device)
#             self.clip_model.eval()
#             logger.info("CLIP model loaded successfully")
#         except Exception as e:
#             logger.warning(f"Could not load CLIP model: {e}. Using fallback.")
#             self.clip_model = None
#             self.clip_processor = None
        
#         # Predefined categories for classification
#         self.categories = [
#             "nature", "animals", "people", "architecture", "food",
#             "vehicles", "technology", "art", "sports", "travel",
#             "business", "medical", "education", "entertainment", "other"
#         ]
        
#         # Image preprocessing
#         self.image_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             ),
#         ])
    
#     async def process(self, file_path: Path, mime_type: str) -> Dict:
#         """
#         Process media file and return embeddings, category, and metadata
        
#         Args:
#             file_path: Path to media file
#             mime_type: MIME type of the file
            
#         Returns:
#             Dictionary with embeddings, category, and metadata
#         """
#         try:
#             if mime_type.startswith("image/"):
#                 return await self._process_image(file_path, mime_type)
#             elif mime_type.startswith("video/"):
#                 return await self._process_video(file_path, mime_type)
#             else:
#                 raise ValueError(f"Unsupported media type: {mime_type}")
#         except Exception as e:
#             logger.error(f"Error processing media {file_path}: {e}", exc_info=True)
#             # Return default values on error
#             return {
#                 "embeddings": None,
#                 "category": "uncategorized",
#                 "metadata": {"error": str(e)},
#             }
    
#     async def _process_image(self, file_path: Path, mime_type: str) -> Dict:
#         """Process image file"""
#         loop = asyncio.get_event_loop()
        
#         # Load image
#         image = Image.open(file_path).convert("RGB")
        
#         # Extract basic metadata
#         metadata = {
#             "width": image.width,
#             "height": image.height,
#             "format": image.format,
#             "mode": image.mode,
#             "mime_type": mime_type,
#         }
        
#         # Generate embeddings
#         embeddings = None
#         category = "uncategorized"
        
#         if self.clip_model and self.clip_processor:
#             try:
#                 # Generate CLIP embeddings
#                 inputs = self.clip_processor(
#                     images=image,
#                     return_tensors="pt",
#                     padding=True
#                 ).to(self.device)
                
#                 with torch.no_grad():
#                     image_features = self.clip_model.get_image_features(**inputs)
#                     embeddings = image_features.cpu().numpy().flatten().tolist()
                
#                 # Classify using text-image similarity
#                 category = await loop.run_in_executor(
#                     None, self._classify_image, image, embeddings
#                 )
                
#             except Exception as e:
#                 logger.warning(f"CLIP processing failed: {e}")
        
#         return {
#             "embeddings": embeddings,
#             "category": category,
#             "metadata": metadata,
#         }
    
#     def _classify_image(self, image: Image.Image, embeddings: List[float]) -> str:
#         """Classify image into category using CLIP"""
#         if not self.clip_model or not embeddings:
#             return "uncategorized"
        
#         try:
#             # Create text prompts for categories
#             category_texts = [f"a photo of {cat}" for cat in self.categories]
            
#             # Get text embeddings
#             text_inputs = self.clip_processor(
#                 text=category_texts,
#                 return_tensors="pt",
#                 padding=True
#             ).to(self.device)
            
#             with torch.no_grad():
#                 text_features = self.clip_model.get_text_features(**text_inputs)
#                 text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
#             # Compute similarity
#             image_emb = torch.tensor(embeddings).unsqueeze(0).to(self.device)
#             image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            
#             similarity = (image_emb @ text_features.T).cpu().numpy().flatten()
#             best_category_idx = np.argmax(similarity)
            
#             # Only return category if similarity is above threshold
#             if similarity[best_category_idx] > 0.2:
#                 return self.categories[best_category_idx]
#             else:
#                 return "uncategorized"
        
#         except Exception as e:
#             logger.warning(f"Classification failed: {e}")
#             return "uncategorized"
    
#     async def _process_video(self, file_path: Path, mime_type: str) -> Dict:
#         """Process video file (extract frame and process as image)"""
#         # For now, we'll extract a frame from the video
#         # In production, use ffmpeg or similar
#         logger.info(f"Video processing for {file_path} - using placeholder")
        
#         metadata = {
#             "mime_type": mime_type,
#             "type": "video",
#             "note": "Full video processing requires ffmpeg",
#         }
        
#         # TODO: Implement video frame extraction
#         # For now, return basic metadata
#         return {
#             "embeddings": None,
#             "category": "uncategorized",
#             "metadata": metadata,
#         }
    
#     def get_embeddings_dim(self) -> int:
#         """Get dimension of embeddings"""
#         if self.clip_model:
#             return 512  # CLIP ViT-B/32 produces 512-dim embeddings
#         return 0

