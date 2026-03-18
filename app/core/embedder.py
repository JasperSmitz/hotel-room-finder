from typing import List
import torch
import open_clip
from PIL import Image


class ImageEmbedder:
    def __init__(self, model_name: str, pretrained: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model_name = model_name
        self.pretrained = pretrained

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> List[float]:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().cpu().float().tolist()