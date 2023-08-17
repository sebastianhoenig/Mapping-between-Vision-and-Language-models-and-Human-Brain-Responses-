import torch
from transformers import ViTImageProcessor, ViTModel, ViTConfig
from dataloader import StimuliDataset
from tqdm import tqdm
from utils import make_path
import os


class VitModel:
    def __init__(self, device, dataset, random=False):
        self.random = random
        self.device = device
        self.dataset = dataset
        self.model = self.load_model()
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch32-224-in21k")
        self.save_path = "vit_feats_random/" if random else "vit_feats/"
        self.feature_dict = {}
        make_path(self.save_path)

    def load_model(self):
        if self.random:
            config = ViTConfig()
            config.output_hidden_states = True
            config.patch_size = 32
            model = ViTModel(config).to(self.device)
        else:
            model = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k", output_hidden_states=True).to(self.device)
        return model

    def extract_image_features(self):
        for idx in tqdm(range(len(self.dataset))):
            image, _ = self.dataset[idx]  # get image
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            hs = torch.stack(outputs.hidden_states)[1:, :, 0, :]
            for i in range(12):
                v = hs[i].squeeze()
                if i in self.feature_dict.keys():
                    self.feature_dict[i] = torch.cat((self.feature_dict[i], v), dim=0)
                else:
                    self.feature_dict[i] = v
        self.save_features()

    def save_features(self):
        for k, v in self.feature_dict.items():
            file_name = f"{self.save_path}ViTLayer{k}.pt"
            torch.save(v, file_name)




