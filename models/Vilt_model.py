import torch
from transformers import ViltProcessor, ViltForImageAndTextRetrieval, ViltConfig
from tqdm import tqdm
from utils import add_feature_extractor, make_path


class ViltModel:
    def __init__(self, device, dataset, random=False):
        self.device = device
        self.random = random
        self.dataset = dataset
        self.layers = [f'vilt.encoder.layer.{i}' for i in range(12)]
        self.model, self.preprocessor = self.load_model()
        self.save_path = "vilt_feats_random2/multi/" if random else "vilt_feats2/multi/"
        self.feature_dict = {}
        make_path(self.save_path)

    def load_model(self):
        if self.random:
            model = ViltForImageAndTextRetrieval(ViltConfig()).to(self.device)
        else:
            model = ViltForImageAndTextRetrieval.from_pretrained(
            "dandelin/vilt-b32-finetuned-coco", output_hidden_states=True).to(self.device)
        preprocess = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        model.eval()  # set model to evaluation mode
        model = add_feature_extractor(model, self.layers)
        return model, preprocess

    def extract_features(self):
        for idx in tqdm(range(len(self.dataset))):  # loop through dataset
            image, txt = self.dataset[idx]  # get image
            img_ft = self.preprocessor.feature_extractor(images=image, return_tensors="pt").to(self.device)
            encoding = self.preprocessor.tokenizer(
                txt, return_tensors="pt", padding=True, truncation=True).to(self.device) # encode text
            _ = self.model(**encoding, pixel_values=img_ft['pixel_values'],
                                 pixel_mask=img_ft['pixel_mask'],)  # get features with forward pass
            for k, v in self.model.__layer_activations__.items():  # loop through features
                if k in self.layers:  # only grab relevant layers
                    v = torch.squeeze(v)  # remove batch dimension
                    v = v[0, :]  # state of first token serves as the image representation (image transformer paper)
                    if k in self.feature_dict.keys():  # for all other images
                        self.feature_dict[k] = torch.cat((self.feature_dict[k], v))
                    else:
                        self.feature_dict[k] = v
        self.save_multi_features()

    def save_multi_features(self):
        for k, v in self.feature_dict.items():
            file_name = f"{self.save_path}{k}.pt"
            torch.save(v, file_name)





#self.layers = [n for s in [[f'vilt.encoder.layer.{i}',
#                            f'vilt.encoder.layer.{i}.attention',
#                            f'vilt.encoder.layer.{i}.attention.attention',
#                            f'vilt.encoder.layer.{i}.attention.output',
#                            f'vilt.encoder.layer.{i}.intermediate',
#                            f'vilt.encoder.layer.{i}.output'] for i in range(12)] for n in s] #+ \
                  #['vilt.layernorm', 'vilt.pooler.dense', 'vilt.pooler.activation',
                  # 'vilt.embeddings.text_embeddings.word_embeddings', 'vilt.embeddings.text_embeddings.position_embeddings',
                  # 'vilt.embeddings.text_embeddings.token_type_embeddings', 'vilt.embeddings.text_embeddings.LayerNorm',
                  # 'vilt.embeddings.text_embeddings.dropout', 'vilt.patch_embeddings.projection', 'vilt.embeddings.token_type_embeddings',
                  # 'vilt.embeddings.dropout']
