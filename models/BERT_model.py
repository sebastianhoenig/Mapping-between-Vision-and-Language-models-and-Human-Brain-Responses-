import torch
from transformers import AutoTokenizer, BertModel, BertConfig
from tqdm import tqdm
from utils import make_path
import os


class BERTModel:
    def __init__(self, device, dataset, random=False):
        self.random = random
        self.device = device
        self.dataset = dataset
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.save_path = "bert_feats_random/" if random else "bert_feats/"
        self.feature_dict = {}
        make_path(self.save_path)

    def load_model(self):
        if self.random:
            config = BertConfig()
            config.output_hidden_states = True
            model = BertModel(config).to(self.device)
        else:
            model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(self.device)
        return model

    def extract_text_features(self):
        prompts = self.dataset.get_all_prompts()
        for prompt in tqdm(prompts):
            inputs = self.tokenizer(prompt, return_tensors="pt")
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
        # make self.save_path if not exists
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for k, v in self.feature_dict.items():
            file_name = f"{self.save_path}BertLayer{k}.pt"
            torch.save(v, file_name)




