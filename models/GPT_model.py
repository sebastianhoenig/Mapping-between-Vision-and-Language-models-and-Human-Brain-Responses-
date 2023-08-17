import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from dataloader import StimuliDataset
from tqdm import tqdm
from utils import make_path
import os


class GPTModel:
    def __init__(self, device, dataset, random=False):
        self.random = random
        self.device = device
        self.dataset = dataset
        self.model = self.load_model()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.save_path = "gpt_feats_random/" if random else "gpt_feats/"
        self.feature_dict = {}
        make_path(self.save_path)

    def load_model(self):
        if self.random:
            config = GPT2Config()
            config.output_hidden_states = True
            model = GPT2Model(config).to(self.device)
        else:
            model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(self.device)
        return model

    def extract_text_features(self):
        prompts = self.dataset.get_all_prompts()
        for prompt in tqdm(prompts):
            # Tokenize and add EOS token
            prompt = prompt + " <|endoftext|>"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            hs = torch.stack(outputs.hidden_states)[1:, 0, -1, :]
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
            file_name = f"{self.save_path}GPTLayer{k}.pt"
            torch.save(v, file_name)



