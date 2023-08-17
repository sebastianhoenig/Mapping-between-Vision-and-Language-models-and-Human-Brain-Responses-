from collections import OrderedDict
import torch
import torch.nn.init as init
from tqdm import tqdm
import clip
from utils import add_feature_extractor, make_path
from clip.model import build_model


class ClipModel:
    def __init__(self, device, dataset, random=False):
        self.device = device
        self.dataset = dataset
        self.random = random
        self.img_layers = [f'visual.transformer.resblocks.{i}' for i in range(12)]
        self.txt_layers = [f'transformer.resblocks.{i}' for i in range(12)]
        self.model, self.preprocess = self.load_model()
        self.img_feature_dict = {}
        self.txt_feature_dict = {}
        self.save_path = "clip_feats_random/" if random else "clip_feats/"
        self.safe_path_img = "clip_feats_random/img/" if random else"clip_feats/img/"
        self.safe_path_txt = "clip_feats_random/txt/" if random else "clip_feats/txt/"
        make_path(self.save_path)
        make_path(self.safe_path_img)
        make_path(self.safe_path_txt)

    def load_model(self):
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        if self.random:
            model = self.randomize_weights(model)
        model.eval()  # set model to evaluation mode
        layers = self.img_layers + self.txt_layers
        model = add_feature_extractor(model, layers)
        return model, preprocess

    def randomize_weights(self, model):
        weights = model.state_dict()

        for key in weights:
            if weights[key].dim() < 2:
                continue
            if key.endswith("weight"):
                init_method = init.xavier_normal_
            elif key.endswith("bias"):
                init_method = init.zeros_
            else:
                continue

            init_method(weights[key])

        r_model = build_model(weights).to(self.device)
        r_model.float()
        return r_model

    def extract_features(self):
        self.extract_image_features()
        self.extract_text_features()
        self.save_features()

    def extract_image_features(self):
        for idx in tqdm(range(len(self.dataset))):  # loop through dataset
            with torch.no_grad():
                self.model.__features__ = OrderedDict()
                image, _ = self.dataset[idx]  # get image
                tensor_image = self.preprocess(image).unsqueeze(0).to(self.device)  # apply transform
                _ = self.model.encode_image(tensor_image)  # encode image
                for k, v in self.model.__layer_activations__.items():  # loop through features
                    if k in self.img_layers: # only grab relevant layers
                        v = v[0] # state of first token serves as the image representation (image transformer paper)
                        v = torch.squeeze(v)
                        if k in self.img_feature_dict.keys():  # for all other images
                            self.img_feature_dict[k] = torch.cat((self.img_feature_dict[k], v), dim=0)
                        else:
                            self.img_feature_dict[k] = v  # for first image

    def extract_text_features(self):
        prompts = self.dataset.get_all_prompts()

        tokens = clip.tokenize(prompts).to(self.device)
        # 49407 is the eot token. It represents the end of the text sequence. It is a special token that carries
        # important contextual information for the model during text encoding and processing.
        tokens_idxs = (tokens == torch.tensor(49407)).nonzero()
        self.model.__features__ = OrderedDict()
        with torch.no_grad():
            txt_ft = self.model.encode_text(tokens)
        for k, v in self.model.__layer_activations__.items():  # loop through features
            if k in self.txt_layers:
                # ensure shape is (77, 81, 512)
                current_shape = v.size()
                if current_shape == (77, 81, 512):
                    pass
                elif current_shape == (81, 77, 512):
                    v = v.permute(1, 0, 2)
                a = v[tokens_idxs[:, 1], tokens_idxs[:, 0], :]
                self.txt_feature_dict[k] = a
        return txt_ft

    def save_features(self):
        for k, v in self.img_feature_dict.items():
            file_name = f"{self.safe_path_img}{k}.pt"
            torch.save(v, file_name)
        for k, v in self.txt_feature_dict.items():
            file_name = f"{self.safe_path_txt}{k}.pt"
            torch.save(v, file_name)


# self.txt_layers = [n for s in [[f'transformer.resblocks.{i}',
#                                f'transformer.resblocks.{i}.attn',
#                                f'transformer.resblocks.{i}.attn.out_proj',
#                                f'transformer.resblocks.{i}.ln_1',
#                                f'transformer.resblocks.{i}.mlp',
#                                f'transformer.resblocks.{i}.mpl.c_fc',
#                                f'transformer.resblocks.{i}.mlp.gelu',
#                                f'transformer.resblocks.{i}.mlp.c_proj',
#                                f'transformer.resblocks.{i}.ln_2'] for i in range(12)] for n in s] + \
#                    ['transformer', 'token_embedding', 'ln_final']

# self.img_layers = [n for s in [[f'visual.transformer.resblocks.{i}',
#                                f'visual.transformer.resblocks.{i}.attn',
#                                f'visual.transformer.resblocks.{i}.attn.out_proj',
#                                f'visual.transformer.resblocks.{i}.ln_1',
#                                f'visual.transformer.resblocks.{i}.mlp',
#                                f'visual.transformer.resblocks.{i}.mpl.c_fc',
#                                f'visual.transformer.resblocks.{i}.mlp.gelu',
#                                f'visual.transformer.resblocks.{i}.mlp.c_proj',
#                                f'visual.transformer.resblocks.{i}.ln_2'] for i in range(12)] for n in s] + \
#                  ['visual.ln_post', 'visual', 'visual.conv1, visual.ln_pre', 'visual.transformer', 'visual.ln_post']

