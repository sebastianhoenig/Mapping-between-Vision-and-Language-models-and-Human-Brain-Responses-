import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from tqdm import tqdm
from utils import add_feature_extractor, make_path
from ALBEF.models.model_retrieval import ALBEF


class ALBEFModel:
    def __init__(self, device, dataset, random=False):
        self.random = random
        self.device = device
        self.dataset = dataset
        self.img_layers = [f'visual_encoder.blocks.{i}' for i in range(12)]
        self.txt_layers = [f'text_encoder.encoder.layer.{i}' for i in range(6)]
        self.multi_layers = [f'text_encoder.encoder.layer.{i}' for i in range(6, 12)]
        self.model, self.preprocessor = self.load_model()
        self.img_feature_dict = {}
        self.txt_feature_dict = {}
        self.multi_feature_dict = {}
        self.save_path = "albef_feats_random/" if random else "albef_feats/"
        self.save_img_path = "albef_feats_random/img/" if random else"albef_feats/img/"
        self.save_text_path = "albef_feats_random/txt/" if random else"albef_feats/txt/"
        self.save_multi_path = "albef_feats_random/multi/" if random else"albef_feats/multi/"
        make_path(self.save_path)
        make_path(self.save_img_path)
        make_path(self.save_text_path)
        make_path(self.save_multi_path)

    def load_model(self):
        if self.random:
            config = dict(
                bert_config=('ALBEF/configs/config_bert.json'),
                bert_dir=('ALBEF/configs/bert/'),
                image_res=384,  # dimension
                queue_size=810,  # num images
                momentum=0.995,
                vision_width=768,
                embed_dim=256,
                temp=0.07,
                distill=False,
                checkpoint_vit=False,
                device=self.device,
            )
            model = ALBEF(config)
        else:
            config = dict(
                checkpoint_path=('ALBEF/configs/albef_mscoco.pth'),
                bert_config=('ALBEF/configs/config_bert.json'),
                bert_dir=('ALBEF/configs/bert/'),
                image_res=384,  # dimension
                queue_size=810,  # num images
                momentum=0.995,
                vision_width=768,
                embed_dim=256,
                temp=0.07,
                distill=False,
                checkpoint_vit=False,
                device=self.device,
            )
            model = ALBEF.from_cktp(config)
        model.to(self.device)
        model.eval()  # set model to evaluation mode
        preprocess = Compose([
            Resize(
                (config['image_res'], config['image_res']),
                interpolation=InterpolationMode.BICUBIC
            ),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            ),
        ])
        layers = self.img_layers + self.txt_layers + self.multi_layers
        model = add_feature_extractor(model, layers)
        return model, preprocess

    def extract_features(self):
        with open(self.save_path + 'results.txt', 'w') as f:
            for idx in tqdm(range(len(self.dataset))):  # loop through dataset
                with torch.no_grad():
                    image, txt = self.dataset[idx]
                    tensor_image = self.preprocessor(image).unsqueeze(0).to(self.device)
                    self._extract_features(tensor_image, txt)
            self.save_img_features()
            self.save_text_multi_features()

    def _extract_features(self, tensor_image, txt):
        txt_features = self.model.tokenizer(txt, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            _ = self.model.similarity_and_matching(tensor_image, txt_features, pairwise=True)
            for k, v in self.model.__layer_activations__.items():  # loop through features
                if k in self.img_layers:  # only grab relevant layers
                    v = torch.squeeze(v)  # remove batch dimension
                    v = v[0, :]  # get cls token
                    if k in self.img_feature_dict.keys():  # for all other images
                        self.img_feature_dict[k] = torch.cat((self.img_feature_dict[k], v))
                    else:
                        self.img_feature_dict[k] = v  # for first image
            for k, v in self.model.__layer_activations__.items():  # loop through features
                if k in self.txt_layers:  # only grab relevant layers
                    v = torch.squeeze(v)  # remove batch dimension
                    v = v[0, :]  # get first token
                    if k in self.txt_feature_dict.keys():  # for all other images
                        self.txt_feature_dict[k] = torch.cat((self.txt_feature_dict[k], v))
                    else:
                        self.txt_feature_dict[k] = v
            for k, v in self.model.__layer_activations__.items():  # loop through features
                if k in self.multi_layers:
                    v = torch.squeeze(v)
                    v = v[0, :]
                    if k in self.multi_feature_dict.keys():
                        self.multi_feature_dict[k] = torch.cat((self.multi_feature_dict[k], v))
                    else:
                        self.multi_feature_dict[k] = v

    def save_img_features(self):
        for k, v in self.img_feature_dict.items():
            file_name = f"{self.save_img_path}{k}.pt"
            torch.save(v, file_name)

    def save_text_multi_features(self):
        for k, v in self.txt_feature_dict.items():
            file_name = f"{self.save_text_path}{k}.pt"
            torch.save(v, file_name)
        for k, v in self.multi_feature_dict.items():
            file_name = f"{self.save_multi_path}{k}.pt"
            torch.save(v, file_name)


# self.img_layers = [n for s in [[f'visual_encoder.blocks.{i}',
#                                f'visual_encoder.blocks.{i}.attn',
#                                f'visual_encoder.blocks.{i}.mlp'] for i in range(12)] for n in s]
# self.txt_layers = [n for s in [[f'text_encoder.encoder.layer.{i}',
#                                f'text_encoder.encoder.layer.{i}.attention',
#                                f'text_encoder.encoder.layer.{i}.attention.self',
#                                f'text_encoder.encoder.layer.{i}.attention.output',
#                                f'text_encoder.encoder.layer.{i}.intermediate',
#                                f'text_encoder.encoder.layer.{i}.output'] for i in range(6)] for n in s]
# self.multi_layers = [n for s in [[f'text_encoder.encoder.layer.{i}',
#                                  f'text_encoder.encoder.layer.{i}.attention',
#                                  f'text_encoder.encoder.layer.{i}.attention.self',
#                                  f'text_encoder.encoder.layer.{i}.attention.output',
#                                  f'text_encoder.encoder.layer.{i}.crossattention',
#                                  f'text_encoder.encoder.layer.{i}.crossattention.self',
#                                  f'text_encoder.encoder.layer.{i}.crossattention.output',
#                                  f'text_encoder.encoder.layer.{i}.intermediate',
#                                  f'text_encoder.encoder.layer.{i}.output'] for i in range(6, 12)] for n in s]
