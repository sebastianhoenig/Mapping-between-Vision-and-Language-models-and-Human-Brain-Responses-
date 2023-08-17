from PIL import Image
from torch.utils.data import Dataset
import os
import natsort
import glob
import re
from utils import _get_prompt, sort_list_by_last_word


class StimuliDataset(Dataset):
    def __init__(self, directory, device):
        self.device = device
        self.main_dir = directory # directory where images are stored
        all_imgs = glob.glob(os.path.join(directory, "*"))
        all_imgs = [os.path.basename(x) for x in all_imgs]
        self.total_imgs = natsort.natsorted(all_imgs)
        self.all_classes = self.make_classes() # get classes from dataset

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx]) # get image path
        obj_name = re.search(r'\/(\w+)(?=\d{3}\.png$)', img_loc).group(1)
        txt = _get_prompt(obj_name)
        image = Image.open(img_loc).convert("RGB")  # transform image to RGB
        return image, txt

    def make_classes(self):
        classes = []
        for i in range(len(self.total_imgs)):
            classes.append(self.total_imgs[i][:-7])
        return list(set(classes)) # get unique classes
    
    def get_all_prompts(self):
        prompts = []
        for cls in self.all_classes:
            prompts += _get_prompt(cls)
        prompts = sort_list_by_last_word(prompts)
        return prompts
    
    def get_img(self, idx):
        return self.total_imgs[idx]


class AlgonautsDataset(Dataset):
    def __init__(self, directory, device):
        self.device = device
        self.main_dir = directory # directory where images are stored
        all_imgs = glob.glob(os.path.join(directory, "*"))
        all_imgs = [os.path.basename(x) for x in all_imgs]
        # only if img is png
        all_imgs = [img for img in all_imgs if img.endswith(".png")]
        self.total_imgs = sorted(all_imgs)
        self.caption_dict = self.load_dict_from_txt(os.path.join(directory, "captions.txt"))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx]) # get image path
        image = Image.open(img_loc).convert("RGB") # transform image to RGB
        prompt = self.caption_dict[self.total_imgs[idx][:-4]]
        return image, prompt

    def get_all_prompts(self):
        l = []
        for img in self.total_imgs:
            l.append(self.caption_dict[img[:-4]])
        return l

    def load_dict_from_txt(self, file_path):
        data_dict = {}
        with open(file_path, 'r') as file:
            for line in file:
                if line == '\n':
                    continue
                key, value = line.strip().split(': ')
                data_dict[key] = value
        return data_dict
