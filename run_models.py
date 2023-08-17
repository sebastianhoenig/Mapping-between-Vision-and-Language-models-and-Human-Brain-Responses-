from dataloader import AlgonautsDataset
import sys
from ALBEF_model import ALBEFModel
from models.ViT_model import VitModel
from models.BERT_model import BERTModel
from models.GPT_model import GPTModel
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = AlgonautsDataset("stimuli/shared_img", device)
    #clip_model = ClipModel(device, dataset)
    albef_model = ALBEFModel(device, dataset)
    #vilt_model = ViltModel(device, dataset)
    #clip_model_r = ClipModel(device, dataset, random=True)
    #albef_model_r = ALBEFModel(device, dataset, random=True)
    #vilt_model_r = ViltModel(device, dataset, random=True)
    #vit_model = VitModel(device, dataset, random=True)
    #bert_model = BERTModel(device, dataset, random=True)
    #gpt_model = GPTModel(device, dataset, random=True)

    print("Extracting features from CLIP model...")
    #clip_model.extract_features()
    print("Extracting features from ALBEF model...")
    albef_model.extract_features()
    print("Extracting features from Vilt model...")
    #vilt_model.extract_features()

    print("Extracting features from CLIP R model...")
    #clip_model_r.extract_features()
    print("Extracting features from ALBEF R model...")
    #albef_model_r.extract_features()
    print("Extracting features from Vilt R model...")
    #vilt_model_r.extract_features()

    print("Extracting features from ViT model...")
    #vit_model.extract_image_features()
    print("Extracting features from Bert model...")
    #bert_model.extract_text_features()
    print("Extracting features from GPT model...")
    #gpt_model.extract_text_features()


if __name__ == "__main__":
    main()
