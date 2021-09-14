"""
this file is for creating a dataset of _translated images_ using an image translator,
namely munit, unit, or drit. The dataset is replicated, and then for every image in the
simulated dataset it's translated to the "real" domain and then saved, overwritten.
The original dataset is specified in the config file, datasetA. That dataset
is cloned and then modified.
"""
import os
import sys
from data.dataset import DatasetConverter
from trainers import MUNIT_Trainer, UNIT_Trainer, Baseline_Trainer, ADA_Trainer
from utils import prepare_sub_folder, write_loss, get_config, write_2images, Timer
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/tusimple/munit_s2r.yaml', help='Path to the config file.')
parser.add_argument('--new_data_folder', type=str, help='root folder to generate the new dataset.')
parser.add_argument('--checkpoint_dir', type=str, help='directory containing model weights, in .pt files')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
opts = parser.parse_args()

config = get_config(opts.config)
config['vgg_model_path'] = opts.output_path

# clone the dataset folder

dataset_root = config["dataA_root"]
new_dataset = opts.new_data_folder

# print("making a copy of the dataset")
# os.system(f"cp -r {dataset_root} {new_dataset}")
# print("done copying the dataset")

# Setup model

print(f"Loading {config['trainer']} trainer")
if config['trainer'] == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif config['trainer'] == 'UNIT':
    trainer = UNIT_Trainer(config)
elif config['trainer'] == 'Baseline':
    trainer = Baseline_Trainer(config)
elif config['trainer'] == 'ADA':
    trainer = ADA_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT|Baseline|ADA")
# Initialize dataset

trainer.resume(opts.checkpoint_dir, config)
trainer.cuda()

image_dim = (config["input_height"], config["input_width"])

img_transform = transforms.Compose([
    transforms.Resize(image_dim),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def get_tusimple_row_anchor(image_height):
    return [int((160+i*10)/720*image_height) for i in range(56)]

dataset_to_translate = DatasetConverter(new_dataset,
                               os.path.join(new_dataset, 'list/train_gt.txt'),
                               img_transform=img_transform,
                               image_dim=image_dim,
                                row_anchor=get_tusimple_row_anchor(image_dim[0]),
                               return_label=False)

iterator = torch.utils.data.DataLoader(dataset_to_translate)

def unnormalize(image):
    # multiply by the previous stddev, then add add the mean back
    image.squeeze()
    image = image * (0.229, 0.224, 0.225)
    image = image + (0.485, 0.456, 0.406)
    return image

print("starting conversion")
with torch.no_grad():
    for el in iterator:
        image, label, image_path = el
        sim2real, real2sim = trainer.forward(image.cuda(), image.cuda())
        image_tensor = unnormalize(sim2real)
        save_image(image_tensor.cpu(), image_path)  # overwrite



