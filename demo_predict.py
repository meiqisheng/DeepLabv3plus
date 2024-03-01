from torch.utils.data import dataset
import network
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 1 == 1:
    num_classes = 21
    decode_fn = VOCSegmentation.decode_target
else:
    num_classes = 19
    decode_fn = Cityscapes.decode_target

device = torch.device('cuda:0')

images = Image.open("samples/114_image.png").convert('RGB')
images = transform(images).unsqueeze(0) # To tensor of NCHW
images = images.to(device)

model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=21, output_stride=16)

model.load_state_dict(torch.load("model/best_deeplabv3plus_mobilenet_voc_os16.pth", map_location=torch.device('cpu'))["model_state"])
model = nn.DataParallel(model)

model.to(device)
#del checkpoint
model.eval()

outputs = model(images)
preds = outputs.max(1)[1].detach().cpu().numpy()
colorized_preds = decode_fn(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
# Do whatever you like here with the colorized segmentation maps
colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
colorized_preds.show()
