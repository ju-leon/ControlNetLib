import cv2
import einops
import numpy as np
import random
import torch
import sys
from pytorch_lightning import seed_everything

sys.path.insert(1, 'ControlNet')
from share import *
import config
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import imageio
import numpy as np
import matplotlib.pyplot as plt

prompt = "mri brain scan"
num_samples = 1
image_resolution = 512
strength = 1.0
guess_mode = False
low_threshold =  50
high_threshold = 100
ddim_steps = 10
scale = 9.0
seed = 1
eta = 0.0
a_prompt = 'good quality' # 'best quality, extremely detailed'
n_prompt = 'animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

class ControlNet(torch.nn.Module):
    def __init__(self, model_path: str, seed: int):
        """
        Initialize ControlNet instance.

        Args:
            model_path (str): The path to the pre-trained ControlNet model.
        """
        super().__init__()

        self.model = create_model('/repo/ControlNet/models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict(model_path, location='cuda'))
        self.model = self.model.cuda()

        self.ddim_sampler = DDIMSampler(self.model)

        self.seed = seed

    def forward(self, img: torch.Tensor, promt: str) -> torch.Tensor:
        """
        Forward method for ControlNet inference.

        Args:
            img (torch.Tensor): The input image.
            promt (str): The prompt to control the image generation.

        Returns:
            torch.Tensor: The output image.
        """
        seed_everything(seed)

        # img = resize_image(HWC3(img), image_resolution)
        # H, W, C = img.shape

        # control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        # control = torch.stack([control for _ in range(num_samples)], dim=0)
        _, C, H, W = img.shape

        cond = {"c_concat": [img], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [img], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5) #.cpu().numpy().clip(0, 255).astype(np.uint8)

        return x_samples