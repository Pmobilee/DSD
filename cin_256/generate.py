import os
import sys
import torch
import importlib
import tqdm
import numpy as np
from omegaconf import OmegaConf
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import *
import torch.nn as nn
# import matplotlib.pyplot as plt
import copy
import wandb
import math
import traceback
from pytorch_fid import fid_score
import shutil
from self_distillation import *
from distillation import *
from saving_loading import *
from util import *



@torch.no_grad()
def generate_images(model, sampler, num_imgs=1, steps=20, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False):
    """
    Params: model, sampler, num_imgs=1, steps=20, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False. 
    Task: returns final generated samples from the provided model and accompanying sampler. Unless the class prompt is specified,
    all generated images are of one of the random classes. Pred_x0 and samples_ddim are identical when the final denoising step is returned.
    """
    NUM_CLASSES = 1000
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)

    if class_prompt == None:
        class_prompt = torch.randint(0, NUM_CLASSES, (num_imgs,))

    with torch.no_grad():
        with model.ema_scope():
                uc = model.get_learned_conditioning(
                        {model.cond_stage_key: torch.tensor(num_imgs*[1000]).to(model.device)}
                        )
                
                
                xc = torch.tensor(num_imgs*[class_prompt])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _, x_T_copy, pred_x0, a_t = sampler.sample(S=steps,
                                                conditioning=c,
                                                batch_size=1,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                x_T=x_T,
                                             
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=eta,
                                                keep_intermediates=keep_intermediates,
                                                intermediate_step=None,
                                                total_steps=steps,
                                                steps_per_sampling=steps)
          
                                    
    # display as grid
    x_samples_ddim = model.decode_first_stage(pred_x0)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                min=0.0, max=1.0)


    grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image = Image.fromarray(grid.astype(np.uint8))

    return image, x_T_copy, class_prompt, _["x_inter"]

@torch.no_grad()
def generate_images_celeb(model, sampler, num_imgs=1, steps=20, total_steps=64, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False):
    """
    Params: model, sampler, num_imgs=1, steps=20, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False. 
    Task: returns final generated samples from the provided model and accompanying sampler. Unless the class prompt is specified,
    all generated images are of one of the random classes.
    """
    NUM_CLASSES = 1000
    sampler.make_schedule(ddim_num_steps=total_steps, ddim_eta=eta, verbose=False)

    if class_prompt == None:
        class_prompt = torch.randint(0, NUM_CLASSES, (num_imgs,))

    with torch.no_grad():
        with model.ema_scope():
         
                
                
                samples_ddim, _, x_T_copy, pred_x0, a_t = sampler.sample(S=steps,
                                               
                                                batch_size=1,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                x_T=x_T,
                                             
                                                unconditional_guidance_scale=scale,
                                                eta=eta,
                                                keep_intermediates=keep_intermediates,
                                                intermediate_step=None,
                                                total_steps=total_steps,
                                                steps_per_sampling=steps)
          
                                    
    # display as grid
    x_samples_ddim = model.decode_first_stage(_["x_inter"][-1])
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                min=0.0, max=1.0)


    grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image = Image.fromarray(grid.astype(np.uint8))

    return image, x_T_copy, class_prompt, _["x_inter"]