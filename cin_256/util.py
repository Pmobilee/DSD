import os
import sys
import torch
import importlib
import numpy as np
from omegaconf import OmegaConf
from taming.models import vqgan 
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import *
import torch.nn as nn
import matplotlib.pyplot as plt

cwd = os.getcwd()

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(config_path, model_path):
    config = OmegaConf.load(config_path)  
    model = load_model_from_config(config, model_path)
    return model


@torch.no_grad()
def generate(model, sampler, num_imgs=1, steps=20, eta=0.0, scale=3.0, x_T=None, class_prompt=None):
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
                
                samples_ddim, _, x_T_copy = sampler.sample(S=steps,
                                                conditioning=c,
                                                batch_size=1,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                x_T=x_T,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=eta,
                                                keep_intermediates=False,
                                                intermediate_step=None,
                                                total_steps=None)
          
                                    
    # display as grid
    x_samples_ddim = model.decode_first_stage(_["x_inter"][-1])
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                min=0.0, max=1.0)


    grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image = Image.fromarray(grid.astype(np.uint8))

    return image, x_T_copy, class_prompt

@torch.no_grad()
def return_intermediates(model, sampler, steps=20, eta=0.0, scale=3.0):
    NUM_CLASSES = 1000
    ddim_steps = steps
    ddim_eta = eta
    scale = scale
    updates = int(ddim_steps / 2)
    intermediates = list()

    with torch.no_grad():
        with model.ema_scope():
                uc = model.get_learned_conditioning(
                        {model.cond_stage_key: torch.tensor(1*[1000]).to(model.device)}
                        )
                for class_prompt in torch.randint(0, NUM_CLASSES, (1,)):
                        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
                        xc = torch.tensor([class_prompt])
                        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                        x_T = None
                        for steps in range(updates):
                            samples_ddim, _, x_T_copy = sampler.sample(S=2,
                                                            conditioning=c,
                                                            batch_size=1,
                                                            shape=[3, 64, 64],
                                                            verbose=False,
                                                            x_T=x_T,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc, 
                                                            eta=ddim_eta,
                                                            keep_intermediates=True,
                                                            intermediate_step = steps*2,
                                                            steps_per_sampling = 2,
                                                            total_steps = ddim_steps)
                            if steps == 0:
                                starting_noise = x_T_copy
                            x_T = _["x_inter"][-1]
                            intermediates.append(x_T)

    return torch.stack(intermediates), starting_noise


def latent_to_img(model, latent):
    x_samples_ddim = model.decode_first_stage(latent)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
    grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image = Image.fromarray(grid.astype(np.uint8)) 
    return image

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')