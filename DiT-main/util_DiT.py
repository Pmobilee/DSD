import os
import sys
import torch
import importlib
import tqdm
import numpy as np
# from taming.models import vqgan 
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import wandb
import math
import traceback
from models import DiT_XL_2

from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

"""
This module is property of the Vrije Universiteit Amsterdam, department of Beta Science. It contains in part code
snippets obtained from Rombach et al., https://github.com/CompVis/latent-diffusion. No rights may be attributed.

The module presents both helper modules for loading, saving, generating from, and training of diffusion models, as
well as components for the process of knowledge distillation of teacher DDIMs into students, requiring fewer denoising
steps after every iteration, retaining original sampling quality at reduced computational expense.
"""



device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

# Receiving base current working directory
cwd = os.getcwd()

def save_model(model, optimizer, scheduler, name, steps, run_name):
    """
    Params: model, sampler, optimizer, scherduler, name, steps. Task: saves both the student and sampler models under 
    "/data/trained_models/{steps}/"
    """
    path = f"{cwd}/data/trained_models/{run_name}/{steps}/"
    if not os.path.exists(f"{cwd}/data/trained_models/{run_name}/"):
        os.mkdir(f"{cwd}/data/trained_models/{run_name}/")
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save({"model":model.module.state_dict(), "optimizer":optimizer, "scheduler":scheduler}, path + f"transformer_{name}.pt")


def load_trained(model_path):
    image_size = 256 #@param [256, 512]
    latent_size = int(image_size) // 8
    model = DiT_XL_2(input_size=latent_size).to(device)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.cuda()
    return model

def get_optimizer(sampler, iterations, lr=0.0000001):
    """
    Params: sampler, iterations, lr=1e-8. Task: 
    returns both an optimizer (Adam, lr=1e-8, eps=1e-08, decay=0.001), and a scheduler for the optimizer
    going from a learning rate of 1e-8 to 0 over the course of the specified iterations
    """
    lr = lr
    optimizer = torch.optim.Adam(sampler.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, last_epoch=-1, verbose=False)
    return optimizer, scheduler

def wandb_log(name, lr, model, tags, notes):
    """
    Params: wandb name, lr, model, wand tags, wandb notes. Task: returns a wandb session with CIFAR-1000 information,
    logs: Loss, Generational Loss, hardware specs, model gradients
    """
    session = wandb.init(
    project="diffusion-thesis", 
    name=name, 
    config={"learning_rate": lr, "architecture": "Diffusion Model","dataset": "CIFAR-1000"}, tags=tags, notes=notes)
    session.watch(model, log="all", log_freq=100)
    return session

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)

@torch.no_grad()
def sample_step(sample_fn, diffusion, step, model_kwargs, timesteps, samples):
    t = torch.tensor([timesteps[(step*2) +1]] * samples.shape[0], device="cuda")
    out = diffusion.p_mean_variance(sample_fn,samples,t,clip_denoised=False,denoised_fn=None,
        model_kwargs=model_kwargs)
    eps = diffusion._predict_eps_from_xstart(samples, t, out["pred_xstart"])
    alpha_bar = _extract_into_tensor(diffusion.alphas_cumprod, t, samples.shape)
    alpha_bar_prev = _extract_into_tensor(diffusion.alphas_cumprod_prev, t, samples.shape)
    sigma = (0.0 * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev))
    noise = torch.randn_like(samples)
    mean_pred = (
        out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
        + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
    )
    nonzero_mask = (
        (t != 0).float().view(-1, *([1] * (len(samples.shape) - 1)))
    )  # no noise when t == 0
    samples = mean_pred + nonzero_mask * sigma * noise
    pred_xstart = out["pred_xstart"]
    return samples, pred_xstart

@torch.enable_grad()
def sample_step_grad(sample_fn, diffusion, step, model_kwargs, timesteps, samples):
    t = torch.tensor([timesteps[(step*2) +1]] * samples.shape[0], device="cuda")
    out = diffusion.p_mean_variance_grad(sample_fn,samples,t,clip_denoised=False,denoised_fn=None,
        model_kwargs=model_kwargs)
    eps = diffusion._predict_eps_from_xstart(samples, t, out["pred_xstart"])
    alpha_bar = _extract_into_tensor(diffusion.alphas_cumprod, t, samples.shape)
    alpha_bar_prev = _extract_into_tensor(diffusion.alphas_cumprod_prev, t, samples.shape)
    sigma = (0.0 * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev))
    noise = torch.randn_like(samples)
    mean_pred = (
        out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
        + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
    )
    nonzero_mask = (
        (t != 0).float().view(-1, *([1] * (len(samples.shape) - 1)))
    )  # no noise when t == 0
    samples = mean_pred + nonzero_mask * sigma * noise
    pred_xstart = out["pred_xstart"]
    return samples, pred_xstart

@torch.enable_grad()
def internal_distill_loop(sample_fn, diffusion, model_kwargs, timesteps, 
                          optimizer, step, criterion, scheduler, losses, model, samples):
    
    with torch.enable_grad():
        optimizer.zero_grad()
        
        with autocast():
            samples, pred_xstart = sample_step_grad(sample_fn, diffusion, step, model_kwargs, timesteps, samples)
        samples.detach()
        with torch.no_grad():
            with autocast():
                samples, pred_xstart_second = sample_step(sample_fn, diffusion, step, model_kwargs, timesteps, samples)
            
        with torch.enable_grad():
            with autocast():
                loss = criterion(pred_xstart, pred_xstart_second.detach())
            # loss.backward()
            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            scheduler.step()
            losses.append(loss.item())
 
    return losses, samples
