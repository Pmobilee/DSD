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
import util
import generate

# Receiving base current working directory
cwd = os.getcwd()

def save_model(sampler, optimizer, scheduler, name, steps, run_name):
    """
    Params: model, sampler, optimizer, scheduler, name, steps. Task: saves both the student and sampler models under 
    "/data/trained_models/{steps}/"
    """
    path = f"{cwd}/data/trained_models/{name}/{run_name}/"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({"model":sampler.model.state_dict(), "optimizer":optimizer, "scheduler":scheduler}, path + f"{steps}.pt")

def load_trained(model_path, config):
    """
    Params: model_path, config. Task: returns model, sampler, optimizer, scheduler for the provided model path and configuration
    """
    config = OmegaConf.load(config)  
    ckpt = torch.load(model_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    model.cuda()
    sampler = DDIMSampler(model)
    return model, sampler, ckpt["optimizer"], ckpt["scheduler"]

def get_optimizer(sampler, iterations, lr=0.000000003):
    """
    Params: sampler, iterations, lr=1e-8. Task: 
    returns both an optimizer (Adam, lr=1e-8, eps=1e-08, decay=0.001), and a scheduler for the optimizer
    going from a learning rate of 1e-8 to 0 over the course of the specified iterations
    """
    lr = lr
    # optimizer = torch.optim.Adam(sampler.model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.0005)
    optimizer = torch.optim.Adam(sampler.model.parameters(), lr=lr)#, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(sampler.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations,eta_min=lr *0.1, last_epoch=-1, verbose=False)
    return optimizer, scheduler

def wandb_log(name, lr, model, tags, notes, project="diffusion-thesis"):
    """
    Params: wandb name, lr, model, wand tags, wandb notes. Task: returns a wandb session with CIFAR-1000 information,
    logs: Loss, Generational Loss, hardware specs, model gradients
    """
    session = wandb.init(
    project=project, 
    name=name, 
    config={"learning_rate": lr, "architecture": "Diffusion Model","dataset": "Imagenet-1000"}, tags=tags, notes=notes)
    # session.watch(model, log="all", log_freq=1000)
    return session

def instantiate_from_config(config):
    """
    Params: model config file. Task: returns target model features for load_model_from_config()
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_model_from_config(config, ckpt):
    """
    Params: model config, model_location. Task: returns model with profided model configuration, used by get_model()
    """
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    try:
        sd = pl_sd["model"]
    except KeyError:
        sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(config_path, model_path):
    """
    Params: model configuration path, model path. Task: returns model with specified configuration
    """
    config = OmegaConf.load(config_path)  
    model = load_model_from_config(config, model_path)
    return model

def save_images(model, sampler, num_imgs, name, steps, verbose=False, celeb=False, total_steps=64, x_0=False):
    """
    Params: model, sampler, num_imgs, name, steps, verbose=False. Task: saves generated images to the specified folder name
    """
    basic_path = f"{cwd}/saved_images/"
    imgs_per_batch = num_imgs
    if not os.path.exists(basic_path + name + "/"):
        os.mkdir(basic_path + name + "/")
    
    for step in steps:
        num_imgs = imgs_per_batch
        new_path = basic_path + name + "/" + str(step) + "/"
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        items_present = len(os.listdir(new_path))
        if verbose:
            print(f"Folder {step} contains {items_present} images, generating {num_imgs-items_present} more images")
        if items_present >= num_imgs:
            if verbose:
                print(f"Folder already contains {num_imgs} images, skipping")
            continue
        num_imgs = num_imgs - items_present
        for i in tqdm.tqdm(range(num_imgs)):
            if celeb==False:
                image, _, class_prompt, _ = generate.generate_images(model, sampler, steps=step, x_0=x_0)
            else:
                image, _, class_prompt, _ = generate.generate_images_celeb(model, sampler, steps=step,x_0=x_0)
            image.save(new_path + str(class_prompt.item()) + "_" + str(i) + ".png")

@torch.no_grad()
def return_intermediates_for_student(model, sampler, steps=20, eta=0.0, scale=3.0):
    """
    Params: model, sampler, steps=20, eta=0.0, scale=3.0. Task: returns intermediate samples from the provided model and accompanying sampler.
    Has not been updated to work with the newest version of the code, as self-distillation does not require teacher intermediates.
    """
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
                            samples_ddim, _, x_T_copy, pred_x0 = sampler.sample(S=2,
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
                                intermediates.append(x_T_copy)

                            intermediates.append(pred_x0)

    return torch.stack(intermediates), starting_noise, class_prompt

@torch.no_grad()
def return_intermediates_for_student_celeb(model, sampler, steps=20, eta=0.0, scale=3.0):
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
                            samples_ddim, _, x_T_copy, pred_x0 = sampler.sample(S=2,
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
                                intermediates.append(x_T_copy)

                            intermediates.append(pred_x0)

    return torch.stack(intermediates), starting_noise, class_prompt

@torch.no_grad()
def make_dataset(model, sampler, num_images, sampling_steps, path, name):
    """
    Params: model, sampler, num_images, sampling_steps, path, name. Task: creates a dataset of generated images and saves it to the specified path.
    Only used for indirect self-distillation between a teacher and a student initialized from the teacher.
    """
    dataset = dict()
    if not os.path.exists(path):
        os.mkdir(path)
    for i in tqdm.tqdm(range(num_images)):
        new_dict = dict()
        intermediates, starting_noise, class_prompt = return_intermediates_for_student(model, sampler, steps=sampling_steps)
        new_dict["class"] = class_prompt
        new_dict["intermediates"] = intermediates
        dataset[str(i)] = new_dict
    
    new_path = path + f"{num_images}_" + name
    torch.save(dataset, new_path)
    del dataset
    dataset = dict()

@torch.no_grad()
def create_models(config_path, model_path, student=False):
    """
    Create a model and sampler from a config and model path.
    """
    model = get_model(config_path=config_path, model_path=model_path)
    sampler = DDIMSampler(model)
    if student == True:
        student = copy.deepcopy(model)
        sampler_student = DDIMSampler(student)
        return model, sampler, student, sampler_student
    else:
        return model, sampler