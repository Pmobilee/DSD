import os
import sys
import torch
import importlib
import tqdm
import numpy as np
from omegaconf import OmegaConf
# from taming.models import vqgan 
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import *
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import wandb
import math
import traceback
from pytorch_fid import fid_score
import shutil

"""
This module is property of the Vrije Universiteit Amsterdam, department of Beta Science. It contains in part code
snippets obtained from Rombach et al., https://github.com/CompVis/latent-diffusion. No rights may be attributed.

The module presents both helper modules for loading, saving, generating from, and training of diffusion models, as
well as components for the process of knowledge distillation of teacher DDIMs into students, requiring fewer denoising
steps after every iteration, retaining original sampling quality at reduced computational expense.
"""





# Receiving base current working directory
cwd = os.getcwd()

def save_model(sampler, optimizer, scheduler, name, steps, run_name):
    """
    Params: model, sampler, optimizer, scheduler, name, steps. Task: saves both the student and sampler models under 
    "/data/trained_models/{steps}/"
    """
    path = f"{cwd}/data/trained_models/{run_name}/{steps}/"
    if not os.path.exists(f"{cwd}/data/trained_models/{run_name}/"):
        os.mkdir(f"{cwd}/data/trained_models/{run_name}/")
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save({"model":sampler.model.state_dict(), "optimizer":optimizer, "scheduler":scheduler}, path + f"student_{name}.pt")


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

def get_optimizer(sampler, iterations, lr=0.0000001):
    """
    Params: sampler, iterations, lr=1e-8. Task: 
    returns both an optimizer (Adam, lr=1e-8, eps=1e-08, decay=0.001), and a scheduler for the optimizer
    going from a learning rate of 1e-8 to 0 over the course of the specified iterations
    """
    lr = lr
    optimizer = torch.optim.Adam(sampler.model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.001)
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


@torch.no_grad()
def generate(model, sampler, num_imgs=1, steps=20, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False):
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
                                                total_steps=None)
          
                                    
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


def save_images(model, sampler, num_imgs, name, steps, verbose=False):
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
                print("Folder already contains 50000 images, skipping")
            continue
        num_imgs = num_imgs - items_present
        for i in range(num_imgs):
            image, _, class_prompt, _ = generate(model, sampler, steps=step)
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


def latent_to_img(model, latent):
    """
    Params: model, latent. Task: converts a latent vector to an image
    """
    x_samples_ddim = model.decode_first_stage(latent)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
    grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image = Image.fromarray(grid.astype(np.uint8)) 
    return image

def print_size_of_model(model):
    """
    Params: model. Task: prints the size of the model in MB
    """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

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
    

def teacher_train_student(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test"):
    """
    Params: teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test". 
    Task: trains the student model using the identical teacher model as a guide. Not used in direct self-distillation where a teacher distills into itself.
    """
    NUM_CLASSES = 1000
    generations = generations
    intermediate_generation_save = generations // 2
    intermediate_generation_compare = generations // 4

    ddim_steps_teacher = steps
    ddim_steps_student = int(ddim_steps_teacher / 2)
    TEACHER_STEPS = 2
    STUDENT_STEPS = 1
    ddim_eta = 0.0
    scale = 3.0
    updates = int(ddim_steps_teacher / TEACHER_STEPS)
    optimizer=optimizer
    averaged_losses = []
    teacher_samples = list()
    criterion = nn.MSELoss()
    instance = 0
    generation = 0
    
    # if session != None:
    #     session.log({"ddim_eta":ddim_eta})

    a_t = np.linspace(0, 1, updates)

    all_losses = []
    ets = []
    with torch.no_grad():
        with student.ema_scope():
                
                sampler_teacher.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
                sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                # for class_prompt in tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))):
                uc = teacher.get_learned_conditioning(
                            {teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)}
                            )
                sc = teacher.get_learned_conditioning(
                            {student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)}
                            )
                with tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))) as tepoch:
                    for i, class_prompt in enumerate(tepoch):
                        
            

                        generation += 1
                        losses = []        
                        xc = torch.tensor([class_prompt])
                        c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                        c_student = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                        
                        samples_ddim_teacher = None
                        predictions_temp = []
                        for steps in range(updates):          
                                    instance += 1

                                    samples_ddim_teacher, teacher_intermediate, x_T, pred_x0_teacher, a_t_teacher = sampler_teacher.sample(S=TEACHER_STEPS,
                                                                    conditioning=c,
                                                                    batch_size=1,
                                                                    shape=[3, 64, 64],
                                                                    verbose=False,
                                                                    x_T=samples_ddim_teacher,
                                                                    # quantize_x0 = True,
                                                                    unconditional_guidance_scale=scale,
                                                                    unconditional_conditioning=uc, 
                                                                    eta=ddim_eta,
                                                                    keep_intermediates=False,
                                                                    intermediate_step = steps*2,
                                                                    steps_per_sampling = TEACHER_STEPS,
                                                                    total_steps = ddim_steps_teacher)      
                                    
                                    

                                    # x_T_teacher_decode = sampler_teacher.model.decode_first_stage(pred_x0_teacher)
                                    # teacher_target = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                    

                                    with torch.enable_grad():
                                        # sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                                        optimizer.zero_grad()
                                        samples, pred_x0_student, st, at = sampler_student.sample_student(S=STUDENT_STEPS,
                                                                        conditioning=c_student,
                                                                        batch_size=1,
                                                                        shape=[3, 64, 64],
                                                                        verbose=False,
                                                                        x_T=x_T,
                                                                        # quantize_x0 = True,
                                                                        unconditional_guidance_scale=scale,
                                                                        unconditional_conditioning=sc, 
                                                                        eta=ddim_eta,
                                                                        keep_intermediates=False,
                                                                        intermediate_step = steps,
                                                                        steps_per_sampling = STUDENT_STEPS,
                                                                        total_steps = ddim_steps_student)
                                        
                  
                                        # x_T_student_decode = sampler_student.model.differentiable_decode_first_stage(pred_x0_student)
                                        # student_target  = torch.clamp((x_T_student_decode +1.0)/2.0, min=0.0, max=1.0)
                            
                                        signal = at
                                        noise = 1 - at
                                        log_snr = torch.log(signal / noise)
                                        weight = max(log_snr, 1)
                                        # loss = torch.exp(torch.log(((at  ) / (st )))) * criterion(pred_x0_student, pred_x0_teacher)
                                        loss = weight * criterion(pred_x0_student, pred_x0_teacher)
         
                                   
                                        loss.backward()
                                        
                                        
                                        torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                        
                                        optimizer.step()
                                        scheduler.step()
                                        losses.append(loss.item())
                                        
                                        if session != None:
                                            if generation > 0 and generation % intermediate_generation_compare == 0:
                                                x_T_teacher_decode = sampler_teacher.model.decode_first_stage(pred_x0_teacher.detach())
                                                teacher_target = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                                x_T_student_decode = sampler_teacher.model.decode_first_stage(pred_x0_student.detach())
                                                student_target  = torch.clamp((x_T_student_decode +1.0)/2.0, min=0.0, max=1.0)
                                                predictions_temp.append(teacher_target)
                                                predictions_temp.append(student_target)
                                            session.log({"intermediate_loss":loss.item()})
                                        

                                
                        
                        if session != None:
                            with torch.no_grad():
                                if generation > 0 and generation % intermediate_generation_compare == 0 and session !=None:
                                    img, grid = compare_latents(predictions_temp)
                                    images = wandb.Image(grid, caption="left: Teacher, right: Student")
                                    wandb.log({"Inter_Comp": images})
                                    del img, grid, predictions_temp, x_T_student_decode, x_T_teacher_decode, student_target, teacher_target
                                    torch.cuda.empty_cache()
                                
                            
                        
                        
                        all_losses.extend(losses)
                        # print(scheduler.get_last_lr())
                        averaged_losses.append(sum(losses) / len(losses))
                        if session != None:
                            session.log({"generation_loss":averaged_losses[-1]})
                        tepoch.set_postfix(epoch_loss=averaged_losses[-1])
                        torch.cuda.empty_cache()
                        if early_stop == True and i > 1:
                            if averaged_losses[-1] > (10*losses[-2]):
                                print(f"Early stop initiated: Prev loss: {round(averaged_losses[-2], 5)}, Current loss: {round(averaged_losses[-1], 5)}")
                                plt.plot(range(len(averaged_losses)), averaged_losses, label="MSE LOSS")
                                plt.xlabel("Generations")
                                plt.ylabel("px MSE")
                                plt.title("MSEloss student vs teacher")
                                plt.show()
                                 

                                                            
    plt.plot(range(len(all_losses)), all_losses, label="MSE LOSS")
    plt.xlabel("Generations")
    plt.ylabel("px MSE")
    plt.title("MSEloss student vs teacher")
    plt.show()

@torch.enable_grad()
def train_student_from_dataset(model, sampler, dataset, student_steps, optimizer, scheduler, early_stop=False, session=None, run_name="test"):
    """
    Train a student model from a pre-generated dataset.
    """
    device = torch.device("cuda")
    model.requires_grad=True
    sampler.requires_grad=True
    for param in sampler.model.parameters():
        param.requires_grad = True

    for param in model.model.parameters():
        param.requires_grad = True
    MSEloss = model.criterion
    ddim_steps_student = student_steps
    STUDENT_STEPS = 1
    ddim_eta = 0.0
    scale = 3.0

    averaged_losses = []
    teacher_samples = list()
    criterion = nn.MSELoss()
    optimizer = optimizer
    generation = 0
    instance = 0
    with torch.no_grad():
        
        with model.ema_scope():
            uc = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(1*[1000]).to(model.device)})

            with tqdm.tqdm(range(len(dataset))) as tepoch:
                    for i, _ in enumerate(tepoch):
                        class_prompt = dataset[str(i)]["class"]
                        losses = []
                        sampler.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                        xc = torch.tensor([class_prompt])
                        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                        sampler.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                        c_student = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                        generation += 1
                        for steps, x_T in enumerate(dataset[str(i)]["intermediates"]):
                            instance += 0
                            if steps == ddim_steps_student:
                                continue
                            with torch.enable_grad():
                                optimizer.zero_grad()
                                x_T.requires_grad=True
                                
                                samples_ddim_student, student_intermediate, x_T_copy, a_t = sampler.sample_student(S=STUDENT_STEPS,
                                                                conditioning=c_student,
                                                                batch_size=1,
                                                                shape=[3, 64, 64],
                                                                verbose=False,
                                                                x_T=x_T,
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=uc, 
                                                                eta=ddim_eta,
                                                                keep_intermediates=False,
                                                                intermediate_step = steps*STUDENT_STEPS,
                                                                steps_per_sampling = STUDENT_STEPS,
                                                                total_steps = ddim_steps_student)
                                
                                x_T_student = student_intermediate["x_inter"][-1]
                                # loss = criterion(x_T_student, dataset[str(i)]["intermediates"][steps+1])
                                loss = max(math.log(a_t / (1-a_t)), 1) *  criterion(x_T_student, dataset[str(i)]["intermediates"][steps+1])
                                loss.backward()
                                optimizer.step()
                                scheduler.step()
                                # x_T.detach()
                                losses.append(loss.item())
                                if session != None:
                                    session.log({"loss":loss.item()})  
                                if instance % 10000 == 0 and generation > 2:
                                    save_model(sampler, optimizer, scheduler, name=f"intermediate_{instance}", steps=student_steps, run_name=run_name)
                            
                                

                        # print("Loss: ", round(sum(losses) / len(losses), 5), end= " - ")
                        averaged_losses.append(sum(losses) / len(losses))
                        tepoch.set_postfix(loss=averaged_losses[-1])
                        if session != None:
                            session.log({"generation_loss":averaged_losses[-1]})
                            session.log({"generation":generation})
                        if early_stop == True and i > 1:
                            if averaged_losses[-1] > (10*averaged_losses[-2]):
                                print(f"Early stop initiated: Prev loss: {round(averaged_losses[-2], 5)}, Current loss: {round(averaged_losses[-1], 5)}")
                                plt.plot(range(len(averaged_losses)), averaged_losses, label="MSE LOSS")
                                plt.xlabel("Generations")
                                plt.ylabel("px MSE")
                                plt.title("MSEloss student vs teacher")
                                plt.show()
                                return 
                                                                    
    plt.plot(range(len(averaged_losses)), averaged_losses, label="MSE LOSS")
    plt.xlabel("Generations")
    plt.ylabel("px MSE")
    plt.title("MSEloss student vs teacher")
    plt.show()
    

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

@torch.no_grad()
def compare_latents(images):
    """
    Compare the latents of a batch of images.
    """
    grid = torch.stack(images, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    return Image.fromarray(grid.astype(np.uint8)), grid.astype(np.uint8)
   

@torch.no_grad()
def compare_teacher_student(teacher, sampler_teacher, student, sampler_student, steps=[10], prompt=None):
    """
    Compare the a trained model and an original (teacher). Terms used are teacher and student models, though these may be the same model but at different
    stages of training.
    """
    scale = 3.0
    ddim_eta = 0.0
    images = []

    with torch.no_grad():
        with student.ema_scope():
            for sampling_steps in steps:
                sampler_teacher.make_schedule(ddim_num_steps=sampling_steps, ddim_eta=ddim_eta, verbose=False)
                sampler_student.make_schedule(ddim_num_steps=sampling_steps, ddim_eta=ddim_eta, verbose=False)
                if prompt == None:
                    class_image = torch.randint(0, 999, (1,))
                else:
                    class_image = torch.tensor([prompt])
                uc = teacher.get_learned_conditioning({teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)})
                xc = torch.tensor([class_image])
                c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                teacher_samples_ddim, _, x_T_copy, pred_x0_teacher, a_t= sampler_teacher.sample(S=sampling_steps,
                                                    conditioning=c,
                                                    batch_size=1,
                                                    x_T=None,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=ddim_eta)

                # x_samples_ddim = teacher.decode_first_stage(_["pred_x0"][-1)
                x_samples_ddim = teacher.decode_first_stage(pred_x0_teacher)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)

                uc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})
                c = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                student_samples_ddim, _, x_T_delete, pred_x0_student, a_t = sampler_student.sample(S=sampling_steps,
                                                    conditioning=c,
                                                    batch_size=1,
                                                    x_T=x_T_copy,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=ddim_eta)

                x_samples_ddim = student.decode_first_stage(pred_x0_student)
                # x_samples_ddim = teacher.decode_first_stage(_f)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)

    # from torchmetrics.image.fid import FrechetInceptionDistance
    # print(fid.compute())
    grid = torch.stack(images, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    return Image.fromarray(grid.astype(np.uint8)), grid.astype(np.uint8)

def distill(ddim_steps, generations, run_name, config, original_model_path, lr, start_trained=False):
    """
    Distill a model into a smaller model. This is done by training a student model to match the teacher model with identical initialization.
    This is not direct self-distillation as the teacher model does not distill into itself, but rather into a student model.
    """
    for index, step in enumerate(ddim_steps):
        steps = int(step / 2)
        model_generations = generations // steps
        config_path=config
        if index == 0 and start_trained != True:
            model_path=original_model_path
            teacher, sampler_teacher, student, sampler_student = create_models(config_path, model_path, student=True)
            print("Loading New Student and teacher:", ddim_steps[index])
        else:
            model_path = f"{cwd}/data/trained_models/{run_name}/{ddim_steps[index]}/student_lr8_scheduled.pt"
            print("Loading New Student and teacher:", ddim_steps[index])
            teacher, sampler_teacher, optimizer, scheduler = load_trained(model_path, config_path)
            student = copy.deepcopy(teacher)
            sampler_student = DDIMSampler(student)
        notes = f"""This is a serious attempt to distill the {step} step original teacher into a {steps} step student, trained on {model_generations * ddim_steps[index]} instances"""
        wandb_session = wandb_log(name=f"Train_student_on_{step}_pretrained", lr=lr, model=student, tags=["distillation", "auto", run_name], notes=notes)
        wandb.run.log_code(".")
        optimizer, scheduler = get_optimizer(sampler_student, iterations=generations, lr=lr)
        teacher_train_student(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, steps=step, generations=model_generations, early_stop=False, session=wandb_session, run_name=run_name)
        save_model(sampler_student, optimizer, scheduler, name="lr8_scheduled", steps=steps, run_name = run_name)
        images, grid = compare_teacher_student(teacher, sampler_teacher, student, sampler_student, steps=[1, 2, 4, 8, 16, 32, 64, 128])
        images = wandb.Image(grid, caption="left: Teacher, right: Student")
        wandb.log({"Comparison": images})
        wandb.finish()
        del teacher, sampler_teacher, student, sampler_student, optimizer, scheduler
        torch.cuda.empty_cache()



def self_distillation(student, sampler_student, original, sampler_original, optimizer, scheduler, 
            session=None, steps=20, generations=200, early_stop=True, run_name="test", decrease_steps=False,
            step_scheduler="deterministic"):
    """
    Distill a model into itself. This is done by having a (teacher) model distill knowledge into itself. Copies of the original model and sampler 
    are passed in to compare the original untrained version with the distilled model at scheduled intervals.
    """
    NUM_CLASSES = 1000
    generations = generations
    
    ddim_steps_student = steps
    TEACHER_STEPS = 2
    ddim_eta = 0.0
    scale = 3.0
    updates = int(ddim_steps_student / TEACHER_STEPS)
    optimizer=optimizer
    averaged_losses = []
    criterion = nn.MSELoss()
    instance = 0
    generation = 0
    all_losses = []
    halvings = math.floor(math.log(steps)/math.log(2)) - 1
    halving_steps = []
    for i in range(1, halvings+1):
        halving_steps.append(int(generations * (1 / (halvings + 2)) * i))

    intermediate_generation_compare = int(generations * 0.8 / (halvings + 1))
    if os.path.exists(f"{cwd}/saved_images/FID/{run_name}"):
        print("FID folder exists")
        shutil.rmtree(f"{cwd}/saved_images/FID/{run_name}")
    
    with torch.no_grad():
        with student.ema_scope():
                
            
   
                sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                
                sc = student.get_learned_conditioning(
                            {student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)}
                            )
                
                if step_scheduler!="deterministic":
                    current_fid = get_fid(student, sampler_student, num_imgs=100, name=run_name, instance = 0, steps=[ddim_steps_student])
                with tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))) as tepoch:

                    for i, class_prompt in enumerate(tepoch):
                        if i in halving_steps and decrease_steps==True and step_scheduler=="deterministic":
                            ddim_steps_student = int( ddim_steps_student/ 2)
                            save_model(sampler_student, optimizer, scheduler, name, ddim_steps_student, run_name)
                            updates = int(ddim_steps_student / TEACHER_STEPS)
                            sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                            sc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})
                            print("halved:", ddim_steps_student)

                        generation += 1
                        losses = []        
                        xc = torch.tensor([class_prompt])
                        c_student = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                        samples_ddim= None
                        predictions_temp = []
                        for steps in range(updates):          
                                    instance += 1
                                
                                    with torch.enable_grad():
                                        # sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                                        optimizer.zero_grad()

                                        samples_ddim, pred_x0_student, st, at= sampler_student.sample_student(S=1,
                                                                        conditioning=c_student,
                                                                        batch_size=1,
                                                                        shape=[3, 64, 64],
                                                                        verbose=False,
                                                                        x_T=samples_ddim,
                                                                        unconditional_guidance_scale=scale,
                                                                        unconditional_conditioning=sc, 
                                                                        eta=ddim_eta,
                                                                        keep_intermediates=False,
                                                                        intermediate_step = steps*2,
                                                                        steps_per_sampling = 1,
                                                                        total_steps = ddim_steps_student)
                                        
                                
                                    

                                    with torch.no_grad():
                                        samples_ddim, teacher_intermediate, x_T, pred_x0_teacher, a_t_teacher = sampler_student.sample(S=1,
                                                                    conditioning=c_student,
                                                                    batch_size=1,
                                                                    shape=[3, 64, 64],
                                                                    verbose=False,
                                                                    x_T=samples_ddim,
                                                                    unconditional_guidance_scale=scale,
                                                                    unconditional_conditioning=sc, 
                                                                    eta=ddim_eta,
                                                                    keep_intermediates=False,
                                                                    intermediate_step = steps*2+1,
                                                                    steps_per_sampling = 1,
                                                                    total_steps = ddim_steps_student)     
                                        
                                    with torch.enable_grad():    
                                        signal = at
                                        noise = 1 - at
                                        log_snr = torch.log(signal / noise)
                                        weight = max(log_snr, 1)
                                        

                                        loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                     
         
                                   
                                        loss.backward()
                                        
                                        
                                        torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                        
                                        optimizer.step()
                                        scheduler.step()
                                        losses.append(loss.item())
                                        
                                    if session != None:
                                        if generation > 0 and generation % intermediate_generation_compare == 0:
                                            x_T_teacher_decode = sampler_student.model.decode_first_stage(pred_x0_teacher)
                                            teacher_target = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                            x_T_student_decode = sampler_student.model.decode_first_stage(pred_x0_student.detach())
                                            student_target  = torch.clamp((x_T_student_decode +1.0)/2.0, min=0.0, max=1.0)
                                            predictions_temp.append(teacher_target)
                                            predictions_temp.append(student_target)
                                        session.log({"intermediate_loss":loss.item()})
                                    
                                    

                                  
                        if generation > 0 and generation % 20 == 0 and ddim_steps_student != 1 and step_scheduler!="deterministic":
                            fid = get_fid(student, sampler_student, num_imgs=100, name=run_name, 
                                        instance = instance, steps=[ddim_steps_student])
                            if fid[0] <= current_fid[0] * 0.9 and decrease_steps==True:
                                print(fid[0], current_fid[0])
                                if ddim_steps_student in [16, 8, 4, 2, 1]:
                                    name = "intermediate"
                                    save_model(sampler_student, optimizer, scheduler, name, steps * 2, run_name)
                                if ddim_steps_student != 2:
                                    ddim_steps_student -= 2
                                    updates -= 1
                                    

                                else:
                                    ddim_steps_student = 1
                                    updates = 1    

                                current_fid = fid
                                print("steps decresed:", ddim_steps_student)    

                            
                            
                        if session != None and generation % 20 == 0 and generation > 0:
                            fids = get_fid(student, sampler_student, num_imgs=100, name=run_name, instance = instance+1, steps=[32, 16, 8, 4, 2, 1])
                            session.log({"fid_32":fids[0]})
                            session.log({"fid_16":fids[1]})
                            session.log({"fid_8":fids[2]})
                            session.log({"fid_4":fids[3]})
                            session.log({"fid_2":fids[4]})
                            session.log({"fid_1":fids[5]})
                            session.log({"steps":ddim_steps_student})
                    
                            sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                            sc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})

                        
                        if session != None:
                            with torch.no_grad():
                                if generation > 0 and generation % intermediate_generation_compare == 0 and session !=None:
                                    img, grid = compare_latents(predictions_temp)
                                    images = wandb.Image(grid, caption="left: Teacher, right: Student")
                                    wandb.log({"Inter_Comp": images})
                                    del img, grid, predictions_temp, x_T_student_decode, x_T_teacher_decode, student_target, teacher_target
                                    torch.cuda.empty_cache()
                                
                          
                        
                        
                        all_losses.extend(losses)
                        # print(scheduler.get_last_lr())
                        averaged_losses.append(sum(losses) / len(losses))
                        if session != None:
                            session.log({"generation_loss":averaged_losses[-1]})
                        tepoch.set_postfix(epoch_loss=averaged_losses[-1])
                        torch.cuda.empty_cache()
                        if early_stop == True and i > 1:
                            if averaged_losses[-1] > (10*losses[-2]):
                                print(f"Early stop initiated: Prev loss: {round(averaged_losses[-2], 5)}, Current loss: {round(averaged_losses[-1], 5)}")
                                plt.plot(range(len(averaged_losses)), averaged_losses, label="MSE LOSS")
                                plt.xlabel("Generations")
                                plt.ylabel("px MSE")
                                plt.title("MSEloss student vs teacher")
                                plt.show()
                                 

                                                            
    plt.plot(range(len(all_losses)), all_losses, label="MSE LOSS")
    plt.xlabel("Generations")
    plt.ylabel("px MSE")
    plt.title("MSEloss student vs teacher")
    plt.show()


def get_fid(model, sampler, num_imgs, name,instance, steps =[4, 2, 1]):
    """
    Calculates the FID score for a given model and sampler. Potentially useful for monitoring training, or comparing distillation
    methods.
    """
    fid_list = []
    if not os.path.exists(f"{cwd}/saved_images/FID/{name}/{instance}"):
        os.makedirs(f"{cwd}/saved_images/FID/{name}/{instance}")
    with torch.no_grad():
        run_name = f"FID/{name}/{instance}/"
        save_images(model, sampler, num_imgs, run_name, steps, verbose=False)
        for step in steps:
            fid = fid_score.calculate_fid_given_paths(["C:/val_saved/real_fid_both.npz", 
            f"{cwd}/saved_images/FID/{name}/{instance}/{step}"], batch_size = num_imgs, device='cuda', dims=2048)
            fid_list.append(fid)
    return fid_list
