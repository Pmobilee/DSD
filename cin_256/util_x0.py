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


"""
This module is property of the Vrije Universiteit Amsterdam, department of Beta Science. It contains in part code
snippets obtained from Rombach et al., https://github.com/CompVis/latent-diffusion. No rights may be attributed.

The module presents both helper modules for loading, saving, generating from, and training of diffusion models, as
well as components for the process of knowledge distillation of teacher DDIMs into students, requiring fewer denoising
steps after every iteration, retaining original sampling quality at reduced computational expense.
"""





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
    torch.save({"model":model.state_dict(), "optimizer":optimizer, "scheduler":scheduler}, path + f"student_{name}.pt")


def load_trained(model_path, config):
    config = OmegaConf.load(config)  
    ckpt = torch.load(model_path)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    model.cuda()
    
    return model,  ckpt["optimizer"], ckpt["scheduler"]

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
def generate(model, x0=True, num_imgs=1, steps=20, x_T=None, class_prompt=None, keep_intermediates=False):
    """
    Params: model, sampler, num_imgs=1, steps=20, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False. 
    Task: returns final generated samples from the provided model and accompanying sampler. Unless the class prompt is specified,
    all generated images are of one of the random classes.
    """
    timesteps =  steps
    class_embed = torch.randint(0, 1000, (1,))
    xc = torch.tensor([class_embed])
    model.num_timesteps = timesteps
    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
    img, x, x_T_copy = model.progressive_denoising(cond=c, shape=[3, 64, 64], verbose=True, callback=None, quantize_denoised=False,
                                img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                                score_corrector=None, corrector_kwargs=None, batch_size=1, x_T=None, start_T=None,
                                log_every_t=None)
          
                                    
    # display as grid
    if x0 == True:
        target = x
    else:
        target = img

    x_samples_ddim = model.decode_first_stage(target)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                min=0.0, max=1.0)
                                
    grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image = Image.fromarray(grid.astype(np.uint8))

    return image, x_T_copy, class_prompt


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
    

def teacher_train_student(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test"):
    NUM_CLASSES = 1000
    generations = generations
    intermediate_generation = generations // 5
    ddim_steps_teacher = steps
    ddim_steps_student = int(ddim_steps_teacher / 2)
    TEACHER_STEPS = 2
    STUDENT_STEPS = 1
    ddim_eta = 0.01
    scale = 3.0
    updates = int(ddim_steps_teacher / TEACHER_STEPS)
    optimizer=optimizer
    averaged_losses = []
    teacher_samples = list()
    criterion = nn.MSELoss()
    instance = 0
    generation = 0
    

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
                            {teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)}
                            )
                with tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))) as tepoch:
                    for i, class_prompt in enumerate(tepoch):
                        
            

                        generation += 1
                        losses = []        
                        xc = torch.tensor([class_prompt])
                        c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                        c_student = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                        x_T = None
                        
                        for steps in range(updates):          
                                    instance += 1
                                    
                                    # sampler_teacher.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
                                    samples_ddim_teacher, teacher_intermediate, x_T, pred_x0_teacher, a_t_teacher, e_t_teacher = sampler_teacher.sample(S=TEACHER_STEPS,
                                                                    conditioning=c,
                                                                    batch_size=1,
                                                                    shape=[3, 64, 64],
                                                                    verbose=False,
                                                                    x_T=x_T,
                                                                    # quantize_x0 = True,
                                                                    unconditional_guidance_scale=scale,
                                                                    # unconditional_conditioning=uc, 
                                                                    eta=ddim_eta,
                                                                    keep_intermediates=False,
                                                                    intermediate_step = steps*TEACHER_STEPS,
                                                                    steps_per_sampling = TEACHER_STEPS,
                                                                    total_steps = ddim_steps_teacher)      
                                    
                                

                                    # x_T_teacher = teacher_intermediate["x_inter"][-1]
                                    
                                    # x_T_teacher_decode = sampler_teacher.model.decode_first_stage(samples_ddim_teacher)
                                    # x_T_teacher = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                    
                                    # teacher_target = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                    

                                    with torch.enable_grad():
                                        # sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                                        optimizer.zero_grad()
                                        samples_ddim_student, student_intermediate, x_T_copy, a_t, pred_x0_student, sigma_t, e_t_student= sampler_student.sample_student(S=STUDENT_STEPS,
                                                                        conditioning=c_student,
                                                                        batch_size=1,
                                                                        shape=[3, 64, 64],
                                                                        verbose=False,
                                                                        x_T=x_T,
                                                                        # quantize_x0 = True,
                                                                        unconditional_guidance_scale=scale,
                                                                        # unconditional_conditioning=c, 
                                                                        eta=ddim_eta,
                                                                        keep_intermediates=False,
                                                                        intermediate_step = steps*STUDENT_STEPS,
                                                                        steps_per_sampling = STUDENT_STEPS,
                                                                        total_steps = ddim_steps_student)
                                        
                                        # x_T_student = student_intermediate["x_inter"][-1]
                                        
                                        # print("len samples ddim:", samples_ddim_student.shape)
                                        # x_T_student_decode = sampler_student.model.differentiable_decode_first_stage(samples_ddim_student)
                                        # student_target  = torch.clamp((x_T_student_decode +1.0)/2.0, min=0.0, max=1.0)
                                        # loss = max(math.log(a_t**2 / (1-a_t) **2), 1) *  criterion(samples_ddim_student, samples_ddim_teacher)
                                        # loss = math.log(a_t / (sigma_t)) * criterion(pred_x0_student, pred_x0_teacher)
                                        # loss =  abs(math.log(a_t / (1-a_t))) * criterion(e_t_student, e_t_teacher)
                                        loss = math.log(a_t**2 / (1-a_t) **2) * criterion(e_t_student, e_t_teacher)
                                        
                                        # loss =  criterion(samples_ddim_student, samples_ddim_teacher)
                                        # loss = max(math.log(a_t / (1-a_t)), 1) *  criterion(x_T_student, x_T)
                                        # loss = max(math.log(a_t / (1-a_t)), 1) *  criterion(x_T_student_decode, x_T_teacher_decode) 
                                        # print("stp:", steps, "Lss:", loss.item(), end="--")    
                                                                
                                        loss.backward()
                                        
                                        # print(math.log(a_t**2 / (sigma_t ** 2)))
                                        
                                        # torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                        
                                        optimizer.step()
                                        scheduler.step()
                                        losses.append(loss.item())
                                        if session != None:
                                            session.log({"intermediate_loss":loss.item()})

                                

                                # del samples_ddim, samples_ddim_student, teacher_intermediate, student_intermediate, x_T_copy, a_t, pred_x0_student
                                # torch.cuda.empty_cache()

                        # with torch.no_grad():
                        #     if generation > 2 and generation % intermediate_generation == 0:
                        #         save_model(sampler_student, optimizer, scheduler, name=f"intermediate_{instance}", steps=ddim_steps_student, run_name=run_name)
                        #         images, grid = compare_teacher_student(teacher, sampler_teacher, student, sampler_student, steps=[1, 2, 4, 6, 8, 10, 16, 20, 32, 64, 128])
                        #         images = wandb.Image(grid, caption=f"{instance} steps, left: Teacher, right: Student")
                        #         wandb.log({"Intermediate": images})
                        
                        
                        
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
    

    

@torch.no_grad()
def create_models(config_path, model_path, student=False, timesteps=1000):
    model = get_model(config_path=config_path, model_path=model_path)
    model.reregister_schedule(timesteps=timesteps,linear_start=0.0015 * (timesteps/1000) , linear_end=0.0195 * (timesteps/1000), beta_schedule="linear", force=True)
    if student == True:
        student = copy.deepcopy(model)
        return model, student
    else:
        return model

@torch.no_grad()
def compare_teacher_student(teacher, student, steps=[10], class_prompt=None):
    
    images = []

    with torch.no_grad():
        with teacher.ema_scope():
            for sampling_steps in steps:
                if class_prompt == None:
                    class_image = torch.randint(0, 1000, (1,))
                else:
                    class_image = torch.tensor([class_prompt])
                student.num_timesteps=sampling_steps
                teacher.num_timesteps=sampling_steps
                # student.reregister_schedule(timesteps=sampling_steps,linear_start=0.0015 * (sampling_steps/1000) , linear_end=0.0195 * (sampling_steps/1000), beta_schedule="linear", force=True)
                # teacher.reregister_schedule(timesteps=sampling_steps,linear_start=0.0015 * (sampling_steps/1000) , linear_end=0.0195 * (sampling_steps/1000), beta_schedule="linear", force=True)

                xc = torch.tensor([class_image])
                c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                c_student = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                img = None
                img, x0, x_T_copy = teacher.progressive_denoising(cond=c, shape=[3, 64, 64], verbose=True, callback=None, quantize_denoised=False,
                                    img_callback=None, mask=None, x0=None, temperature=1.0, noise_dropout=0.,
                                    score_corrector=None, corrector_kwargs=None, batch_size=1, x_T=img,
                                    log_every_t=None) 


                x_samples_ddim = teacher.decode_first_stage(x0)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)

             
                img_student, x0_student, x_T_copy = student.progressive_denoising(cond=c_student, shape=[3, 64, 64], verbose=True, callback=None, quantize_denoised=False,
                                        img_callback=None, mask=None, x0=None, temperature=1.0, noise_dropout=0.,
                                        score_corrector=None, corrector_kwargs=None, batch_size=1, x_T=x_T_copy,
                                        log_every_t=None)   

                x_samples_ddim = student.decode_first_stage(x0_student)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)

    grid = torch.stack(images, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    return Image.fromarray(grid.astype(np.uint8)), grid.astype(np.uint8)

def distill(ddim_steps, generations, run_name, config, original_model_path, lr):
    for index, step in enumerate(ddim_steps):
        steps = int(step / 2)
        model_generations = generations // steps
        if index == 0:
            config_path=config
            model_path=original_model_path
            teacher, sampler_teacher, student, sampler_student = create_models(config_path, model_path, student=True)
        else:
            model_path = f"{cwd}/data/trained_models/{run_name}/{ddim_steps[index]}/student_lr8_scheduled.pt"
            teacher, sampler_teacher, optimizer, scheduler = load_trained(model_path, config_path)
            student = copy.deepcopy(teacher)
            sampler_student = DDIMSampler(student)
        notes = f"""This is a serious attempt to distill the {step} step original teacher into a {steps} step student, trained on {model_generations * ddim_steps[index]} instances"""
        wandb_session = wandb_log(name=f"Train_student_on_{step}_pretrained", lr=lr, model=student, tags=["distillation", "auto", run_name], notes=notes)
        optimizer, scheduler = get_optimizer(sampler_student, iterations=model_generations * steps, lr=lr)
        teacher_train_student(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, steps=step, generations=model_generations, early_stop=False, session=wandb_session, run_name=run_name)
        save_model(sampler_student, optimizer, scheduler, name="lr8_scheduled", steps=steps, run_name = run_name)
        images, grid = compare_teacher_student(teacher, sampler_teacher, student, sampler_student, steps=[1, 2, 4, 8, 16, 32, 64, 128])
        images = wandb.Image(grid, caption="left: Teacher, right: Student")
        wandb.log({"Comparison": images})
        wandb.finish()
        del teacher, sampler_teacher, student, sampler_student, optimizer, scheduler
        torch.cuda.empty_cache()