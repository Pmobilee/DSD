import os
import sys
import torch
import importlib
import tqdm
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
import copy

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
def generate(model, sampler, num_imgs=1, steps=20, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False):
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
                                                keep_intermediates=keep_intermediates,
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

    return image, x_T_copy, class_prompt, _["x_inter"]

@torch.no_grad()
def return_intermediates_for_student(model, sampler, steps=20, eta=0.0, scale=3.0):
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
                                intermediates.append(x_T_copy)
                            x_T = _["x_inter"][-1]
                            intermediates.append(x_T)

    return torch.stack(intermediates), starting_noise, class_prompt


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



@torch.no_grad()
def make_dataset(model, sampler, num_images, sampling_steps, path, name):
    dataset = dict()
    if not os.path.exists(path):
        os.mkdir(path)
    for i in tqdm.tqdm(range(num_images)):
        new_dict = dict()
        intermediates, starting_noise, class_prompt = return_intermediates_for_student(model, sampler, steps=sampling_steps)
        new_dict["class"] = class_prompt
        new_dict["intermediates"] = intermediates
        dataset[str(i)] = new_dict
        if (i + 1) % 100 == 0:
            new_path = path + f"{i+1}_" + name
            torch.save(dataset, new_path)
            del dataset
            dataset = dict()
    if num_images < 100:
        new_path = path + f"{num_images}_" + name
        torch.save(dataset, new_path)
        del dataset
        dataset = dict()
    


def teacher_train_student(teacher, sampler_teacher, student, sampler_student, steps=20, lr = 0.00000001, generations=200):
    
    MSEloss = nn.MSELoss()

    NUM_CLASSES = 1000
    generations = generations
    ddim_steps_teacher = steps
    ddim_steps_student = int(ddim_steps_teacher / 2)
    TEACHER_STEPS = 2
    STUDENT_STEPS = 1
    ddim_eta = 0.0
    scale = 3.0
    updates = int(ddim_steps_teacher / TEACHER_STEPS)

    averaged_losses = []
    teacher_samples = list()

    optimizer = torch.optim.Adam(sampler_student.model.parameters(), lr=0.00000001)

    with torch.no_grad():
        
        with teacher.ema_scope():
                uc = teacher.get_learned_conditioning(
                        {teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)}
                        )
                
                # for class_prompt in tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))):
                with tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))) as tepoch:
                        for i, class_prompt in enumerate(tepoch):
                            losses = []
                            sampler_teacher.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
                            xc = torch.tensor([class_prompt])
                            c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                            x_T = None
                            sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                            c_student = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                            for steps in range(updates):
                                    
                                    samples_ddim, teacher_intermediate, x_T_copy = sampler_teacher.sample(S=TEACHER_STEPS,
                                                                    conditioning=c,
                                                                    batch_size=1,
                                                                    shape=[3, 64, 64],
                                                                    verbose=False,
                                                                    x_T=x_T,
                                                                    unconditional_guidance_scale=scale,
                                                                    unconditional_conditioning=uc, 
                                                                    eta=ddim_eta,
                                                                    keep_intermediates=True,
                                                                    intermediate_step = steps*TEACHER_STEPS,
                                                                    steps_per_sampling = TEACHER_STEPS,
                                                                    total_steps = ddim_steps_teacher)
                                    x_T = teacher_intermediate["x_inter"][-1]

                                    with torch.enable_grad():
                                            optimizer.zero_grad()
                                            samples_ddim_student, student_intermediate, x_T_copy = sampler_student.sample_student(S=STUDENT_STEPS,
                                                                            conditioning=c_student,
                                                                            batch_size=1,
                                                                            shape=[3, 64, 64],
                                                                            verbose=False,
                                                                            x_T=x_T_copy,
                                                                            unconditional_guidance_scale=scale,
                                                                            unconditional_conditioning=uc, 
                                                                            eta=ddim_eta,
                                                                            keep_intermediates=True,
                                                                            intermediate_step = steps*STUDENT_STEPS,
                                                                            steps_per_sampling = STUDENT_STEPS,
                                                                            total_steps = ddim_steps_student)
                                            
                                            x_T_student = student_intermediate["x_inter"][-1]
                                            loss = MSEloss(x_T_student, x_T)
                                            loss.backward()
                                            optimizer.step()
                                            losses.append(loss.item())

                            # print("Loss: ", round(sum(losses) / len(losses), 5), end= " - ")
                            averaged_losses.append(sum(losses) / len(losses))
                            tepoch.set_postfix(loss=averaged_losses[-1])
                                                            
    plt.plot(range(len(averaged_losses)), averaged_losses, label="MSE LOSS")
    plt.xlabel("Generations")
    plt.ylabel("px MSE")
    plt.title("MSEloss student vs teacher")
    plt.show()


@torch.enable_grad()
def train_student_from_dataset(model, sampler, dataset, student_steps, lr=0.00000001, early_stop=False):
    device = torch.device("cuda")
    model.requires_grad=True
    sampler.requires_grad=True
    for param in sampler.model.parameters():
        param.requires_grad = True

    for param in model.model.parameters():
        param.requires_grad = True
    MSEloss = nn.MSELoss()
    ddim_steps_student = student_steps
    STUDENT_STEPS = 1
    ddim_eta = 0.0
    scale = 3.0

    averaged_losses = []
    teacher_samples = list()

    optimizer = torch.optim.Adam(sampler.model.parameters(), lr=lr)

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
                        
                        for steps, x_T in enumerate(dataset[str(i)]["intermediates"]):
                            if steps == ddim_steps_student:
                                continue
                            with torch.enable_grad():
                                optimizer.zero_grad()
                                x_T.requires_grad=True
                                
                                samples_ddim_student, student_intermediate, x_T_copy = sampler.sample_student(S=STUDENT_STEPS,
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
                                loss = MSEloss(x_T_student, dataset[str(i)]["intermediates"][steps+1])
                                loss.backward()
                                optimizer.step()
                                # x_T.detach()
                                losses.append(loss.item())
                                

                        # print("Loss: ", round(sum(losses) / len(losses), 5), end= " - ")
                        averaged_losses.append(sum(losses) / len(losses))
                        tepoch.set_postfix(loss=averaged_losses[-1])
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
    model = get_model(config_path=config_path, model_path=model_path)
    sampler = DDIMSampler(model)
    if student == True:
        student = copy.deepcopy(model)
        sampler_student = DDIMSampler(student)
        return model, sampler, student, sampler_student
    else:
        return model, sampler


@torch.no_grad()
def compare_teacher_student(teacher, sampler_teacher, student, sampler_student, steps=[10]):
    scale = 3.0
    eta = 0.0
    ddim_eta = 0.0
    images = []

    with torch.no_grad():
        with teacher.ema_scope():
            for sampling_steps in steps:
                class_image = torch.randint(0, 999, (1,))
                uc = teacher.get_learned_conditioning({teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)})
                xc = torch.tensor([class_image])
                c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                teacher_samples_ddim, _, x_T_copy = sampler_teacher.sample(S=sampling_steps,
                                                    conditioning=c,
                                                    batch_size=1,
                                                    x_T=None,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=ddim_eta)

                x_samples_ddim = teacher.decode_first_stage(teacher_samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)

                uc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})
                c = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                student_samples_ddim, _, x_T_delete = sampler_student.sample(S=sampling_steps,
                                                    conditioning=c,
                                                    batch_size=1,
                                                    x_T=x_T_copy,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=ddim_eta)

                x_samples_ddim = student.decode_first_stage(student_samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)

    grid = torch.stack(images, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    return Image.fromarray(grid.astype(np.uint8))