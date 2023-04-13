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
import matplotlib.pyplot as plt
import copy
import wandb
import math
import traceback
from pytorch_fid import fid_score
import shutil
import util
import saving_loading
import generate

# Receiving base current working directory
cwd = os.getcwd()

"""
This module is property of the Vrije Universiteit Amsterdam, department of Beta Science. It contains in part code
snippets obtained from Rombach et al., https://github.com/CompVis/latent-diffusion. No rights may be attributed.

The module presents both helper modules for loading, saving, generating from, and training of diffusion models, as
well as components for the process of knowledge distillation of teacher DDIMs into students, requiring fewer denoising
steps after every iteration, retaining original sampling quality at reduced computational expense.
"""


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
                                                                temperature=0.0,
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
                                    saving_loading.save_model(sampler, optimizer, scheduler, name=f"intermediate_{instance}", steps=student_steps, run_name=run_name)
                            
                                

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
    

@torch.enable_grad()
def train_student_from_dataset_celeb(model, sampler, dataset, student_steps, optimizer, scheduler, early_stop=False, session=None, run_name="test"):
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
                        
                        losses = []
                        sampler.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                        sampler.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                        generation += 1
                        for steps, x_T in enumerate(dataset[str(i)]["intermediates"]):
                            instance += 0
                            if steps == ddim_steps_student:
                                continue
                            with torch.enable_grad():
                                optimizer.zero_grad()
                                x_T.requires_grad=True
                                
                                samples_ddim_student, student_intermediate, x_T_copy, a_t = sampler.sample_student(S=STUDENT_STEPS,   
                                                                batch_size=1,
                                                                shape=[3, 64, 64],
                                                                verbose=False,
                                                                x_T=x_T,
                                                                temperature=0.0,
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
                                    saving_loading.save_model(sampler, optimizer, scheduler, name=f"intermediate_{instance}", steps=student_steps, run_name=run_name)
                            
                                

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
        with teacher.ema_scope():
                
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
                                                                    temperature=0.0,
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
                                        with student.ema_scope():
                                            # sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                                            optimizer.zero_grad()
                                            samples, pred_x0_student, st, at = sampler_student.sample_student(S=STUDENT_STEPS,
                                                                            conditioning=c_student,
                                                                            batch_size=1,
                                                                            shape=[3, 64, 64],
                                                                            verbose=False,
                                                                            x_T=x_T,
                                                                            temperature=0.0,
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
                                    img, grid = util.compare_latents(predictions_temp)
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


def teacher_train_student_celeb(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test"):
    """
    Params: teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test". 
    Task: trains the student model using the identical teacher model as a guide. Not used in direct self-distillation where a teacher distills into itself.
    """
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
    all_losses = []

    with torch.no_grad():
        with teacher.ema_scope():
                
                sampler_teacher.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
                sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                # for class_prompt in tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))):

                with tqdm.tqdm(generations) as tepoch:
                    for i in range(tepoch):
                        
                        generation += 1
                        losses = []        
                        samples_ddim_teacher = None

                        for steps in range(updates):          
                                    instance += 1
                                    samples_ddim_teacher, teacher_intermediate, x_T, pred_x0_teacher, a_t_teacher = sampler_teacher.sample(S=TEACHER_STEPS,
                                                                    conditioning=None,
                                                                    batch_size=1,
                                                                    shape=[3, 64, 64],
                                                                    verbose=False,
                                                                    x_T=samples_ddim_teacher,
                                                                    unconditional_guidance_scale=scale,
                                                                    unconditional_conditioning=None, 
                                                                    eta=ddim_eta,
                                                                    temperature=0.0,
                                                                    keep_intermediates=False,
                                                                    quantize_x0=False,
                                                                    intermediate_step = steps*TEACHER_STEPS,
                                                                    steps_per_sampling = TEACHER_STEPS,
                                                                    total_steps = ddim_steps_teacher)     
                                    
                                    

                                  
                                    

                                    with torch.enable_grad():
                                        with student.ema_scope():
                                            optimizer.zero_grad()
                                            samples, pred_x0_student, st, at= sampler_student.sample_student(S=STUDENT_STEPS,
                                                                            conditioning=None,
                                                                            batch_size=1,
                                                                            shape=[3, 64, 64],
                                                                            verbose=False,
                                                                            x_T=samples_ddim_teacher,
                                                                            unconditional_guidance_scale=scale,
                                                                            unconditional_conditioning=None, 
                                                                            quantize_x0=False,
                                                                            temperature=0.0,
                                                                            eta=ddim_eta,
                                                                            keep_intermediates=False,
                                                                            intermediate_step = steps*STUDENT_STEPS,
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
                                    img, grid = util.compare_latents(predictions_temp)
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
            teacher, sampler_teacher, student, sampler_student = saving_loading.create_models(config_path, model_path, student=True)
            print("Loading New Student and teacher:", ddim_steps[index])
        else:
            model_path = f"{cwd}/data/trained_models/{run_name}/{ddim_steps[index]}/student_lr8_scheduled.pt"
            print("Loading New Student and teacher:", ddim_steps[index])
            teacher, sampler_teacher, optimizer, scheduler = saving_loading.load_trained(model_path, config_path)
            student = copy.deepcopy(teacher)
            sampler_student = DDIMSampler(student)
        notes = f"""This is a serious attempt to distill the {step} step original teacher into a {steps} step student, trained on {model_generations * ddim_steps[index]} instances"""
        wandb_session = util.wandb_log(name=f"Train_student_on_{step}_pretrained", lr=lr, model=student, tags=["distillation", "auto", run_name], notes=notes)
        wandb.run.log_code(".")
        optimizer, scheduler = saving_loading.get_optimizer(sampler_student, iterations=generations, lr=lr)
        teacher_train_student(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, steps=step, generations=model_generations, early_stop=False, session=wandb_session, run_name=run_name)
        saving_loading.save_model(sampler_student, optimizer, scheduler, name="lr8_scheduled", steps=steps, run_name = run_name)
        images, grid = util.compare_teacher_student(teacher, sampler_teacher, student, sampler_student, steps=[1, 2, 4, 8, 16, 32, 64, 128])
        images = wandb.Image(grid, caption="left: Teacher, right: Student")
        wandb.log({"Comparison": images})
        wandb.finish()
        del teacher, sampler_teacher, student, sampler_student, optimizer, scheduler
        torch.cuda.empty_cache()

def distill_celeb(ddim_steps, generations, run_name, config, original_model_path, lr, tags):
    for index, step in enumerate(ddim_steps):
        steps = int(step / 2)
        model_generations = generations // steps
        if index == 0:
            config_path=config
            model_path=original_model_path
            teacher, sampler_teacher, student, sampler_student = saving_loading.create_models(config_path, model_path, student=True)
        else:
            model_path = f"{cwd}/data/trained_models/{run_name}/{ddim_steps[index]}/student_lr8_scheduled.pt"
            teacher, sampler_teacher, optimizer, scheduler = saving_loading.load_trained(model_path, config_path)
            student = copy.deepcopy(teacher)
            sampler_student = DDIMSampler(student)
        notes = f"""This is a serious attempt to distill the {step} step original teacher into a {steps} step student, trained on {model_generations * ddim_steps[index]} instances"""
        wandb_session = util.wandb_log(name=run_name, lr=lr, model=student, tags=tags, notes=notes)
        wandb.run.log_code(".")
        optimizer, scheduler = saving_loading.get_optimizer(sampler_student, iterations=model_generations * steps, lr=lr)
        teacher_train_student_celeb(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, steps=step, generations=model_generations, early_stop=False, session=wandb_session, run_name=run_name)
        saving_loading.save_model(sampler_student, optimizer, scheduler, name="lr8_scheduled", steps=steps, run_name = run_name)
        images, grid = util.compare_teacher_student_celeb(teacher, sampler_teacher, student, sampler_student, steps=[1, 2, 4, 8, 16, 32, 64, 128])
        images = wandb.Image(grid, caption="left: Teacher, right: Student")
        wandb.log({"Comparison": images})
        wandb.finish()
        del teacher, sampler_teacher, student, sampler_student, optimizer, scheduler
        torch.cuda.empty_cache()