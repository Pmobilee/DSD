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
import saving_loading
import generate

# Receiving base current working directory
cwd = os.getcwd()

from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()


def self_distillation_CIN(student, sampler_student, original, sampler_original, optimizer, scheduler,
            session=None, steps=20, generations=200, early_stop=True, run_name="test", decrease_steps=False,
            step_scheduler="deterministic", type="snellius"):
    """
    Distill a model into itself. This is done by having a (teacher) model distill knowledge into itself. Copies of the original model and sampler 
    are passed in to compare the original untrained version with the distilled model at scheduled intervals.
    """
    NUM_CLASSES = 1000
    gradient_updates = generations
    ddim_steps_student = steps
    TEACHER_STEPS = 2
    ddim_eta = 0.0
    scale = 3.0
    optimizer=optimizer
    averaged_losses = []
    criterion = nn.MSELoss()
    instance = 0
    generation = 0
    all_losses = []

    if step_scheduler == "iterative":
        halvings = math.floor(math.log(32)/math.log(2)) + 1
        updates_per_halving = int(gradient_updates / halvings)
    intermediate_generation_compare = int(gradient_updates * 0.8 / 10)
    if step_scheduler == "FID":
        if os.path.exists(f"{cwd}/saved_images/FID/{run_name}"):
            print("FID folder exists")
            shutil.rmtree(f"{cwd}/saved_images/FID/{run_name}")

    with torch.no_grad():
        with student.ema_scope():              
              
                sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                sampler_original.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                sc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})
                
                if step_scheduler =="FID":
                    current_fid = util.get_fid(student, sampler_student, num_imgs=100, name=run_name, instance = 0, steps=[ddim_steps_student])
                
                updates = ddim_steps_student

                for i in range(halvings):
                    if instance != 0:
                        util.save_model(sampler_student, optimizer, scheduler, name=step_scheduler, steps=updates, run_name=run_name)
                    updates = int(updates / TEACHER_STEPS)
                    generations = updates_per_halving // updates
                    print("Distilling to:", updates)
                    
                    
                    with tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))) as tepoch:

                        for i, class_prompt in enumerate(tepoch):
                            generation += 1
                            losses = []        
                            xc = torch.tensor([class_prompt])
                            c_student = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                            samples_ddim= None
                            predictions_temp = []
                            for steps in range(updates):  

                                    # with autocast():

                                        with torch.enable_grad():
                                            
                                            instance += 1
                                        
                                            
                                            
                                            optimizer.zero_grad()
                                            
                                            
                                            samples_ddim, pred_x0_student, _, at= sampler_student.sample_student(S=1,
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
                                                                                total_steps = updates*2)
                                        

                                        with torch.no_grad():
                                            
                                            samples_ddim, _, _, pred_x0_teacher, _ = sampler_student.sample(S=1,
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
                                                                            total_steps = updates*2)     
                                        
                                        
                                    
                                        with torch.enable_grad():    
                                            if type == "home":
                                                # AUTOCAST:
                                                signal = at
                                                noise = 1 - at
                                                log_snr = torch.log(signal / noise)
                                                weight = max(log_snr, 1)
                                                loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                                scaler.scale(loss).backward()
                                                scaler.step(optimizer)
                                                scaler.update()
                                                # torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                                
                                                scheduler.step()
                                                losses.append(loss.item())

                                            else:
                                                # NO AUTOCAST:
                                                signal = at
                                                noise = 1 - at
                                                log_snr = torch.log(signal / noise)
                                                weight = max(log_snr, 1)
                                                loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                                loss.backward()
                                                optimizer.step()
                                                scheduler.step()
                                                # torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                                
                                                losses.append(loss.item())
                                            
                                        if session != None and generation % 200 == 0 and generation > 0:
                                                
                                            x_T_teacher_decode = sampler_student.model.decode_first_stage(pred_x0_teacher)
                                            teacher_target = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                            x_T_student_decode = sampler_student.model.decode_first_stage(pred_x0_student.detach())
                                            student_target  = torch.clamp((x_T_student_decode +1.0)/2.0, min=0.0, max=1.0)
                                            predictions_temp.append(teacher_target)
                                            predictions_temp.append(student_target)
                                            
                                        
                                    

                                        if session != None and instance % 10000 == 0 and generation > 0:
                                            fids = util.get_fid(student, sampler_student, num_imgs=100, name=run_name, instance = instance+1, steps=[64, 32, 16, 8, 4, 2, 1])
                                            session.log({"fid_64":fids[0]})
                                            session.log({"fid_32":fids[1]})
                                            session.log({"fid_16":fids[2]})
                                            session.log({"fid_8":fids[3]})
                                            session.log({"fid_4":fids[4]})
                                            session.log({"fid_2":fids[5]})
                                            session.log({"fid_1":fids[6]})
                                        
                                        if session != None and instance % 2000 == 0:
                                            
                                            with torch.no_grad():
                                                images, _ = util.compare_teacher_student(original, sampler_original, student, sampler_student, steps=[64, 32, 16, 8,  4, 2, 1], prompt=992)
                                                images = wandb.Image(_, caption="left: Teacher, right: Student")
                                                wandb.log({"pred_x0": images})
                                                # images, _ = util.compare_teacher_student_with_schedule(original, sampler_original, student, sampler_student, steps=[64, 32, 16, 8,  4, 2, 1], prompt=992)
                                                # images = wandb.Image(_, caption="left: Teacher, right: Student")
                                                # wandb.log({"schedule": images})
                                                sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                                                sampler_original.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)

                            if generation > 0 and generation % 20 == 0 and ddim_steps_student != 1 and step_scheduler=="FID":
                                fid = util.get_fid(student, sampler_student, num_imgs=100, name=run_name, 
                                            instance = instance, steps=[ddim_steps_student])
                                if fid[0] <= current_fid[0] * 0.9 and decrease_steps==True:
                                    print(fid[0], current_fid[0])
                                    if ddim_steps_student in [16, 8, 4, 2, 1]:
                                        name = "intermediate"
                                        saving_loading.save_model(sampler_student, optimizer, scheduler, name, steps * 2, run_name)
                                    if ddim_steps_student != 2:
                                        ddim_steps_student -= 2
                                        updates -= 1
                                    else:
                                        ddim_steps_student = 1
                                        updates = 1    
                                    current_fid = fid
                                    print("steps decresed:", ddim_steps_student)    

                            if session != None:
                                with torch.no_grad():
                                    if session != None and generation % 200 == 0 and generation > 0:
                                        img, grid = util.compare_latents(predictions_temp)
                                        images = wandb.Image(grid, caption="left: Teacher, right: Student")
                                        wandb.log({"Inter_Comp": images})
                                        del img, grid, predictions_temp, x_T_student_decode, x_T_teacher_decode, student_target, teacher_target
                                        torch.cuda.empty_cache()
                            
                            all_losses.extend(losses)
                            averaged_losses.append(sum(losses) / len(losses))
                            if session != None:
                                session.log({"generation_loss":averaged_losses[-1]})
                            tepoch.set_postfix(epoch_loss=averaged_losses[-1])
                           
                                                                           
def self_distillation_CELEB(student, sampler_student, original, sampler_original, optimizer, scheduler,
        session=None, steps=20, generations=200, early_stop=True, run_name="test", decrease_steps=False,
        step_scheduler="deterministic", type="snellius"):
    """
    Distill a model into itself. This is done by having a (teacher) model distill knowledge into itself. Copies of the original model and sampler 
    are passed in to compare the original untrained version with the distilled model at scheduled intervals.
    """
    gradient_updates = generations
    ddim_steps_student = steps
    TEACHER_STEPS = 2
    ddim_eta = 0.0
    scale = 3.0
    optimizer=optimizer
    averaged_losses = []
    criterion = nn.MSELoss()
    instance = 0
    generation = 0
    all_losses = []

    if step_scheduler == "iterative":
        halvings = math.floor(math.log(32)/math.log(2)) + 1
        updates_per_halving = int(gradient_updates / halvings)
    intermediate_generation_compare = int(gradient_updates * 0.8 / 10)
    if step_scheduler == "FID":
        if os.path.exists(f"{cwd}/saved_images/FID/{run_name}"):
            print("FID folder exists")
            shutil.rmtree(f"{cwd}/saved_images/FID/{run_name}")

    with torch.no_grad():
        with student.ema_scope():              
                
                sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                sampler_original.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                
                
                if step_scheduler =="FID":
                    current_fid = util.get_fid(student, sampler_student, num_imgs=100, name=run_name, instance = 0, steps=[ddim_steps_student])
                
                updates = ddim_steps_student

                for i in range(halvings):
                    if instance != 0:
                        util.save_model(sampler_student, optimizer, scheduler, name=step_scheduler, steps=updates, run_name=run_name)
                    updates = int(updates / TEACHER_STEPS)
                    generations = updates_per_halving // updates
                    print("Distilling to:", updates)
                    
                    
                    with tqdm.tqdm(range(generations)) as tepoch:

                        for i in tepoch:
                            generation += 1
                            losses = []        
                
                            samples_ddim= None
                            predictions_temp = []
                            for steps in range(updates):  

                                    # with autocast():

                                        with torch.enable_grad():
                                            
                                            instance += 1
                                        
                                            
                                            
                                            optimizer.zero_grad()
                                            
                                            
                                            samples_ddim, pred_x0_student, _, at= sampler_student.sample_student(S=1,
                                                                                conditioning=None,
                                                                                batch_size=1,
                                                                                shape=[3, 64, 64],
                                                                                verbose=False,
                                                                                x_T=samples_ddim,
                                                                        
                                                                                unconditional_guidance_scale=scale,
                                                                                unconditional_conditioning=None, 
                                                                                eta=ddim_eta,
                                                                                keep_intermediates=False,
                                                                                intermediate_step = steps*2,
                                                                                steps_per_sampling = 1,
                                                                                total_steps = updates*2)
                                        

                                        with torch.no_grad():
                                            
                                            samples_ddim, _, _, pred_x0_teacher, _ = sampler_student.sample(S=1,
                                                                            conditioning=None,
                                                                            batch_size=1,
                                                                            shape=[3, 64, 64],
                                                                            verbose=False,
                                                                            x_T=samples_ddim,
                                                                            unconditional_guidance_scale=scale,
                                                                            unconditional_conditioning=None, 
                                                                            eta=ddim_eta,
                                                                            keep_intermediates=False,
                                                                            intermediate_step = steps*2+1,
                                                                            steps_per_sampling = 1,
                                                                            total_steps = updates*2)     
                                        
                                        
                                    
                                        with torch.enable_grad():    
                                            if type == "home":
                                                # AUTOCAST:
                                                signal = at
                                                noise = 1 - at
                                                log_snr = torch.log(signal / noise)
                                                weight = max(log_snr, 1)
                                                loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                                scaler.scale(loss).backward()
                                                scaler.step(optimizer)
                                                scaler.update()
                                                # torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                                
                                                scheduler.step()
                                                losses.append(loss.item())

                                            else:
                                                # NO AUTOCAST:
                                                signal = at
                                                noise = 1 - at
                                                log_snr = torch.log(signal / noise)
                                                weight = max(log_snr, 1)
                                                loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                                loss.backward()
                                                optimizer.step()
                                                scheduler.step()
                                                # torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                                
                                                losses.append(loss.item())
                                            
                                        if session != None and generation % 100 == 0 and generation > 0:
                                                
                                            x_T_teacher_decode = sampler_student.model.decode_first_stage(pred_x0_teacher)
                                            teacher_target = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                            x_T_student_decode = sampler_student.model.decode_first_stage(pred_x0_student.detach())
                                            student_target  = torch.clamp((x_T_student_decode +1.0)/2.0, min=0.0, max=1.0)
                                            predictions_temp.append(teacher_target)
                                            predictions_temp.append(student_target)
                                            
                                    

                                        # if session != None and instance % 10000 == 0 and generation > 0:
                                        #     fids = util.get_fid_celeb(student, sampler_student, num_imgs=100, name=run_name, instance = instance+1, steps=[64, 32, 16, 8, 4, 2, 1])
                                        #     session.log({"fid_64":fids[0]})
                                        #     session.log({"fid_32":fids[1]})
                                        #     session.log({"fid_16":fids[2]})
                                        #     session.log({"fid_8":fids[3]})
                                        #     session.log({"fid_4":fids[4]})
                                        #     session.log({"fid_2":fids[5]})
                                        #     session.log({"fid_1":fids[6]})
                                        
                                        if session != None and instance % 2000 == 0:
                                    
                                            with torch.no_grad():
                                                images, _ = util.compare_teacher_student_celeb(original, sampler_original, student, sampler_student, steps=[64, 32, 16, 8,  4, 2, 1])
                                                images = wandb.Image(_, caption="left: Teacher, right: Student")
                                                wandb.log({"pred_x0": images})
                                                # images, _ = util.compare_teacher_student_with_schedule(original, sampler_original, student, sampler_student, steps=[64, 32, 16, 8,  4, 2, 1], prompt=992)
                                                # images = wandb.Image(_, caption="left: Teacher, right: Student")
                                                # wandb.log({"schedule": images})
                                                sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                                                sampler_original.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)

                            # if generation > 0 and generation % 20 == 0 and ddim_steps_student != 1 and step_scheduler=="FID":
                            #     fid = util.get_fid(student, sampler_student, num_imgs=100, name=run_name, 
                            #                 instance = instance, steps=[ddim_steps_student])
                            #     if fid[0] <= current_fid[0] * 0.9 and decrease_steps==True:
                            #         print(fid[0], current_fid[0])
                            #         if ddim_steps_student in [16, 8, 4, 2, 1]:
                            #             name = "intermediate"
                            #             saving_loading.save_model(sampler_student, optimizer, scheduler, name, steps * 2, run_name)
                            #         if ddim_steps_student != 2:
                            #             ddim_steps_student -= 2
                            #             updates -= 1
                            #         else:
                            #             ddim_steps_student = 1
                            #             updates = 1    
                            #         current_fid = fid
                            #         print("steps decresed:", ddim_steps_student)    

                            if session != None:
                                with torch.no_grad():
                                    if session != None and generation % 100 == 0 and generation > 0:
                                        img, grid = util.compare_latents(predictions_temp)
                                        images = wandb.Image(grid, caption="left: Teacher, right: Student")
                                        wandb.log({"Inter_Comp": images})
                                        del img, grid, predictions_temp, x_T_student_decode, x_T_teacher_decode, student_target, teacher_target
                                        torch.cuda.empty_cache()
                            
                            all_losses.extend(losses)
                            averaged_losses.append(sum(losses) / len(losses))
                            if session != None:
                                session.log({"generation_loss":averaged_losses[-1]})
                            tepoch.set_postfix(epoch_loss=averaged_losses[-1])