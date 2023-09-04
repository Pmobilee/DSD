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
    sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
    sampler_original.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
    if step_scheduler == "iterative":
        halvings = math.floor(math.log(64)/math.log(2))
        updates_per_halving = int(gradient_updates / halvings)
        step_sizes = []
        for i in range(halvings):
            step_sizes.append(int((steps) / (2**i)))
        update_list = []
        for i in step_sizes:
            update_list.append(int(updates_per_halving / int(i/ 2)))
    elif step_scheduler == "naive":
        step_sizes=[ddim_steps_student]
        update_list=[gradient_updates // int(ddim_steps_student / 2)]
    elif step_scheduler == "gradual_linear":
        step_sizes = np.arange(steps, 0, -2)
        update_list = (1/len(np.append(step_sizes[1:], 1)) * gradient_updates / np.append(step_sizes[1:], 1)).astype(int)
    elif step_scheduler == "gradual_exp":
        step_sizes = np.arange(64, 0, -2)
        update_list = np.exp(1 / np.append(step_sizes[1:],1)) / np.sum(np.exp(1 / np.append(step_sizes[1:],1)))
        update_list = (update_list * gradient_updates /  np.append(step_sizes[1:],1)).astype(int)

    
    if step_scheduler == "FID":
        if os.path.exists(f"{cwd}/saved_images/FID/{run_name}"):
            print("FID folder exists")
            shutil.rmtree(f"{cwd}/saved_images/FID/{run_name}")

    with torch.no_grad():
        with student.ema_scope():              
              
                if step_scheduler =="FID":
                    current_fid = util.get_fid(student, sampler_student, num_imgs=100, name=run_name, instance = 0, steps=[ddim_steps_student])
                

                for i, step in enumerate(step_sizes):
                    if instance != 0 and "gradual" not in step_scheduler:
                        util.save_model(sampler_student, optimizer, scheduler, name=step_scheduler, steps=updates, run_name=run_name)
                    updates = int(step / 2)
                    generations = update_list[i]
                    print("Distilling to:", updates)
                    
                    
                    
                    
                    with tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))) as tepoch:

                        for i, class_prompt in enumerate(tepoch):
                            generation += 1
                            losses = []        
                            xc = torch.tensor([class_prompt])
                            c_student = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                            samples_ddim= None
                            predictions_temp = []
                            sc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})
                            for steps in range(updates):  
                                    
                                        with autocast() and torch.enable_grad():

                                            # with torch.enable_grad():
                                                optimizer.zero_grad()
                                                instance += 1
                                            
                                                
                                                samples_ddim, pred_x0_student, _, at= sampler_student.sample_student(S=1,
                                                                                    conditioning=c_student,
                                                                                    batch_size=1,
                                                                                    shape=[3, 64, 64],
                                                                                    verbose=False,
                                                                                    x_T=samples_ddim,
                                                                            
                                                                                    unconditional_guidance_scale=scale,
                                                                                    unconditional_conditioning=sc, 
                                                                                    # unconditional_conditioning=None, 
                                                                                    eta=ddim_eta,
                                                                                    keep_intermediates=False,
                                                                                    intermediate_step = steps*2,
                                                                                    steps_per_sampling = 1,
                                                                                    total_steps = step)
                                                # decode_student = student.differentiable_decode_first_stage(pred_x0_student)
                                                # # # # # decode_student = student.differentiable_decode_first_stage(samples_ddim)
                                                # reconstruct_student = torch.clamp((decode_student+1.0)/2.0, min=0.0, max=1.0)
                                                # # print(pred_x0_student.grad)

                                                with torch.no_grad():
                                                    samples_ddim.detach()
                                                    samples_ddim, _, _, pred_x0_teacher, _ = sampler_student.sample(S=1,
                                                                                conditioning=c_student,
                                                                                batch_size=1,
                                                                                shape=[3, 64, 64],
                                                                                verbose=False,
                                                                            
                                                                                x_T=samples_ddim,
                                                                                unconditional_guidance_scale=scale,
                                                                                unconditional_conditioning=sc, 
                                                                                # unconditional_conditioning=None, 
                                                                                eta=ddim_eta,
                                                                                keep_intermediates=False,
                                                                                intermediate_step = steps*2+1,
                                                                                steps_per_sampling = 1,
                                                                                total_steps = step)     

                                                    # decode_teacher = student.decode_first_stage(pred_x0_teacher)
                                                    # # # # decode_teacher = student.differentiable_decode_first_stage(samples_ddim)
                                                    # reconstruct_teacher = torch.clamp((decode_teacher+1.0)/2.0, min=0.0, max=1.0)
                                            
                                    
                                            # with torch.enable_grad():    
                                            
                                            
                                                # # AUTOCAST:
                                                # signal = at
                                                # noise = 1 - at
                                                # log_snr = torch.log(signal / noise)
                                                # weight = max(log_snr, 1)
                                                # loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                                # scaler.scale(loss).backward()
                                                # scaler.step(optimizer)
                                                # scaler.update()
                                                # # torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                                
                                                # scheduler.step()
                                                # losses.append(loss.item())

                                                
                                                # NO AUTOCAST:
                                                
                                                # signal = at ** 2
                                                # noise = 1 - signal
                                                # log_snr = torch.log(signal / noise)
                                                # # weight = max(log_snr, 1)
                                                # weight = torch.exp(log_snr)
                                                # # weight = max(torch.exp(log_snr), 1)
                                                # # print(weight)
                                                
                                                loss = criterion(pred_x0_student, pred_x0_teacher.detach())
                                                # loss = criterion(pred_x0_student, pred_x0_teacher.detach())
                                                # loss = weight * criterion(samples_ddim_student, samples_ddim_teacher.detach())
                                                # loss = criterion(decode_student, decode_teacher.detach())
                                                # loss = weight * criterion(reconstruct_student, reconstruct_teacher.detach())
                                                # loss = criterion(reconstruct_student, reconstruct_teacher.detach())
                                                
                                                
                                                loss.backward()
                                                optimizer.step()
                                                scheduler.step()
                                                # print(scheduler.get_last_lr())
                                                torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                                
                                                losses.append(loss.item())



                                                # print("model identical:", torch.all(sampler_student.model.state_dict()["model.diffusion_model.output_blocks.2.0.in_layers.2.weight"] == sampler_original.model.state_dict()["model.diffusion_model.output_blocks.2.0.in_layers.2.weight"]))
                                            
                                        
                                        
                                        if session != None and generation % 200 == 0 and generation > 0:
                                                
                                            x_T_teacher_decode = sampler_student.model.decode_first_stage(pred_x0_teacher)
                                            teacher_target = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                            x_T_student_decode = sampler_student.model.decode_first_stage(pred_x0_student.detach())
                                            student_target  = torch.clamp((x_T_student_decode +1.0)/2.0, min=0.0, max=1.0)
                                            predictions_temp.append(teacher_target)
                                            predictions_temp.append(student_target)
                                            
                                        
                                    

                                        # if session != None and instance % 10000 == 0 and generation > 0:
                                        #     fids = util.get_fid(student, sampler_student, num_imgs=100, name=run_name, instance = instance+1, steps=[64, 32, 16, 8, 4, 2, 1])
                                        #     session.log({"fid_64":fids[0]})
                                        #     session.log({"fid_32":fids[1]})
                                        #     session.log({"fid_16":fids[2]})
                                        #     session.log({"fid_8":fids[3]})
                                        #     session.log({"fid_4":fids[4]})
                                        #     session.log({"fid_2":fids[5]})
                                        #     session.log({"fid_1":fids[6]})
                                        
                                        if session != None and instance % 100 == 0:
                        
                                            with torch.no_grad():
                                                images, _ = util.compare_teacher_student_x0(original, sampler_original, student, sampler_student, steps=[16, 8,  4, 2, 1], prompt=992)
                                                images = wandb.Image(_, caption="left: Teacher, right: Student")
                                                wandb.log({"pred_x0": images})
                                                # images, _ = util.compare_teacher_student_with_schedule(original, sampler_original, student, sampler_student, steps=[64, 32, 16, 8,  4, 2, 1], prompt=992)
                                                # images = wandb.Image(_, caption="left: Teacher, right: Student")
                                                # wandb.log({"schedule": images})
                                                sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                                                sampler_original.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                                                images, _ = util.compare_teacher_student(original, sampler_original, student, sampler_student, steps=[16, 8,  4, 2, 1], prompt=992)
                                                images = wandb.Image(_, caption="left: Teacher, right: Student")
                                                wandb.log({"pred": images})
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
                                    print("steps decreased:", ddim_steps_student)    

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

                if step_scheduler == "naive" or "gradual" in step_scheduler:
                    util.save_model(sampler_student, optimizer, scheduler, name=step_scheduler, steps=updates, run_name=run_name)

                                                                           
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
        halvings = math.floor(math.log(64)/math.log(2))
        updates_per_halving = int(gradient_updates / halvings)
        step_sizes = []
        for i in range(halvings):
            step_sizes.append(int((steps) / (2**i)))
        update_list = []
        for i in step_sizes:
            update_list.append(int(updates_per_halving / int(i/ 2)))
    elif step_scheduler == "naive":
        step_sizes=[ddim_steps_student]
        update_list=[gradient_updates // int(ddim_steps_student / 2)]
    elif step_scheduler == "gradual_linear":
        step_sizes = np.arange(steps, 0, -2)
        update_list = (1/len(np.append(step_sizes[1:], 1)) * gradient_updates / np.append(step_sizes[1:], 1)).astype(int)
    elif step_scheduler == "gradual_exp":
        step_sizes = np.arange(64, 0, -2)
        update_list = np.exp(1 / np.append(step_sizes[1:],1)) / np.sum(np.exp(1 / np.append(step_sizes[1:],1)))
        update_list = (update_list * gradient_updates /  np.append(step_sizes[1:],1)).astype(int)

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

                for i, step in enumerate(step_sizes):
                    if instance != 0 and "gradual" not in step_scheduler:
                        util.save_model(sampler_student, optimizer, scheduler, name=step_scheduler, steps=updates, run_name=run_name)
                    updates = int(step / 2)
                    generations = update_list[i]
                    print("Distilling to:", updates)
                  
               
                    with tqdm.tqdm(range(generations)) as tepoch:

                        for j in tepoch:
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
                                                                                total_steps = step)
                                        

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
                                                                            total_steps = step)     
                                        
                                        
                                    
                                        with torch.enable_grad():    
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
                                            
                                        with torch.no_grad():
                                            if session != None and generation % 100 == 0:
                                                    
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
                                    if session != None and generation % 100 == 0:
                                        images, _ = util.compare_teacher_student_celeb(original, sampler_original, student, sampler_student, steps=[64, 32, 16, 8,  4, 2, 1])
                                        images = wandb.Image(_, caption="left: Teacher, right: Student")
                                        wandb.log({"pred_x0": images})
                                        img, grid = util.compare_latents(predictions_temp)
                                        images = wandb.Image(grid, caption="left: Teacher, right: Student")
                                        wandb.log({"Inter_Comp": images})
                                        sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                                        sampler_original.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                                        del img, grid, predictions_temp, x_T_student_decode, x_T_teacher_decode, student_target, teacher_target
                                        torch.cuda.empty_cache()
                            
                            all_losses.extend(losses)
                            averaged_losses.append(sum(losses) / len(losses))
                            if session != None:
                                session.log({"generation_loss":averaged_losses[-1]})
                            tepoch.set_postfix(epoch_loss=averaged_losses[-1])

                if step_scheduler == "naive" or "gradual" in step_scheduler:
                    util.save_model(sampler_student, optimizer, scheduler, name=step_scheduler, steps=updates, run_name=run_name)