import argparse
import self_distillation
import distillation
import saving_loading
import generate
import wandb
import util
import os

cwd = os.getcwd()

parser = argparse.ArgumentParser(description='Direct Self-Distillation')


parser.add_argument('--task', '-t', type=str, default= "DSDI", help='Task to perform', choices=['TSD', "DSDN", "DSDI", "DSDGL", "DSDGEXP", "SI", "SI_orig", "CD", "DFD", "FID", "NPZ", "NPZ_single", "retrain"])
parser.add_argument('--model', '-m', type=str, default= "cin", help='Model type', choices=['cin', 'celeb'])
parser.add_argument('--steps', '-s', type=int, default= 64, help='DDIM steps to distill from')
parser.add_argument('--updates', '-u', type=int, default= 100000, help='Number of total weight updates')
parser.add_argument('--learning_rate', '-lr', default= 0.000000002, type=float, help='Learning Rate')
parser.add_argument('--cas', '-c', type=bool, default= False, help='Include Cosine Annealing Scheduler for learning rate')
parser.add_argument('--name', '-n', type=str, help='Name to give the run, or type of run to save')
parser.add_argument('--save', '-sv', type=bool, default= True, help='Save intermediate models')
parser.add_argument('--compare', type=bool, default= True, help='Compare to original model')
parser.add_argument('--wandb', '-w', type=bool, default=True, help='Weights and Biases upload')
parser.add_argument('--cuda', '-cu', type=str, default="True", help='Cuda on/off')
parser.add_argument('--predict', '-pred', type=str, default="x0", help='either x0 of eps prediction, x0 uses the retrained model, eps uses the original model')
parser.add_argument('--pixels', '-p', type=int, default=256, help='256/64 pixel outputs')


if __name__ == '__main__':
    
    os.environ['WANDB_NOTEBOOK_NAME'] = "Cin_256_custom.ipynb"
    args = parser.parse_args()

    if "False" in args.cuda:
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'
        print("Running on CPU")
    else:
        device = 'cuda'
    # Instatiate selected model
    if args.pixels == 256:
        if args.model == "cin":
            if args.predict == "x0":
                model_path=f"{cwd}/models/cin256_retrained.pt"
                config_path = f"{cwd}/models/configs/cin256-v2-custom_x0.yaml"
            else:
                config_path=f"{cwd}/models/configs/cin256-v2-custom.yaml"
                model_path=f"{cwd}/models/cin256_original.ckpt"
            npz = f"{cwd}/val_saved/real_fid_both.npz"
        elif args.model == "celeb":
            config_path=f"{cwd}/models/configs/celebahq-ldm-vq-4.yaml"
            model_path=f"{cwd}/models/CelebA.ckpt"
            npz = f"{cwd}/val_saved/celeb.npz"
            # npz = f"C:\Diffusion_Thesis\cin_256\celeba_hq_256"
            # npz = f"C:\Diffusion_Thesis\cin_256\celeb_64.npz"
    elif args.pixels == 64:
        print("64 model")
        if args.model == "cin":
            config_path=f"{cwd}/models/configs/cin256-v2-custom.yaml"
            model_path=f"{cwd}/models/64x64_diffusion.pt"
            npz = f"{cwd}/val_saved/real_fid_both.npz"
        elif args.model == "celeb":
            config_path=f"{cwd}/models/configs/celebahq-ldm-vq-4.yaml"
            model_path=f"{cwd}/models/CelebA.ckpt"
            npz = f"{cwd}/val_saved/celeb.npz"
            # npz = f"C:\Diffusion_Thesis\cin_256\celeba_hq_256"
            # npz = f"C:\Diffusion_Thesis\cin_256\celeb_64.npz"



    # Start Task


    if args.task == "retrain":
        # print("RETRAINING USING PREVIOUSLY TRAINED MODEL")
        if args.name is None:
            args.name = f"{args.model}_retrain_{args.steps}_{args.learning_rate}_{args.updates}"
        
        
        # teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        # if args.compare:
        #     original, sampler_original = util.create_models(config_path, model_path, student=False)
        
        if args.model == "cin":
            distillation.retrain(ddim_steps=args.steps, generations=args.updates, run_name=args.name, config=config_path, 
                    original_model_path=model_path, lr=args.learning_rate, start_trained=False, cas=args.cas, compare=args.compare, use_wandb=args.wandb)
        else:
            distillation.distill_celeb(ddim_steps=args.steps, generations=args.updates, run_name=args.name, config=config_path, 
                    original_model_path=model_path, lr=args.learning_rate, start_trained=False, cas=args.cas, compare=args.compare, use_wandb=args.wandb)



    if args.task == "TSD":
        
        if args.name is None:
            args.name = f"{args.model}_TSD_{args.predict}_{args.steps}_{args.learning_rate}_{args.updates}"
        

        # teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        # if args.compare:
        #     original, sampler_original = util.create_models(config_path, model_path, student=False)
        
        if args.model == "cin":
            distillation.distill(args, config=config_path, original_model_path=model_path, start_trained=False)
        else:
            distillation.distill_celeb(ddim_steps=args.steps, generations=args.updates, run_name=args.name, config=config_path, 
                    original_model_path=model_path, lr=args.learning_rate, start_trained=False, cas=args.cas, compare=args.compare, use_wandb=args.wandb)


    elif args.task == "DSDN":
        if args.name is None:
            args.name = f"{args.model}_DSDN_{args.steps}_{args.learning_rate}_{args.updates}"

        teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        if args.compare:
            original, sampler_original = util.create_models(config_path, model_path, student=False)

        step_scheduler = "naive"
        decrease_steps = True
        optimizer, scheduler = util.get_optimizer(sampler_teacher, iterations=args.updates, lr=args.learning_rate)
        wandb_session = util.wandb_log(name=args.name, lr=args.learning_rate, model=teacher, tags=["DSDN"], 
                notes=f"Direct Naive Self-Distillation from {args.steps} steps with {args.updates} weight updates",  project="Self-Distillation")
        wandb.run.log_code(".")
        
        if args.model == "cin":
            self_distillation.self_distillation_CIN(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler, x0=args.predict)
        elif args.model == "celeb":
            self_distillation.self_distillation_CELEB(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler)

    elif args.task == "DSDI":

        if args.name is None:
            args.name = f"{args.model}_DSDI_{args.steps}_{args.learning_rate}_{args.updates}"

        teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        if args.compare:
            original, sampler_original = util.create_models(config_path, model_path, student=False)

        step_scheduler = "iterative"
        decrease_steps = True
        optimizer, scheduler = util.get_optimizer(sampler_teacher, iterations=args.updates, lr=args.learning_rate)
        wandb_session = util.wandb_log(name=args.name, lr=args.learning_rate, model=teacher, tags=["DSDI"], 
                notes=f"Direct Iterative Self-Distillation from {args.steps} steps with {args.updates} weight updates",  project="Self-Distillation")
        wandb.run.log_code(".")
        
        if args.model == "cin":
            self_distillation.self_distillation_CIN(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, gradient_updates=args.updates, run_name=args.name, step_scheduler=step_scheduler)
        elif args.model == "celeb":
            self_distillation.self_distillation_CELEB(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, gradient_updates=args.updates, run_name=args.name, step_scheduler=step_scheduler)
            
    elif args.task == "DSDGL":

        if args.name is None:
            args.name = f"{args.model}_DSDGL_{args.steps}_{args.learning_rate}_{args.updates}"

        teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        if args.compare:
            original, sampler_original = util.create_models(config_path, model_path, student=False)

        step_scheduler = "gradual_linear"
        decrease_steps = True
        optimizer, scheduler = util.get_optimizer(sampler_teacher, iterations=args.updates, lr=args.learning_rate)
        wandb_session = util.wandb_log(name=args.name, lr=args.learning_rate, model=teacher, tags=["DSDGL"], 
                notes=f"Direct Gradual Linear Self-Distillation from {args.steps} steps with {args.updates} weight updates",  project="Self-Distillation")
        wandb.run.log_code(".")
        
        if args.model == "cin":
            self_distillation.self_distillation_CIN(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, gradient_updates=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler)
        elif args.model == "celeb":
            self_distillation.self_distillation_CELEB(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, gradient_updates=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler)

    elif args.task == "DSDGEXP":

        if args.name is None:
            args.name = f"{args.model}_DSDGEXP_{args.steps}_{args.learning_rate}_{args.updates}"

        teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        if args.compare:
            original, sampler_original = util.create_models(config_path, model_path, student=False)

        step_scheduler = "gradual_exp"
        decrease_steps = True
        optimizer, scheduler = util.get_optimizer(sampler_teacher, iterations=args.updates, lr=args.learning_rate)
        wandb_session = util.wandb_log(name=args.name, lr=args.learning_rate, model=teacher, tags=["DSDGEXP"], 
                notes=f"Direct Gradual Exp Self-Distillation from {args.steps} steps with {args.updates} weight updates",  project="Self-Distillation")
        wandb.run.log_code(".")
        
        if args.model == "cin":
            self_distillation.self_distillation_CIN(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler)
        elif args.model == "celeb":
            self_distillation.self_distillation_CELEB(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler)

    elif args.task == "SI":
        
        import torch
        from omegaconf import OmegaConf
        from ldm.models.diffusion.ddim import DDIMSampler
        # config_path=f"{cwd}/models/configs/cin256-v2-custom.yaml"
        # config = OmegaConf.load(config_path)  
        if args.updates == 100000:
            print("Doing 100k, did you mean to do this? Change -u for a specific amount of generated images")
        start_path = f"{cwd}/data/trained_models/final_versions/{args.model}/"
        for train_type in os.listdir(start_path):
            if args.name != None:
                if args.name != train_type:
                    continue
            print(train_type)
            model_path = f"{start_path}{train_type}"
            model_name = f"{os.listdir(model_path)[0]}"
            model_path = f"{model_path}/{model_name}"

            config = OmegaConf.load(config_path)  
            if device == "cuda":
                ckpt = torch.load(model_path)
            else:
                ckpt = torch.load(model_path, map_location=torch.device("cpu"))
            model = saving_loading.instantiate_from_config(config.model)
            model.to(device)
            if device == "cuda":
                model.cuda()
            else:
                model.cpu()
                model.to(torch.device("cpu"))
            model.load_state_dict(ckpt["model"], strict=False)
            model.eval()
            sampler = DDIMSampler(model)
            # model, sampler, optimizer, scheduler = util.load_trained(config_path, model_path)
            if args.model == "cin":
              util.save_images(model, sampler, args.updates, train_type, [4,8], verbose=True)
            else:
              saving_loading.save_images(model, sampler, args.updates, train_type, [4,8], verbose=True, celeb=True)
            del model, sampler, ckpt#, optimizer, scheduler
            torch.cuda.empty_cache()

    elif args.task == "SI_orig":
        import torch
        from omegaconf import OmegaConf
        from ldm.models.diffusion.ddim import DDIMSampler
        # config_path=f"{cwd}/models/configs/cin256-v2-custom.yaml"
        # config = OmegaConf.load(config_path)  
        if args.updates == 100000:
            print("Doing 100k, did you mean to do this? Change -u for a specific amount of generated images")
         
        # model, sampler, optimizer, scheduler = util.load_trained(config_path, model_path)
        if args.model == "cin":
            original, sampler_original = util.create_models(config_path, model_path, student=False)
            original.eval()
            original
            if device == "cuda":
                original.cuda()
            else:
                original.cpu()
            util.save_images(original, sampler_original, args.updates,"cin_original", [2,4,8,16],verbose=True)
        else:
            config_path=f"{cwd}/models/configs/celebahq-ldm-vq-4.yaml"
            model_path=f"{cwd}/models/CelebA.ckpt"
            original, sampler_original = util.create_models(config_path, model_path, student=False)
            util.save_images(original, sampler_original, args.updates,"celeb_original", [2,4,8,16], verbose=True)
        del original, sampler_original
        torch.cuda.empty_cache()

    elif args.task == "FID":
        import pandas as pd
        import os
        import torch_fidelity
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        filename = 'metrics.csv'
  
        if not os.path.isfile(filename):
            df = pd.DataFrame({
                "model": [],
                "type" : [],
                "step": [],
                "fid": [],
                "isc": [],
                "kid": []
            })
            df.to_csv(filename, index=False)
        df = pd.read_csv(filename)
        cin_target = r"C:\imagenet\val"
        celeb_target = r"C:\Diffusion_Thesis\cin_256\celeba_hq_256"
        # cin_target = r"C:\imagenet\DIFFERENCE\fill"
        # celeb_target = r"C:\imagenet\DIFFERENCE\fill"
        for model in ["cin", "celeb"]:  
            target = cin_target if model == "cin" else celeb_target 
            basic_path_source = f"{cwd}/saved_images/{model}/"
            model_names = [name for name in os.listdir(basic_path_source)]
         
            for model_name in model_names:
                model_path_source = basic_path_source + f"{model_name}/"
                steps = [step for step in os.listdir(model_path_source)]
                for step in steps:
                    current_path_source = model_path_source + f"{step}/"
                    if  df.loc[(df['model'] == model) & (df['step'] == step) & (df['type'] == model_name)].empty:
                        try:
                            metrics = torch_fidelity.calculate_metrics(gpu=0, fid=True, isc=True, kid=True, input1=current_path_source, input2=target)
                            metrics_df = pd.DataFrame({
                            "model" : [model],
                            "type": [model_name],
                            "step": [step],
                            "fid": [metrics["frechet_inception_distance"]],
                            "isc" :[metrics["inception_score_mean"]],
                            "kid": [metrics["kernel_inception_distance_mean"]]})                            
                            df = pd.concat([df, metrics_df])
                            df.to_csv(filename, index=False)
                        except Exception as e:
                            print("Failed to create metrics for:", current_path_source)
                            print(e)
                    else:
                        print("Already have metrics for:", current_path_source)
    



    elif args.task == "NPZ": 
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        for model in ["cin", "celeb"]:   
            basic_path_source = f"{cwd}/saved_images/{model}/"
            basic_path_target = f"{cwd}/NPZ/{model}"
            model_names = [name for name in os.listdir(basic_path_source)]
 
            for model_name in model_names:
                model_path_source = basic_path_source + f"{model_name}/"
                model_path_target = basic_path_target + f"_{model_name}"
                
                steps = [step for step in os.listdir(model_path_source)]
                for step in steps:
                    current_path_source = model_path_source + f"{step}/"
                    current_path_target = model_path_target + f"_{step}"
                    try:
                        util.generate_npz(current_path_source, current_path_target)
                   
                    except Exception as e:
                        print("Failed to generate npz for ", current_path_source)
                        print(e)

    elif args.task == "NPZ_single":
        
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        current_path_source = "C:\Diffusion_Thesis\cin_256\saved_images\cin\cin_original\\64"
        current_path_target = "C:\Diffusion_Thesis\cin_256\\NPZ\cin_cin_original_64"
        util.generate_npz(current_path_source, current_path_target)



# fidelity --gpu 0 --fid --input1 C:\Diffusion_Thesis\cin_256\saved_images\cin\cin_original\32 --input2 C:\imagenet\test2