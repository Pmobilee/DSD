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


parser.add_argument('--task', '-t', type=str, default= "DSDI", help='Task to perform', choices=['TSD', "DSDN", "DSDI", "DSDGL", "DSDGEXP", "SI", "CD", "DFD", "FID"])
parser.add_argument('--model', '-m', type=str, default= "cin", help='Model type', choices=['cin', 'celeb'])
parser.add_argument('--steps', '-s', type=int, default= 64, help='DDIM steps to distill from')
parser.add_argument('--updates', '-u', type=int, default= 100000, help='Number of total weight updates')
parser.add_argument('--learning_rate', '-lr', default= 0.000000002, type=float, help='Learning Rate')
parser.add_argument('--cas', '-c', type=bool, default= False, help='Include Cosine Annealing Scheduler for learning rate')
parser.add_argument('--name', '-n', type=str, help='Name to give the run, or type of run to save')
parser.add_argument('--save', '-sv', type=bool, default= True, help='Save intermediate models')
parser.add_argument('--compare', type=bool, default= True, help='Compare to original model')
parser.add_argument('--wandb', '-w', type=bool, default=True, help='Weights and Biases upload')


if __name__ == '__main__':
    
    os.environ['WANDB_NOTEBOOK_NAME'] = "Cin_256_custom.ipynb"
    args = parser.parse_args()

    # Instatiate selected model
    if args.model == "cin":
        config_path=f"{cwd}/models/configs/cin256-v2-custom.yaml"
        model_path=f"{cwd}/models/cin256_original.ckpt"
    elif args.model == "celeb":
        config_path=f"{cwd}/models/configs/celebahq-ldm-vq-4.yaml"
        model_path=f"{cwd}/models/CelebA.ckpt"


    # Start Task
    if args.task == "TSD":
        
        if args.name is None:
            args.name = f"{args.model}_TSD_{args.steps}_{args.learning_rate}_{args.updates}"
        

        # teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        # if args.compare:
        #     original, sampler_original = util.create_models(config_path, model_path, student=False)
        
        if args.model == "cin":
            distillation.distill(ddim_steps=args.steps, generations=args.updates, run_name=args.name, config=config_path, 
                    original_model_path=model_path, lr=args.learning_rate, start_trained=False, cas=args.cas, compare=args.compare, use_wandb=args.wandb)
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
                        steps=args.steps, generations=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler)
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
                        steps=args.steps, generations=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler)
        elif args.model == "celeb":
            self_distillation.self_distillation_CELEB(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler)
            
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
                        steps=args.steps, generations=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler)
        elif args.model == "celeb":
            self_distillation.self_distillation_CELEB(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name, decrease_steps=decrease_steps, step_scheduler=step_scheduler)

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
            ckpt = torch.load(model_path)
            model = saving_loading.instantiate_from_config(config.model)
            model.load_state_dict(ckpt["model"], strict=False)
            model.eval()
            model.cuda()
            sampler = DDIMSampler(model)
            
            # model, sampler, optimizer, scheduler = util.load_trained(config_path, model_path)
            if args.model == "cin":
              util.save_images(model, sampler, args.updates, train_type, [2,4,8,16], verbose=True)
            else:
              saving_loading.save_images(model, sampler, args.updates, train_type, [2,4,8,16], verbose=True, celeb=True)
            del model, sampler, ckpt#, optimizer, scheduler
            torch.cuda.empty_cache()