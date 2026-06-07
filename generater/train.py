import logging
import math
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache' # Set before running any code
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
from pathlib import Path
from typing import Iterable, Optional
from tqdm.auto import tqdm
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from config import parse_args
from utils_model import save_model, load_model

import wandb
from PIL import Image

args = parse_args()
# =====================================================================
# Directly hardcode hyperparameters here: Heterogeneous Federated Learning mode (Heterogeneous FL)
# =====================================================================
args.pretrained_model_name_or_path = r"diffmodels\stable-diffusion-v1-5"
args.dataset = "ca"
args.domains = ["blue", "green", "red", "nir"]
args.categories =[f"{i:04d}" for i in range(1, 501)]

# New output folder for storing heterogeneous training weights
args.output_dir = "./outputs/ca"
args.logging_dir = "logs"

args.train_type = "prompt"
args.num_shot = 1
args.client_num = 4          # Exactly 4 Clients corresponding to 4 spectral bands

args.train_batch_size = 1
args.gradient_accumulation_steps = 4
args.mixed_precision = "bf16"
args.gradient_checkpointing = True
args.resolution = 512

args.learning_rate = 1e-4
args.lr_scheduler = "constant"
args.lr_warmup_steps = 0
args.adam_beta1 = 0.9
args.adam_beta2 = 0.999
args.adam_weight_decay = 1e-2
args.adam_epsilon = 1e-08
args.max_train_steps = 1000
args.num_train_epochs = 100
args.scale_lr = False

args.seed = 42
args.report_to = "tensorboard"
args.center_crop = True
args.random_flip = True
args.skip_evaluation = False
args.log_every_steps = 50
args.log_every_epochs = 1
# =====================================================================
logger = get_logger(__name__)

def get_prompt_embeddings(prompt_domain, prompt_class, labels, tokenizer,
                          text_encoder, padding_type="do_not_pad",
                          num_prompt_class=None, num_prompt_domain=None):
    prompt_init =[]
    padding = True
    max_length = None

    for cid in labels:
        if args.dataset in ['bloodmnist', 'dermamnist', 'ucm']:
            c = args.categories[cid].lower().replace("_", " ")
            max_length = tokenizer.model_max_length
            if args.dataset=='dermamnist':
                prompt_init.append(f'A dermatoscopic image of a {c}, a type of pigmented skin lesions')
            elif args.dataset=='bloodmnist':
                prompt_init.append(f'A microscopic image of a {c}, a type of blood cell')
            else:
                prompt_init.append(f'A centered satellite photo of a {c.lower().replace("_", " ")}')
        else:
            # X represents a placeholder, which will be replaced by actual trainable tensors during forward propagation
            prompt_init.append('a X illuminated grayscale image of a human hand palmprint, showing detailed palm lines and wrinkles, black and white biometric scan of X')

    inputs = tokenizer(prompt_init,
        padding=padding,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = torch.LongTensor(inputs.input_ids)
    text_f = text_encoder(input_ids.to('cuda'))[0]

    if args.dataset in ['bloodmnist', 'dermamnist', 'ucm']:
        st_idx_map_class = {'bloodmnist': 7, 'dermamnist': 8, 'ucm': 7}
        start_idx = st_idx_map_class[args.dataset]
        for idx, cid in enumerate(labels):
            text_f[idx][start_idx:start_idx+num_prompt_class[cid]] = prompt_class[cid]

        start_idx_domain = 2
        num_prompt_domain_map = {'dermamnist': 4, 'ucm': 3}
        num_prompt_domain = num_prompt_domain_map[args.dataset]
        for idx, cid in enumerate(labels):
            text_f[idx][start_idx_domain:start_idx_domain+num_prompt_domain] = prompt_domain

    else:
        num_prompt_domain = 1
        text_f[:, 2:2+num_prompt_domain] = prompt_domain.unsqueeze(0).repeat(labels.shape[0], 1, 1)
        num_prompt_class = 1
        text_f[:, -1-num_prompt_class:-1] = prompt_class[labels]

    return text_f

def main():
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    #tokenizer = CLIPTokenizer.from_pretrained(
    #    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    #)
    #text_encoder = CLIPTextModel.from_pretrained(
    #    args.pretrained_model_name_or_path,
    #    subfolder="text_encoder",
    #    revision=args.revision,
    #)
    #vae = AutoencoderKL.from_pretrained(
    #    args.pretrained_model_name_or_path,
    #    subfolder="vae",
    #    revision=args.revision,
    #)
    #unet = UNet2DConditionModel.from_pretrained(
    #    args.pretrained_model_name_or_path,
    #    subfolder="unet",
    #    revision=args.revision,
    #)
    from diffusers import StableDiffusionPipeline

    # Please replace the path below with the absolute path to your v1-5-pruned-emaonly-fp16.safetensors file
    single_file_path = r"E:\models\stable-diffusion-v1-5\v1-5-pruned-emaonly-fp16.safetensors"
    # Local complete model directory containing config files
    local_model_dir = r"E:\models\stable-diffusion-v1-5"
    print(f"Loading model from single file: {single_file_path}")
    pipe = StableDiffusionPipeline.from_single_file(
        single_file_path,
        config=local_model_dir,        # Key: specify reading various json config files from local directory
        local_files_only=True,         # Key: disable all network requests
        safety_checker=None,           # Key: in newer versions, use None to completely disable the safety checker
        requires_safety_checker=False  # Key: explicitly disable in conjunction
    )

    # Directly extract the four core components needed by subsequent code from the pipeline
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    # Release pipe to free memory (optional)
    del pipe

    def tokenize_captions(examples, is_train=True):
        captions =[]
        for caption in examples:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example[1] for example in examples]
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        domain_ids = torch.tensor([example[2] for example in examples])
        class_ids = torch.tensor([example[3] for example in examples])
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
            "domain_ids": domain_ids,
            "class_ids": class_ids,
        }

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_transforms = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if args.dataset=='domainnet':
        from domainnet_data import get_dataloader, get_dataloader_domain

    elif args.dataset=='officehome':
        from officehome_data import get_dataloader, get_dataloader_domain
    elif args.dataset=='ucm':
        from ucm_data import get_dataloader, get_dataloader_domain
    elif args.dataset=='dermamnist':
        from dermamnist_data import get_dataloader, get_dataloader_domain
    elif args.dataset=='bloodmnist':
        from bloodmnist_data import get_dataloader, get_dataloader_domain
    else :
        from pacs_data import get_dataloader, get_dataloader_domain
    split = 'train'
    trainloaders =[]

    # Core modification 1: Let 4 Clients each load 4 different spectral data
    for i in range(args.client_num):
        client_domain = args.domains[i]  # 0:Blue, 1:Green, 2:Red, 3:NIR
        trainloader = get_dataloader_domain(
                args, args.train_batch_size, None,
                'train', client_domain, tokenize_captions,
                collate_fn, num_shot=args.num_shot)
        trainloaders.append(trainloader)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(trainloaders[-1]) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    device=torch.device("cuda")
    print("Start Heterogeneous Training...")

    for idx, train_dataloader in enumerate(trainloaders):
        prompt_init =[]
        global_step = 0
        loss_history=[]
        train_loss = 0.0
        curious_time=0
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

        # Core modification 2: Get the physical spectrum corresponding to the current Client
        current_domain = args.domains[idx]
        progress_bar.set_description(f"Client {idx} ({current_domain})")

        for c in args.categories:
            if args.dataset=='ucm':
                prompt_init.append(f'A centered satellite photo of a {c.lower().replace("_", " ")}')
            elif args.dataset=='bloodmnist':
                prompt_init.append(f'A microscopic image of a {c.lower().replace("_", " ")}, a type of blood cell')
            elif args.dataset=='dermamnist':
                prompt_init.append(f'A dermatoscopic image of a {c}, a type of pigmented skin lesions')
            else:
                # Core modification 3: Initialize soft prompt with current spectrum name, letting the model know which light band it is looking at
                prompt_init.append(f'a {current_domain.lower()} illuminated grayscale image of a human hand palmprint, showing detailed palm lines and wrinkles, black and white biometric scan of {c.lower()}')

        inputs = tokenizer(prompt_init,
            max_length=tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        text_f = text_encoder(inputs.input_ids.to(accelerator.device))[0]
        num_prompt_class = None
        num_prompt_domain = 1
        prompt_domain = text_f[0][2:2+num_prompt_domain]
        num_prompt_class = 1
        prompt_class = text_f[:, -1-num_prompt_class:-1]
        prompt_domain.requires_grad_(True)
        prompt_class.requires_grad_(True)
        trainable_params = [prompt_domain, prompt_class]

        optimizer = torch.optim.Adam(
                trainable_params,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        # --- Performance statistics initialization (Client level) ---
        step_counter = 0
        perf_data_load_time = 0.0
        perf_prompt_embed_time = 0.0
        perf_fwd_bwd_time = 0.0

        # Create CUDA Events for high-precision hardware timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        t_batch_start = torch.cuda.Event(enable_timing=True)
        t_batch_end = torch.cuda.Event(enable_timing=True)

        # --- Performance statistics initialization (Client level) ---
        step_counter = 0
        perf_data_load_time = 0.0
        perf_prompt_embed_time = 0.0
        perf_fwd_bwd_time = 0.0

        # Create CUDA Events for high-precision hardware timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Instantiate data-loading-specific timestamps
        start_time_stamp = torch.cuda.Event(enable_timing=True)
        end_time_stamp = torch.cuda.Event(enable_timing=True)

        for epoch in range(args.num_train_epochs):
            unet.train()

            for step, batch in enumerate(train_dataloader):
                # ====== 1. Data loading & VAE encoding overhead ======
                if step > 0:
                    # Core fix: only from the second step onwards, record the end anchor and calculate the time difference
                    end_time_stamp.record()
                    torch.cuda.synchronize()
                    perf_data_load_time += start_time_stamp.elapsed_time(end_time_stamp)

                # VAE feature extraction
                latents = vae.encode(batch["pixel_values"].to(weight_dtype).to(device)).latent_dist.sample()
                latents = latents * 0.18215

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                labels = batch["class_ids"].to(device)

                # ====== 2. Soft prompt (Prompt Embeddings) composition overhead ======
                start_event.record()
                if 'concept' in args.train_type:
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                    class_concepts = unet.one_hot_concept[labels]
                    batch["input_conditions"] = class_concepts.to(device)
                elif 'prompt' in args.train_type:
                    encoder_hidden_states = get_prompt_embeddings(prompt_domain, prompt_class, labels, tokenizer, text_encoder, num_prompt_class=num_prompt_class)
                    batch["input_conditions"] = None
                end_event.record()
                torch.cuda.synchronize()
                perf_prompt_embed_time += start_event.elapsed_time(end_event)

                # ====== 3. UNet forward & backward propagation overhead ======
                start_event.record()
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                train_loss += loss.item()
                curious_time += timesteps.sum().item()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                end_event.record()
                torch.cuda.synchronize()
                perf_fwd_bwd_time += start_event.elapsed_time(end_event)

                progress_bar.update(1)
                global_step += 1
                step_counter += 1  # Accumulate step count

                # ====== 4. Output performance analysis report every 100 steps ======
                if step_counter % 100 == 0:
                    total_step_images = 100 * args.train_batch_size
                    print(f"\n\n" + "="*60)
                    print(f"[Client {idx} - {current_domain}] Last 100 steps training overhead analysis report")
                    print("="*60)
                    print(f" Total images processed during period: {total_step_images} images")
                    print(f" Total hardware time for 100 steps: {(perf_data_load_time + perf_prompt_embed_time + perf_fwd_bwd_time)/1000:.2f} seconds")
                    print(f" Average throughput per image: {((perf_data_load_time + perf_prompt_embed_time + perf_fwd_bwd_time) / total_step_images):.2f} ms")
                    print("-"*60)
                    print(" Core performance bottleneck breakdown (average per image):")
                    print(f"  |-- Data loading & VAE encoding: {(perf_data_load_time / total_step_images):.2f} ms")
                    print(f"  |-- Prompt soft prompt construction: {(perf_prompt_embed_time / total_step_images):.2f} ms")
                    print(f"  |-- UNet forward+backward propagation: {(perf_fwd_bwd_time / total_step_images):.2f} ms")
                    print("="*60 + "\n")

                    # After reporting, clear the current cycle counters and enter the next 100-step statistics
                    perf_data_load_time = 0.0
                    perf_prompt_embed_time = 0.0
                    perf_fwd_bwd_time = 0.0

                if global_step % 1 == 0:
                    train_loss = train_loss/1
                    accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    loss_history.append(train_loss)
                    train_loss = 0.0
                    curious_time = 0

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

                if not args.skip_evaluation and (global_step)%args.log_every_steps==0:
                    if 'concept' in args.train_type:
                        save_model(unet, args.output_dir+'/unet.pth')
                    elif 'prompt' in args.train_type:
                        torch.save(prompt_domain, args.output_dir+f'/prompt_domain_{idx}.pth')
                        torch.save(prompt_class, args.output_dir+f'/prompt_class_{idx}.pth')

                plt.figure()
                plt.plot(loss_history)
                plt.savefig(args.output_dir+f'/loss_history_client_{idx}.png')
                plt.close()

                # Fix: mark the start timestamp for data loading of the next image here (the first step gets stamped at the end, the second step takes effect)
                start_time_stamp.record()

        if epoch%args.log_every_epochs==0 or epoch==args.num_train_epochs-1:
            if 'concept' in args.train_type:
                save_model(unet, args.output_dir+'/unet.pth')
            elif 'prompt' in args.train_type:
                torch.save(prompt_domain, args.output_dir+f'/prompt_domain_{idx}.pth')
                torch.save(prompt_class, args.output_dir+f'/prompt_class_{idx}.pth')

if __name__ == "__main__":
    main()