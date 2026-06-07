import logging
import os; os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch

from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from config import parse_args
from PIL import Image

def get_prompt_embeddings(prompt_domain, prompt_class, labels, tokenizer, text_encoder, args):
    prompt_init =[]
    for cid in labels:
        prompt_init.append('a X illuminated grayscale 2D flatbed scan of palmprint texture, extremely sharp focus, crisp lines, high contrast, macro photography, biometric fingerprint of X')

    inputs = tokenizer(prompt_init, padding=True, truncation=True, return_tensors="pt")
    input_ids = torch.LongTensor(inputs.input_ids)
    text_f = text_encoder(input_ids.to('cuda'))[0]

    num_prompt_domain = 1
    text_f[:, 2:3] = prompt_domain.unsqueeze(0).repeat(labels.shape[0], 1, 1)

    num_prompt_class = 1
    text_f[:, -2:-1] = prompt_class[labels]

    return text_f


def predict_cond(model, prompt_embeds, seed, init_image, img_size, num_inference_steps=20, negative_prompt=None, strength=0.8):
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

    output = model(
        image=init_image,
        strength=strength,
        prompt_embeds=prompt_embeds,
        num_inference_steps=num_inference_steps,
        generator=generator,
        negative_prompt=negative_prompt,
    )
    image = output.images[0]
    return image


def generate_for_all_clients(model, categories, device, args, tokenizer, text_encoder, all_domain_prompts, all_class_prompts, all_latents_mean):
    import time  # Import time module

    # Fix progress bar flooding: disable the diffusers pipeline's built-in progress bar for single images
    model.set_progress_bar_config(disable=True)

    # --- Performance statistics initialization ---
    total_start_time = time.time()  # Record the start time of the entire function
    latents_decode_times = []       # Store VAE decode time for each run (unit: ms)
    diffusion_inference_times = []  # Store Stable Diffusion reverse sampling time for each run (unit: ms)
    io_save_times = []              # Store image save-to-disk time for each run (unit: ms)
    total_generated_count = 0       # Actual total number of generated images

    for cid, c in enumerate(categories):
        print(f"\n=======================================================")
        print(f"generating ID: {c}")

        # Only generate for blue end (client 0)
        for idx in [0]:
            domain = args.domains[idx]

            # Modification point 1: Reassign domain fusion weights
            weights = [0.15] * 4
            weights[idx] = 0.55

            mixed_prompt_domain = sum(w * p for w, p in zip(weights, all_domain_prompts))
            mixed_prompt_class = sum(all_class_prompts) / 4.0

            save_image_dir = os.path.join(args.output_dir, c, f"Mixed_{domain}_client{idx}")
            os.makedirs(save_image_dir, exist_ok=True)

            latents_mean = all_latents_mean[idx]
            m_list = latents_mean.get(c, [])
            if len(m_list) == 0:
                c_stripped = str(int(c))
                m_list = latents_mean.get(c_stripped, [])

            if len(m_list) > 0:
                labels = torch.tensor([cid] * 1).to(device)

                prompt_embeds = get_prompt_embeddings(
                    mixed_prompt_domain, mixed_prompt_class, labels,
                    tokenizer, text_encoder, args)

                # Modification point 2: Create a tqdm progress bar for each Client
                pbar = tqdm(range(args.start_idx, args.end_idx + 1), desc=f"[Client {idx} - {domain}]", leave=True)

                for i in pbar:
                    seed = idx * 1000000 + cid * 1000 + i

                    latent_A = m_list[i % len(m_list)].to(device).squeeze(0)
                    latent_B = m_list[(i + 5) % len(m_list)].to(device).squeeze(0)

                    alpha = random.uniform(0.3, 0.7)
                    mixed_latent = alpha * latent_A + (1 - alpha) * latent_B

                    # Create CUDA Events for precise measurement of VAE and Diffusion GPU overhead
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    # ====== 1. VAE decoding overhead measurement ======
                    start_event.record()
                    with torch.no_grad():
                        scaled_latent = mixed_latent.unsqueeze(0) / model.vae.config.scaling_factor
                        scaled_latent = scaled_latent.to(dtype=model.vae.dtype)
                        decoded_tensor = model.vae.decode(scaled_latent).sample
                        decoded_tensor = (decoded_tensor / 2 + 0.5).clamp(0, 1)
                    end_event.record()
                    torch.cuda.synchronize()  # Force GPU synchronization to ensure timing accuracy
                    latents_decode_times.append(start_event.elapsed_time(end_event))

                    dynamic_strength = random.uniform(0.02, 0.05)

                    # ====== 2. Diffusion inference overhead measurement ======
                    start_event.record()
                    image = predict_cond(
                        model=model,
                        prompt_embeds=prompt_embeds,
                        seed=seed,
                        init_image=decoded_tensor,
                        img_size=args.resolution,
                        num_inference_steps=args.num_inference_steps,
                        negative_prompt=args.negative_prompt,
                        strength=dynamic_strength
                    )
                    end_event.record()
                    torch.cuda.synchronize()
                    diffusion_inference_times.append(start_event.elapsed_time(end_event))

                    # ====== 3. Disk I/O save overhead measurement ======
                    t_io_start = time.time()
                    img_path = os.path.join(save_image_dir, f"{i}.jpg")
                    image.save(img_path)
                    t_io_end = time.time()
                    io_save_times.append((t_io_end - t_io_start) * 1000) # Convert to milliseconds

                    total_generated_count += 1
            else:
                print(f"Warning: Skipping Client {idx} Class {c}: Cannot find corresponding latent features.")

    # =====================================================================
    # Performance overhead analysis report output
    # =====================================================================
    total_elapsed_time = time.time() - total_start_time

    if total_generated_count > 0:
        avg_vae = np.mean(latents_decode_times)
        avg_diff = np.mean(diffusion_inference_times)
        avg_io = np.mean(io_save_times)
        avg_total_per_img = (total_elapsed_time / total_generated_count) * 1000

        print("\n" + "="*60)
        print("                    Performance Overhead Analysis Report                    ")
        print("="*60)
        print(f" Total images successfully generated: {total_generated_count} images")
        print(f" Total runtime of entire process: {total_elapsed_time:.2f} seconds")
        print(f" Average total throughput per image: {avg_total_per_img:.2f} ms (seconds/image: {avg_total_per_img/1000:.3f}s)")
        print("-"*60)
        print(" Core compute/overhead breakdown (average per image):")
        print(f"  |-- VAE Latent decoding stage: {avg_vae:.2f} ms  [proportion {avg_vae / avg_total_per_img * 100:.1f}%]")
        print(f"  |-- UNet reverse iterative sampling: {avg_diff:.2f} ms [proportion {avg_diff / avg_total_per_img * 100:.1f}%]")
        print(f"  |-- Disk storage I/O stage: {avg_io:.2f} ms  [proportion {avg_io / avg_total_per_img * 100:.1f}%]")
        print(f"  |-- Framework scheduling and network overhead: {avg_total_per_img - avg_vae - avg_diff - avg_io:.2f} ms")
        print("="*60 + "\n")
    else:
        print("\nWarning: No images were actually generated, unable to generate overhead report.")

def main():
    args = parse_args()

    # =====================================================================
    # Hyperparameter settings
    # =====================================================================
    args.pretrained_model_name_or_path = r"E:\models\stable-diffusion-v1-5"
    args.dataset = "ca"
    args.domains = ["blue", "green", "red", "nir"]
    args.categories =[f"{i:04d}" for i in range(1, 101)]

    args.output_dir = "./ca"
    args.scheduler = "ddim"
    args.fp16 = True
    args.start_idx = 5
    args.end_idx = 5    # Only generate the 6th image
    args.resolution = 512
    args.num_inference_steps = 60

    args.negative_prompt = "blurry, out of focus, low contrast, soft edges, fingers, thumb, hand outline, deformed digits, horror, 3d, colorful, rgb, blue colorr"
    # =====================================================================

    logging.basicConfig(level=logging.INFO)
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    device = torch.device('cuda')
    base_output_path = os.path.join("outputs", "ca")

    all_domain_prompts = []
    all_class_prompts = []
    all_latents_mean = []

    print("Preloading all federated features and backbones for 4 spectral domains...")
    for idx in range(4):
        dp = torch.load(os.path.join(base_output_path, f"prompt_domain_{idx}.pth")).to(device)
        cp = torch.load(os.path.join(base_output_path, f"prompt_class_{idx}.pth")).to(device)
        all_domain_prompts.append(dp)
        all_class_prompts.append(cp)

        try:
            lm = torch.load(os.path.join(base_output_path, f"mean_{idx}.pt"))
            all_latents_mean.append(lm)
        except FileNotFoundError:
            print(f"Error: Cannot find Client {idx} latent feature file mean_{idx}.pt!")
            return
    print("Federated knowledge loading complete!\n")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", local_files_only=True, torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", local_files_only=True, torch_dtype=weight_dtype)

    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False,
        set_alpha_to_one=False, num_train_timesteps=1000, steps_offset=1,
    )

    model = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
        local_files_only=True, torch_dtype=weight_dtype
    ).to(device)

    generate_for_all_clients(
        model, args.categories, device, args, tokenizer, text_encoder,
        all_domain_prompts, all_class_prompts, all_latents_mean
    )

if __name__ == "__main__":
    main()