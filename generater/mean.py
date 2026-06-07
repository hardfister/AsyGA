import os
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict

# 1. Basic settings
model_path = r"E:\models\stable-diffusion-v1-5"  # Local model path
base_data_dir = r"data/ca"                    # Dataset root directory
output_dir = r"outputs/ca"        # Unified save to new folder for heterogeneous testing

device = "cuda"
os.makedirs(output_dir, exist_ok=True)

# Core modification 1: Define the spectrum domain for each of the 4 clients
domains = ["Blue", "Green", "Red", "NIR"]

# 2. Load VAE
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", local_files_only=True).to(device, dtype=torch.float16)
vae.eval()

# 3. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 4. Start extracting features for each Client's corresponding spectrum
for idx, domain in enumerate(domains):
    print(f"\n=============================================")
    print(f"Extracting latents for Client {idx} (Domain: {domain})...")

    # Core modification 2: Create a new empty dictionary each iteration to prevent data from leaking into other Clients
    latents_mean = defaultdict(list)
    latents_std = defaultdict(list)

    # Construct the physical path for the current Client, e.g., data/polyu/Blue
    domain_dir = os.path.join(base_data_dir, domain)
    if not os.path.exists(domain_dir):
        print(f"Warning: Cannot find folder {domain_dir}, please check the data directory structure!")
        continue

    categories = sorted(os.listdir(domain_dir))

    for c in tqdm(categories, desc=f"Client {idx}-{domain}"):
        class_dir = os.path.join(domain_dir, c)
        if not os.path.isdir(class_dir): continue

        for img_name in os.listdir(class_dir):
            if not img_name.endswith(('.jpg', '.png', '.bmp')): continue
            img_path = os.path.join(class_dir, img_name)

            # Read and transform image
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device, dtype=torch.float16)

            # Pass through VAE to get latent
            with torch.no_grad():
                latent_dist = vae.encode(img_tensor).latent_dist
                latent = latent_dist.sample() * vae.config.scaling_factor

            # Save to dictionary (must move to cpu to avoid memory overflow)
            latents_mean[c].append(latent.cpu())
            latents_std[c].append(torch.zeros_like(latent).cpu())

    # 5. Save current Client's latent feature files
    torch.save(latents_mean, os.path.join(output_dir, f"mean_{idx}.pt"))
    torch.save(latents_std, os.path.join(output_dir, f"std_{idx}.pt"))
    print(f"Client {idx} ({domain}) latent features saved!")

print(f"\nAll heterogeneous feature extraction complete! All stored in {output_dir}")