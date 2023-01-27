# assumes you have run
#!pip install -qqq diffusers==0.11.1 transformers ftfy gradio

# imports
import argparse
import os

import torch
from diffusers import DiffusionPipeline, LMSDiscreteScheduler, PNDMScheduler
from PIL import Image
from tqdm import tqdm
from transformers import CLIPFeatureExtractor, CLIPModel


def image_grid(imgs, rows, cols, grid=None):

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h)) if grid is None else grid
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# Load the pipeline
def load_pipeline(args):
    model_id = "CompVis/stable-diffusion-v1-4"  # @param {type: "string"}
    clip_model_id = args.clip_model_id  # @param ["laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    # "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    # "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
    # "openai/clip-vit-base-patch32",
    # "openai/clip-vit-base-patch16",
    # "openai/clip-vit-large-patch14"]
    scheduler = "plms"  # @param ['plms', 'lms']

    if scheduler == "lms":
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
    else:
        scheduler = PNDMScheduler.from_config(model_id, subfolder="scheduler")

    feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float16)

    lidar_feature = torch.load(args.feature_path)[0:1].half().to("cuda")
    clip_model.get_text_features = lambda *args, **kwargs: lidar_feature

    guided_pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        custom_pipeline="/mimer/NOBACKUP/groups/clippin/users/georghe/lidar-clippin/scripts/clip_guided_stable_diffusion_pipeline/",
        clip_model=clip_model,
        feature_extractor=feature_extractor,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    )
    guided_pipeline.enable_attention_slicing()
    guided_pipeline = guided_pipeline.to("cuda")
    return guided_pipeline


def load_clip_features(args):
    features = torch.load(args.feature_path).half().to("cuda")
    return features


def gen_images(args, guided_pipeline, features, prompt=["a photorealistic image"]):
    batch_size = features.shape[0]
    num_cutouts = args.num_cutouts
    guided_pipeline.clip_model.get_text_features = (
        lambda *args, **kwargs: features.repeat_interleave(num_cutouts, dim=0)
    )

    # Generate
    if len(prompt) == 1:
        prompt = prompt * batch_size
    clip_prompt = ""
    num_samples = 1
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    clip_guidance_scale = args.clip_guidance_scale

    use_cutouts = args.use_cutouts
    unfreeze_unet = args.unfreeze_unet
    unfreeze_vae = args.unfreeze_vae
    seed = args.seed

    if unfreeze_unet:
        guided_pipeline.unfreeze_unet()
    else:
        guided_pipeline.freeze_unet()

    if unfreeze_vae:
        guided_pipeline.unfreeze_vae()
    else:
        guided_pipeline.freeze_vae()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    images = []
    for _ in range(num_samples):
        imgs = guided_pipeline(
            prompt,
            clip_prompt=clip_prompt if clip_prompt.strip() != "" else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            clip_guidance_scale=clip_guidance_scale,
            num_cutouts=num_cutouts,
            use_cutouts=use_cutouts,
            generator=generator,
            height=512,
            width=512,
        ).images
        images.extend(imgs)

    return images


def find_filename(args, idx):
    # name files as args.output_dir/idx_i.png where i is the i-th image generated for that idx
    # try to find the first i that doesn't exist
    i = 0
    while True:
        filename = os.path.join(args.output_dir, f"{idx}_{i}.png")
        if not os.path.exists(filename):
            return filename
        i += 1


def save_images(args, images, idx):
    # images is a list of PIL images

    # mkdir
    os.makedirs(args.output_dir, exist_ok=True)

    filenames = [find_filename(args, i) for i in idx]

    for i, img in enumerate(images):
        img.save(filenames[i])


def main(args):
    # mkdir
    os.makedirs(args.output_dir, exist_ok=True)
    # Get indexes
    # with open(args.indexes_path, "r") as f:
    #    indexes = [int(x) for x in f.readlines()]
    generated_imgs = os.listdir("/mimer/NOBACKUP/groups/clippin/gen_images/once_val_lidar_selected")
    indexes = [int(img.split(".")[0].split("_")[0]) for img in generated_imgs]
    # Discard indexes
    indexes = indexes[args.start_idx :: args.every_n]
    # Check if image already exists
    if not args.regenerate:  # if we don't want to regenerate, remove indexes that already exist
        existing_files = os.listdir(args.output_dir)
        indexes = [idx for idx in indexes if f"{idx}_0.png" not in existing_files]

    # Load features
    features = load_clip_features(args)
    # Load pipeline
    guided_pipeline = load_pipeline(args)

    if len(args.caption_path) > 0:
        captions = torch.load(args.caption_path)
    else:
        captions = None

    # batch index and loop with tqdm
    for i in tqdm(range(0, len(indexes), args.batch_size), desc="Generating images"):
        idx = indexes[i : i + args.batch_size]
        # Get features
        batch_features = features[idx]
        # Get captions
        if captions is not None:
            # captions is a dict of strings
            batch_captions = [captions[i] for i in idx]
        else:
            batch_captions = ["a photorealistic image" for _ in idx]

        # Generate images
        images = gen_images(args, guided_pipeline, batch_features, prompt=batch_captions)
        # Save images
        save_images(args, images, idx)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_path",
        type=str,
        default="/mimer/NOBACKUP/groups/clippin/features/once_vit-l-14_val_lidar.pt",
    )
    parser.add_argument("--clip_model_id", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--output_dir", type=str, default="/mimer/NOBACKUP/groups/clippin/gen_images/once_val"
    )
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--num_inference_steps", type=int, default=250)
    parser.add_argument("--guidance_scale", type=float, default=0.05)
    parser.add_argument("--clip_guidance_scale", type=float, default=1200)
    parser.add_argument("--num_cutouts", type=int, default=12)
    parser.add_argument("--use_cutouts", type=bool, default=True)
    parser.add_argument("--unfreeze_unet", type=bool, default=False)
    parser.add_argument("--unfreeze_vae", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--indexes_path",
        type=str,
        default="/mimer/NOBACKUP/groups/clippin/gen_images/once_val_ids.txt",
    )
    parser.add_argument("--every_n", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--regenerate", type=bool, default=False)
    parser.add_argument("--caption-path", type=str, default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
