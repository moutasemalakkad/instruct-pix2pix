from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
#from stable_diffusion.sgm.modules.diffusionmodules.discretizer import EDMDiscretization #discretizer for sgm
#from stable_diffusion.sgm.models.diffusion import DiffusionEngine #
from stable_diffusion.sgm.modules.encoders.modules import GeneralConditioner
from stable_diffusion.sgm.modules.diffusionmodules.denoiser import DiscreteDenoiser #denoiser for sgm
from stable_diffusion.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

# for debugging
sys.path.append("~/Desktop/stable_diffusion")
sys.path.append('~/Desktop/stable_diffusion/sgm')
sys.path.append('~/Desktop/stable_diffusion/ldm')
# for regular run
sys.path.append("./stable_diffusion")
sys.path.append('./stable_diffusion/sgm')
sys.path.append('./stable_diffusion/ldm')


from stable_diffusion.ldm.util import instantiate_from_config

# import os

# relative_path = './stable_diffusion/ldm'
# absolute_path = os.path.abspath(relative_path)
# sys.path.append(absolute_path)
# print(sys.path)

import sys

print("Python Environment:")
print(sys.executable)


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        # z: torch.Size([1, 4, 32, 32])
        # cfg_z: torch.Size([3, 4, 32, 32])
        # cfg_sigma: torch.Size([3])
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            # torch.Size([3, 77, 768])
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            # torch.Size([3, 4, 32, 32])
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        # out_uncond = out_img_cond torch.Size([1, 4, 32, 32])

        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


# def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
#     print(f"Loading model from {ckpt}")
#     pl_sd = torch.load(ckpt, map_location="cpu")
#     if "global_step" in pl_sd:
#         print(f"Global Step: {pl_sd['global_step']}")
#     sd = pl_sd["state_dict"]
#     if vae_ckpt is not None:
#         print(f"Loading VAE from {vae_ckpt}")
#         vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
#         sd = {
#             k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
#             for k, v in sd.items()
#         }
#     model = instantiate_from_config(config.model)
#     m, u = model.load_state_dict(sd, strict=False)
#     if len(m) > 0 and verbose:
#         print("missing keys:")
#         print(m)
#     if len(u) > 0 and verbose:
#         print("unexpected keys:")
#         print(u)
#     return model

import torch
from safetensors.torch import load_file as load_safetensors

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    # added be me to deal with ckpt and safetensors checkpoints
    if ckpt.endswith("ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError("Unsupported checkpoint file format")

    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :] if k.startswith("first_stage_model.") else v]
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval()#.cuda()

    #if we are using SDXL, we need to use the DiscreteDenoiser
    if args.config == '/Users/moutasemhome/Desktop/instruct-pix2pix/configs/sd_xl_base.yaml':
        model_wrap = DiscreteDenoiser(config["model"]["params"]["denoiser_config"]["params"]["weighting_config"], config["model"]["params"]["denoiser_config"]["params"]["scaling_config"], 1000, config["model"]["params"]["denoiser_config"]["params"]["discretization_config"], do_append_zero=False, quantize_c_noise=True, flip=True)
    # else we use the CompVisDenoiser
    else:
        model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    # representation of empty string
    #null_token = model.get_learned_conditioning([""]) # that is for ldm

    #null_token = config['model']['params']['conditioner_config']['target']

    emb_models_config = config['model']['params']['conditioner_config']['params']['emb_models']
    conditioner = GeneralConditioner(emb_models=emb_models_config)
    null_token = conditioner({'txt':""})


#get_unconditional_conditioning

    config["model"].conditioner.get_unconditional_conditioning()


    seed = random.randint(0, 100000) if args.seed is None else args.seed
    input_image = Image.open(args.input).convert("RGB")
    width, height = input_image.size
    factor = args.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if args.edit == "":
        input_image.save(args.output)
        return

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        # edit command -> tensor torch.Size([1, 77, 768]) || Attention
        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
        # [256, 256, 3] -> [1, 3, 256, 256]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        #[1, 3, 256, 256]
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        # torch.Size([1, 4, 32, 32])
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        # torch.Size([1, 77, 768])
        uncond["c_crossattn"] = [null_token]
        # torch.Size([1, 4, 32, 32])
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])] # remove next zero

        # we need to create an instance of EDMDiscretization  for sdxl# this causes RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
        # discretization_instance = EDMDiscretization()
        # sigmas = discretization_instance.get_sigmas(args.steps)

        #torch.Size([101])
        sigmas = model_wrap.get_sigmas(args.steps)


        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }
        torch.manual_seed(seed)
        # torch.Size([1, 4, 32, 32])
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0] # remove next zero
        # sigmas: torch.Size([101])
        # z torch.Size([1, 4, 32, 32])
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        # torch.Size([256, 256, 3]
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    edited_image.save(args.output)


if __name__ == "__main__":
    main()


