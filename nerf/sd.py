from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, EulerDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd 


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad) 
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype) # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

def add_noise_and_return_std(scheduler, original_samples, noise, timesteps):
    ## modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py#L477
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
    # for the subsequent add_noise calls
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=original_samples.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples, sqrt_one_minus_alpha_prod


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None, opt=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.opt = opt

        print(f'[INFO] loading stable diffusion...')
        print('stable diffusion:', sd_version)
        
        if hf_key is not None:
            print(f'[INFO] using Hugging Face model id/path: {hf_key}')
            model_key = hf_key
        else:
            if self.sd_version == '2.1':
                model_key = "stabilityai/stable-diffusion-2-1-base"
            elif self.sd_version == '2.0':
                model_key = "stabilityai/stable-diffusion-2-base"
            elif self.sd_version == '1.5':
                model_key = "runwayml/stable-diffusion-v1-5"
            else:
                raise ValueError(
                    f"Stable-diffusion version {self.sd_version} not supported. "
                    f"Use --sd_version in [1.5, 2.0, 2.1] or pass --hf_key with a local path / HF model id."
                )

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        
        
        
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
                
        # for p in self.unet.parameters():
        #     p.requires_grad_(False)

        # if is_xformers_available():
        #     self.unet.enable_xformers_memory_efficient_attention()
        
        print("NOT using v-pred")
        opt.v_pred = False
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        alphas_cumprod = self.scheduler.alphas_cumprod
        # self.sigma_min = 0.029167533 
        # self.sigma_max = 14.614647
        self.sigmas = (((1 - alphas_cumprod) / alphas_cumprod) ** 0.5).to(self.device)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * opt.t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def t_lognormal(self, n, P_mean, P_std):
        noise = (torch.randn([n,], device=self.device) * P_std + P_mean).exp()
        # find the closest timestep
        index = torch.cdist(noise.view(1, -1, 1), self.sigmas.view(1, -1, 1)).argmin(2)
        
        return index.view(-1)

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, q_unet = None, pose = None, shading = None,
                   grad_clip = None, as_latent = False, t5 = False, global_step=None, epoch=None):
        
        # interp to 512x512 to be fed into vae.
        assert torch.isnan(pred_rgb).sum() == 0, print(pred_rgb)
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False).contiguous()
        elif self.opt.latent == True:
            latents = pred_rgb
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False).contiguous()
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)        
        if self.opt.t1_max < 0: # using default value
            self.opt.t1_max = self.max_step
        if self.opt.t_dist == 'all_log_normal':
            P_mean = self.opt.P_mean
            P_std = self.opt.P_std
            t = self.t_lognormal(1, P_mean, P_std)
        else:
            if t5: # Anneal time schedule
                if self.opt.t_dist == 'log_normal':
                    P_mean = self.opt.P_mean
                    P_std = self.opt.P_std
                    t = self.t_lognormal(1, P_mean, P_std) # (torch.randn([1], device=self.device) * P_std + P_mean).exp().round().clamp(self.min_step, self.max_step + 1).long()
                elif self.opt.t_dist == 'uniform':
                    t2_max = self.opt.t2_max
                    if t2_max < 0: # using default value
                        t2_max = 500
                    t = torch.randint(self.min_step, t2_max + 1, [1], dtype=torch.long, device=self.device)
            else:
                t = torch.randint(self.min_step, self.opt.t1_max + 1, [1], dtype=torch.long, device=self.device)
        # t = torch.randint(self.min_step, 500 + 1, [latents.shape[0]], dtype=torch.long, device=self.device) # (torch.randn([1], device=self.device) * P_std + P_mean).exp().round().clamp(self.min_step, self.max_step + 1).long()
        
        # predict the noise residual with unet, NO grad!
        # with torch.no_grad(): ## requires gradient!

        # self.unet.requires_grad_(False)
        # for p in self.unet.parameters():
        #     p.requires_grad_(False)
        # self.unet.eval()

        # add noise
        noise = torch.randn_like(latents)

        ## modified to also return noise variance
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # latents_noisy, xt_x0_std = add_noise_and_return_std(self.scheduler, latents, noise, t)

        xt_x0_std = (1 - self.scheduler.alphas_cumprod.to(self.device)[t]) ** 0.5
        xt_x0_std = xt_x0_std.flatten()
        while len(xt_x0_std.shape) < len(latents_noisy.shape):
            xt_x0_std = xt_x0_std.unsqueeze(-1)

        # pred noise
        # latent_model_input = torch.cat([latents_noisy] * 2)

        temb_uncond, temb_cond = text_embeddings.chunk(2)
        noise_pred_uncond = self.unet(latents_noisy, t, encoder_hidden_states=temb_uncond).sample
        noise_pred_text = self.unet(latents_noisy, t, encoder_hidden_states=temb_cond).sample

        # noise_pred = self.unet(latent_model_input, torch.cat([t]*2), encoder_hidden_states=text_embeddings).sample
        # perform guidance (high scale from paper!)
        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        #noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # print("original sds,", self.opt.sds)
        self.opt.sds = False
        if self.opt.sds is False:
            if q_unet is not None:
                if pose is not None:
                    noise_pred_q = q_unet(latents_noisy, t, c = pose, shading = shading).sample
                else:
                    raise NotImplementedError()

                if self.opt.v_pred:
                    sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(latents_noisy.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                    sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(self.device)[t]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(latents_noisy.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                    noise_pred_q = sqrt_alpha_prod * noise_pred_q + sqrt_one_minus_alpha_prod * latents_noisy

        # w(t), sigma_t^2
        factor1 = self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5
        factor2 = torch.sqrt(1 - factor1 ** 2)
        factor = factor2 / (factor1 + 1e-4)
        factor = factor.view(-1,1,1,1)
        use_factor = True

        if self.opt.wgt_type == 'orig':
            w = (1 - self.alphas[t]).view(-1,1,1,1)
        elif self.opt.wgt_type == 'nowgt':
            w = 1.0

        if use_factor:
            w = w * factor

        #--------------------------------sim-3d afterchange--------------------------------
        # FIXME: LORA updating, weimin, stop it when perform SDS
        if epoch <= 140 or (epoch > 140 and global_step % 5 < -1):
            print('sim-3d')
            xt_x0_std = -((1 - self.scheduler.alphas_cumprod.to(self.device)[t]) ** 0.5) / (self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5)
            xt_x0_std = xt_x0_std.flatten()
            while len(xt_x0_std.shape) < len(latents_noisy.shape):
                xt_x0_std = xt_x0_std.unsqueeze(-1)

            noise_diff = (noise_pred - noise_pred_q) * xt_x0_std

            # with torch.no_grad():
            #     cfg_vec = (guidance_scale - 1)*(noise_pred_text - noise_pred_uncond) #guidance_scale可以换成4.5/1.5

            sim_loss = noise_diff / noise_diff.square().sum([1,2,3], keepdims=True).sqrt()
            sim_loss = sim_loss * (noise_pred_q - noise) * xt_x0_std
            loss = sim_loss
            loss = loss.sum([1, 2, 3]).mean()
            pseudo_loss = loss.detach().clone()

        # --------------------------------VSD--------------------------------
        else:
            print('vsd')
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            grad = w * (noise_pred - noise_pred_q)
            grad = torch.nan_to_num(grad)
            target = (latents - grad).detach()
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]
            pseudo_loss = loss.detach().clone()

        # --------------------------------SDS--------------------------------
        # else:
        #     print('sds')
        #     w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        #     grad = w * (noise_pred - noise)
        #     grad = torch.nan_to_num(grad)
        #
        #     # # clip grad for stable training?
        #     # if self.grad_clip_val is not None:
        #     #     grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        #
        #     target = (latents - grad).detach()
        #     # loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]
        #     loss_sds = 0.5 * F.mse_loss(latents, target, reduction="mean")
        #     loss = loss_sds
        #     pseudo_loss = loss.detach().clone()

        return loss, pseudo_loss, latents

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
