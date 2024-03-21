import inspect
import warnings
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from accelerate import Accelerator
import spacy, clip 
# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from torchvision import transforms as tfms

import numpy as np
import cv2, copy
from PIL import Image
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import  LMSDiscreteScheduler
from diffusers.utils import (
    PIL_INTERPOLATION,
    BaseOutput,
    deprecate,

    logging,

)
# from diffusers.pipelines.pipeline_utils import DiffusionPipeline

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

vae_magic = 0.18215
def preprocess(image):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def prepare_mask_and_masked_image(image, mask):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.
    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.
    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.
    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).
    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)

    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def preprocess_mask(mask, batch_size: int = 1):
    if not isinstance(mask, torch.Tensor):
        # preprocess mask
        if isinstance(mask, PIL.Image.Image) or isinstance(mask, np.ndarray):
            mask = [mask]

        if isinstance(mask, list):
            if isinstance(mask[0], PIL.Image.Image):
                mask = [np.array(m.convert("L")).astype(np.float32) / 255.0 for m in mask]
            if isinstance(mask[0], np.ndarray):
                mask = np.stack(mask, axis=0) if mask[0].ndim < 3 else np.concatenate(mask, axis=0)
                mask = torch.from_numpy(mask)
            elif isinstance(mask[0], torch.Tensor):
                mask = torch.stack(mask, dim=0) if mask[0].ndim < 3 else torch.cat(mask, dim=0)

    # Batch and add channel dim for single mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)

    # Batch single mask or add channel dim
    if mask.ndim == 3:
        # Single batched mask, no channel dim or single mask not batched but channel dim
        if mask.shape[0] == 1:
            mask = mask.unsqueeze(0)

        # Batched masks no channel dim
        else:
            mask = mask.unsqueeze(1)

    # Check mask shape
    if batch_size > 1:
        if mask.shape[0] == 1:
            mask = torch.cat([mask] * batch_size)
        elif mask.shape[0] > 1 and mask.shape[0] != batch_size:
            raise ValueError(
                f"`mask_image` with batch size {mask.shape[0]} cannot be broadcasted to batch size {batch_size} "
                f"inferred by prompt inputs"
            )

    if mask.shape[1] != 1:
        raise ValueError(f"`mask_image` must have 1 channel, but has {mask.shape[1]} channels")

    # Check mask is in [0, 1]
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError("`mask_image` should be in [0, 1] range")

    # Binarize mask
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    return mask

class DTIILPipeline(DiffusionPipeline):
    r"""
    Pipeline for imagic image editing.
    See paper here: https://arxiv.org/pdf/2210.09276.pdf
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: CLIPImageProcessor,
        safety_checker=None

    ):
        super().__init__()
            
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )
        self.noisediff_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting',torch_dtype=torch.float,safety_checker=None)
        self.clip = None

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def train(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        generator: Optional[torch.Generator] = None,
        embedding_learning_rate: float = 0.001,
        diffusion_model_learning_rate: float = 2e-6,
        text_embedding_optimization_steps: int = 500,
        model_fine_tuning_optimization_steps: int = 1000,
        first_train = True,
        gamma = 8,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # if 
        self.image = image
        self.height = height
        self.width = width
        self.prompt = prompt 


        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16",
        )

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # Freeze vae and unet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        if accelerator.is_main_process:
            accelerator.init_trackers(
                "imagic",
                config={
                    "embedding_learning_rate": embedding_learning_rate,
                    "text_embedding_optimization_steps": text_embedding_optimization_steps,
                },
            )

        # get text embeddings for prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        )
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()
        text_embeddings_orig = text_embeddings.clone()

        # Initialize the optimizer
        optimizer = torch.optim.Adam(
            [text_embeddings],  # only optimize the embeddings
            lr=embedding_learning_rate,
        )

        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)

        latents_dtype = text_embeddings.dtype
        image = image.to(device=self.device, dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(image).latent_dist
        image_latents = init_latent_image_dist.sample(generator=generator)
        image_latents = 0.18215 * image_latents
        self.image_latents = image_latents

        progress_bar = tqdm(range(text_embedding_optimization_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        global_step = 0

        logger.info("First optimizing the text embedding to better reconstruct the init image")
        for _ in range(text_embedding_optimization_steps):
            with accelerator.accumulate(text_embeddings):
                # Sample noise that we'll add to the latents
                noise = torch.randn(image_latents.shape).to(image_latents.device)
                timesteps = torch.randint(1000, (1,), device=image_latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(image_latents, noise, timesteps)

                # Predict the noise residual
                # import pdb; pdb.set_trace()
                try:
                    noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
                except:
                    noise_pred = self.unet(noisy_latents, timesteps, {"text_embeds": text_embeddings}, added_cond_kwargs={"text_embeds": text_embeddings}).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                st = torch.abs(torch.cdist(text_embeddings, text_embeddings_orig).mean()-gamma)
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            # import pdb; pdb.set_trace()
            
            logs = {"loss": loss.detach().item()}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        text_embeddings.requires_grad_(False)

        # Now we fine tune the unet to better reconstruct the image
        self.unet.requires_grad_(True)
        self.unet.train()
        optimizer = torch.optim.Adam(
            self.unet.parameters(),  # only optimize unet
            lr=diffusion_model_learning_rate,
        )
        progress_bar = tqdm(range(model_fine_tuning_optimization_steps), disable=not accelerator.is_local_main_process)

        logger.info("Next fine tuning the entire model to better reconstruct the init image")
        for _ in range(model_fine_tuning_optimization_steps):
            with accelerator.accumulate(self.unet.parameters()):
                # Sample noise that we'll add to the latents
                noise = torch.randn(image_latents.shape).to(image_latents.device)
                timesteps = torch.randint(1000, (1,), device=image_latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(image_latents, noise, timesteps)

                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()
        # torch.cuda.empty_cache()
        # self.text_embeddings_orig = text_embeddings_orig
        # self.text_embeddings = text_embeddings
        if first_train:
            self.text_embeddings_consist = text_embeddings.clone()
            self.text_embeddings_inconsist = text_embeddings_orig.clone()
        else:
            self.text_embeddings_inconsist = text_embeddings.clone()

        return text_embeddings.clone(), text_embeddings_orig.clone()

    def image2latent(self,im):
        im = tfms.ToTensor()(im).unsqueeze(0)
        with torch.no_grad():
            latent = self.vae.encode(im.to(self.device)*2-1);
        latent = latent.latent_dist.sample() * vae_magic      
        return latent
        
    def latents2images(self,latents):
        latents = latents/vae_magic
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0,1)
        imgs = imgs.detach().cpu().permute(0,2,3,1).numpy()
        imgs = (imgs * 255).round().astype("uint8")
        imgs = [Image.fromarray(i) for i in imgs]
        return imgs
        
    def get_embedding_for_prompt(self,prompt):
        tokenizer = self.tokenizer 
        max_length = tokenizer.model_max_length
        tokens = tokenizer([prompt],padding="max_length",max_length=max_length,truncation=True,return_tensors="pt")
        with torch.no_grad():
            embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
        return embeddings

    def predict_noise(self,text_embeddings,im_latents,seed=torch.seed(),guidance_scale=7,strength=0.5,**kwargs):
        num_inference_steps = 10            # Number of denoising steps
        
        generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise

        uncond = self.get_embedding_for_prompt('')
        text_embeddings = torch.cat([uncond, text_embeddings])

        # Prep Scheduler
        scheduler = self.noisediff_scheduler
        scheduler.set_timesteps(num_inference_steps)

        offset = scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * 1 * 1, device=self.device)
        
        start_step = init_timestep
        noise = torch.randn_like(im_latents)
        latents = scheduler.add_noise(im_latents,noise,timesteps=timesteps)
        latents = latents.to(self.device).float()

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = scheduler.timesteps[t_start:].to(self.device)

        noisy_latent = latents.clone()

        noise_pred = None
        for i, tm in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, tm)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, tm, encoder_hidden_states=text_embeddings)["sample"]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            u = noise_pred_uncond
            g = guidance_scale
            t = noise_pred_text

            # perform guidance
            noise_pred = u + g * (t - u)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, tm, latents).prev_sample

        return self.latents2images(latents)[0],noise_pred

    def calc_diffedit_samples(self,encoded,emb1,emb2,n=10,**kwargs):
        diffs=[]
        # So we can reproduce mask generation we generate a list of n seeds
        torch.manual_seed(torch.seed() if 'seed' not in kwargs else kwargs['seed'])
        seeds = torch.randint(0,2**62,(10,)).tolist()
        for i in range(n):
            kwargs['seed'] = seeds[i] # Important to use same seed for the two noise samples
            if 'model' not in kwargs:
                _im1,n1 = self.predict_noise(emb1,encoded,**kwargs)
                _im2,n2 = self.predict_noise(emb2,encoded,**kwargs)
            else:
                n1 = predict_noise_diff(emb1,encoded,**kwargs)
                n2 = predict_noise_diff(emb2,encoded,**kwargs)

            # Aggregate the channel components by taking the euclidean distance.
            
            diffs.append((n1-n2)[0].pow(2).sum(dim=0).pow(0.5)[None])
        all_masks = torch.cat(diffs)
        return all_masks

    # Given an image latent and two prompts; generate a grayscale diff by sampling the noise predictions
    # between the prompts.
    def calc_diffedit_diff(self,im_latent,p1,p2,**kwargs):
        n = 10 if 'n' not in kwargs else kwargs['n']
        m = self.calc_diffedit_samples(im_latent,p1,p2,n=n)
        m = m.mean(axis=0) # average samples together
        m = (m-m.min())/(m.max()-m.min()) # rescale to interval [0,1]
        m = (m*255.).cpu().numpy().astype(np.uint8)
        m = Image.fromarray(m)
        return m

    # Try to improve the mask thru convolutions etc
    # assume m is a PIL object containing a grayscale 'diff'
    def process_diffedit_mask(self,m,threshold=0.35,**kwargs):
        m = np.array(m).astype(np.float32)
        m = cv2.GaussianBlur(m,(5,5),1)
        m = (m>(255.*threshold)).astype(np.float32)*255
        m = Image.fromarray(m.astype(np.uint8))
        return m

    @torch.no_grad() 
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def generate_mask(self, predict_noise_step=50, noise_pred_step=2, seed=12765083811071779585, threshold=0.2,**kwargs):
        # self.seed = seed
        self.unet.eval()
        # img = self.image2latent(img)
        # self.generator = torch.Generator(self._execution_device).manual_seed(self.seed)
        p1, p2 = self.text_embeddings_consist, self.text_embeddings_inconsist
        img = self.image_latents
        
        mask = self.calc_diffedit_diff(img, p1, p2, seed=seed, **kwargs)
        binarized_mask = self.process_diffedit_mask(mask, threshold=threshold).resize((self.height,self.width))
        blended_mask = self.get_blended_mask(binarized_mask)

        return mask, binarized_mask, blended_mask

    def get_blended_mask(self, mask_gray): # Both expected to be PIL images
        mask_rgb = mask_gray.convert('RGB')
        return Image.blend(self.image.resize((self.height,self.width)),mask_rgb,0.40)

    def save_img(self, result_dict, img_id, save_dir, im):
        import os
        if not os.path.isdir(save_dir+"/"+str(img_id)):
            os.mkdir(save_dir+"/"+str(img_id))
        im.save(save_dir+"/"+str(img_id)+"/"+str(img_id)+'_ori.png')
        result_dict['first_mask'].save(save_dir+"/"+str(img_id)+"/"+str(img_id)+'_mask_1st.png')
        result_dict['final_mask'].save(save_dir+"/"+str(img_id)+"/"+str(img_id)+'_mask.png')
        result_dict['first_edit_img'].save(save_dir+"/"+str(img_id)+"/"+str(img_id)+'_diff_1st.png')
        
    def __call__(
        self,
        prompt,
        im,
        first_step_only=False,
        diffusion_model_learning_rate=4e-6,
        embedding_learning_rate=0.001,
        text_embedding_optimization_steps=200,
        model_fine_tuning_optimization_steps=00,
        num_noise_pred = 8,
        threshold=0.15,
        threshold_final=0.3,
        seed=12765083811071779585,
        img_id = 0,
        save_dir = '',
        gamma=8,
    ):  
        height, width = im.size

        self.train(prompt=prompt,image=im, diffusion_model_learning_rate=diffusion_model_learning_rate,\
            text_embedding_optimization_steps=text_embedding_optimization_steps, \
            model_fine_tuning_optimization_steps=model_fine_tuning_optimization_steps, embedding_learning_rate=embedding_learning_rate, gamma=gamma)

        masklist_first = self.generate_mask(threshold=threshold, seed=seed, n=num_noise_pred)

        generator = torch.Generator(self.device).manual_seed(seed)
        self.inpaint.to(self.device)
        
        # im_result = self.inpaint(prompt=[prompt],image=im,mask_image=masklist_first[1],generator=generator, num_inference_steps=200).images[0]
        im_result = self.inpaint(image=im,prompt_embeds=self.text_embeddings_inconsist,mask_image=masklist_first[1],generator=generator, num_inference_steps=200).images[0]

        result_dict = {
            'first_mask': masklist_first[0],
            'first_bmask': masklist_first[1],
            'first_mask_img': masklist_first[2],
            'first_edit_img': im_result,
        }

        if not first_step_only:
            self.train(prompt=prompt,image=im_result, diffusion_model_learning_rate=diffusion_model_learning_rate, \
                text_embedding_optimization_steps=text_embedding_optimization_steps, first_train=False, \
                model_fine_tuning_optimization_steps=model_fine_tuning_optimization_steps, embedding_learning_rate=embedding_learning_rate,gamma=gamma)

            masklist_final = self.generate_mask(threshold=threshold_final, seed=seed, n=num_noise_pred)
            final_img = None

            result_dict = {
                'first_mask': masklist_first[0],
                'final_mask': masklist_final[0],
                'first_bmask': masklist_first[1],
                'final_bmask': masklist_final[1],
                'first_mask_img': masklist_first[2],
                'final_mask_img': masklist_final[2],
                'first_edit_img': im_result,
                'final_edit_img': final_img,
            }
        self.result_dict = result_dict
        if save_dir is not None:
            self.save_img(result_dict, img_id, save_dir, im)
        return result_dict
    
    def get_max_mask(self, mask, top_n=1):
        max_bbox = [9999,9999,0,0]
        mask = cv2.resize(mask, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, 2)
        areas = [cv2.contourArea(c) for c in contours]
        max_area_idxs = np.array(areas).argsort()[-top_n:]
        # max_contour = contours[max_area_idx] * 8
        max_mask = np.zeros_like(mask)
        for max_area_idx in max_area_idxs:
            max_contour = contours[max_area_idx] 
            max_mask = cv2.fillPoly(max_mask, pts =[max_contour], color=(255,255,255))
            max_bbox = [min(max_contour[:,:,0].min(), max_bbox[0]), min(max_contour[:,:,1].min(), max_bbox[1]),
                    max(max_contour[:,:,0].max(), max_bbox[2]), max(max_contour[:,:,1].max(), max_bbox[3])]
        return max_mask, max_bbox

    def get_words(self,cap):
        nlp = spacy.load('en_core_web_sm')
        labellist = ['NOUN','ADJ','PROPN','PUNCT','PART']
        doc = nlp(cap)
        captionWordList = []
        mystring = ''
        for token in doc:
        # Print the text and the predicted part-of-speech tag
        # print(token.text, token.pos_, token.dep_,token.is_stop)
            if not token.is_stop and token.pos_ in labellist:
                oristring = token.text + ' '
                mystring += oristring
            else:
                if mystring !='':
                    mystring = mystring.strip()
                    captionWordList.append('A photo of '+mystring)
                    mystring = ''

        if mystring != '': captionWordList.append('A photo of '+mystring.strip())
        return captionWordList

    @torch.no_grad()
    def get_inconsist_word(self):
        im = np.array(self.result_dict['first_edit_img'])
        prompt = self.prompt
        words = self.get_words(prompt)
        mask = self.result_dict['final_bmask']
        max_mask, max_bbox = self.get_max_mask(np.array(mask))
        x1,y1,x2,y2 = max_bbox
        if self.clip is None:
            self.clip , self.preprocess = clip.load("ViT-B/32", device=self.device)
            
        cropped_im = im[y1:y2, x1:x2, :]
        cropped_im = self.preprocess(PIL.Image.fromarray(cropped_im)).unsqueeze(0).to(self.device)

        text = clip.tokenize(words, truncate=True).to(self.device)
        cost_matrix = self.clip(cropped_im, text)[0]
        max_ind = cost_matrix.argmax()

        return words[max_ind].replace('A photo of ', '')