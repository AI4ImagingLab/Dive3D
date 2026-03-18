import torch
# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
import os
from pathlib import Path
from tqdm import tqdm

# --- 1. 用户配置区域 ---

# Stable Diffusion model id (HuggingFace) or local path.
# Default: SD 2.1 base.
MODEL_PATH = "stabilityai/stable-diffusion-2-1-base"

# 输入您想使用的6个文本提示 (prompts)
PROMPTS = [
    # "A baby bunny sitting on top of a stack of pancakes.",
    # "A rubbery cactus",
    # "A 3D model of an adorable cottage with a thatched roof",
    # "A lamp casting a warm glow",
    "A brick house",
    "A mug filled with steaming coffee"
]

# 每个prompt需要生成的图片数量
IMAGES_PER_PROMPT = 200

# 保存所有图片的根目录
OUTPUT_DIR = "generated_images"

# 生成参数
IMAGE_SIZE = 512
GUIDANCE_SCALE = 9.0
NUM_INFERENCE_STEPS = 50


# --- 配置结束 ---

def main():
    """
    主执行函数
    """
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        print("警告：未检测到CUDA，将在CPU上运行。速度会非常慢。")
        device = "cpu"
    else:
        device = "cuda"
        print(f"使用设备: {device}")

    # 确保输出根目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"所有图片将保存在根目录: {OUTPUT_DIR}")

    # # --- 加载模型 ---
    # print(f"正在从本地路径加载模型: {MODEL_PATH}")
    # try:
    #     # 使用 EulerDiscreteScheduler，这是一个速度和质量都不错的采样器
    #     scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    #     pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, scheduler=scheduler, torch_dtype=torch.float16)
    #     pipe = pipe.to(device)
    # except Exception as e:
    #     print(f"加载模型失败: {e}")
    #     print("请确认您的模型路径是否正确，以及是否已安装必要的依赖库。")
    #     return

    # --- 加载模型 (修改后的部分) ---
    print(f"正在从本地路径加载模型组件: {MODEL_PATH}")

    try:
        # 1. 单独加载每个组件，并明确指定 subfolder
        text_encoder = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet", torch_dtype=torch.float16)
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
        scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

        # 2. 将加载的组件手动组装成一个完整的 Pipeline
        pipe = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=None,  # 在研究和本地使用时，通常会禁用安全检查器
            feature_extractor=None,
            requires_safety_checker=False,
        )

        pipe = pipe.to(device)
        print("模型组件加载并成功组装 Pipeline！")

    except Exception as e:
        print(f"加载模型组件失败: {e}")
        print("请确认您的模型路径是否正确，以及 diffusers, transformers, accelerate 库是否已正确安装。")
        return

    # --- 开始生成图片 ---
    # 外层循环：遍历每一个prompt
    for prompt in tqdm(PROMPTS, desc="Total Prompts"):
        # 为每个prompt创建一个独立的子文件夹
        # 从prompt中提取一个适合做文件夹名的部分
        prompt_foldername = "".join(x for x in prompt if x.isalnum() or x in " _-").strip()[:50]
        prompt_output_path = os.path.join(OUTPUT_DIR, prompt_foldername)
        os.makedirs(prompt_output_path, exist_ok=True)

        print(f"\n正在处理 Prompt: '{prompt}'")
        print(f"图片将保存至: '{prompt_output_path}'")

        # 内层循环：为当前prompt生成指定数量的图片
        for i in tqdm(range(IMAGES_PER_PROMPT), desc=f"Generating for '{prompt_foldername}'"):
            # 为每一次生成设置不同的随机种子，以确保图片的多样性
            seed = i
            generator = torch.Generator(device=device).manual_seed(seed)

            # 使用pipeline生成图片
            # 使用 with torch.autocast("cuda") 可以开启自动混合精度，节省显存并加速
            with torch.autocast(device):
                image = pipe(
                    prompt,
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    generator=generator
                ).images[0]

            # 定义输出文件名并保存
            output_filename = os.path.join(prompt_output_path, f"{prompt_foldername}_seed{seed}.png")
            image.save(output_filename)

    print("\n所有任务处理完毕！")


if __name__ == "__main__":
    main()