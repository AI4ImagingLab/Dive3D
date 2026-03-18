import torch
import os
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import itertools


def calculate_average_pairwise_clip_distance(
    image_folder: str,
    model_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    batch_size: int = 16,
):
    """
    Calculates the average pairwise CLIP distance for all PNG images in a folder.

    Args:
        image_folder (str): Path to the folder containing PNG images.
        model_name (str): The CLIP model to use from Hugging Face.
        batch_size (int): The batch size for processing images to manage memory.

    Returns:
        float: The average pairwise CLIP distance.
    """
    # 1. 设置设备 (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载预训练的CLIP模型和处理器
    try:
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and the 'transformers' library is installed.")
        return None

    # 3. 找到所有PNG图片
    image_paths = list(Path(image_folder).rglob("*.png"))
    if not image_paths:
        print(f"Error: No PNG images found in folder: {image_folder}")
        return None

    print(f"Found {len(image_paths)} PNG images. Processing in batches of {batch_size}...")

    # 4. 批量提取所有图片的特征嵌入
    all_features = []
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP Features"):
            batch_paths = image_paths[i:i + batch_size]

            # 读取并预处理图片
            try:
                images = [Image.open(path).convert("RGB") for path in batch_paths]
                inputs = processor(text=None, images=images, return_tensors="pt", padding=True).to(device)
            except Exception as e:
                print(f"\nError processing a batch of images: {e}")
                continue

            # 获取特征嵌入
            image_features = model.get_image_features(pixel_values=inputs['pixel_values'])

            # L2 归一化，这是计算余弦相似度的标准步骤
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            all_features.append(image_features)

    if not all_features:
        print("Could not extract any features from the images.")
        return None

    # 将所有批次的特征合并成一个张量
    all_features_tensor = torch.cat(all_features, dim=0)
    num_images = all_features_tensor.shape[0]
    print(f"Successfully extracted {num_images} feature embeddings.")

    # 5. 高效计算平均成对余弦距离
    if num_images < 2:
        print("Need at least two images to calculate pairwise distance.")
        return 0.0

    # 通过矩阵乘法计算所有成对的余弦相似度 (N, D) @ (D, N) -> (N, N)
    # 因为特征已经归一化，所以矩阵乘法的结果就是余弦相似度矩阵
    similarity_matrix = torch.matmul(all_features_tensor, all_features_tensor.T)

    # 将相似度转换为距离 (distance = 1 - similarity)
    distance_matrix = 1 - similarity_matrix

    # 为了计算平均值，我们需要对所有唯一的对求和
    # 我们只取上三角部分（不包括对角线），以避免重复计算 (i,j) 和 (j,i)，以及自身与自身的比较 (i,i)
    # torch.triu(distance_matrix, diagonal=1) 会将下三角和对角线置零
    total_distance = torch.sum(torch.triu(distance_matrix, diagonal=1))

    # 计算唯一对的数量: N * (N - 1) / 2
    num_unique_pairs = num_images * (num_images - 1) / 2

    average_distance = total_distance / num_unique_pairs

    return average_distance.item()


if __name__ == '__main__':
    # --- 用户需要修改的部分 ---
    # 请将这里的路径替换为您存放PNG图片的文件夹路径
    TARGET_FOLDER = "./output_frames_all4brickbunny_kl"
    # --- 修改结束 ---

    if not os.path.isdir(TARGET_FOLDER):
        print(f"Error: The specified folder does not exist: {TARGET_FOLDER}")
        print("Please update the 'TARGET_FOLDER' variable in the script.")
    else:
        avg_clip_d = calculate_average_pairwise_clip_distance(TARGET_FOLDER)
        if avg_clip_d is not None:
            print("\n-------------------------------------------")
            print(f"Average Pairwise CLIP-Distance (CLIP-D): {avg_clip_d:.4f}")
            print("-------------------------------------------")
            print("Note: A higher score indicates greater semantic diversity among the images.")