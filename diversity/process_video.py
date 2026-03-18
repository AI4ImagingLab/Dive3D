import cv2
import os
from pathlib import Path  # 引入pathlib来方便地处理文件名


def process_multiple_videos(video_paths,
                            output_folder,
                            interval_sec=0.5,
                            initial_crop_size=700,
                            final_size=512):
    """
    批量处理多个视频文件，进行中心裁剪、缩放并保存到同一个文件夹。

    参数:
    video_paths (list): 包含多个输入MP4文件路径的列表。
    output_folder (str): 保存PNG图片的文件夹路径。
    interval_sec (float): 提取帧的时间间隔（秒）。
    initial_crop_size (int): 初始中心裁剪的尺寸。
    final_size (int): 最终保存图片的尺寸。
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    print(f"所有图片将保存到: '{output_folder}'")

    # 遍历视频文件列表中的每一个视频
    i = 0
    for video_path in video_paths:
        # 检查输入文件是否存在
        if not os.path.exists(video_path):
            print(f"\n警告：视频文件不存在，跳过 '{video_path}'")
            continue

        print(f"\n--- 开始处理视频: {video_path} ---")

        # 使用Pathlib获取不带后缀的文件名，用于命名输出图片
        video_basename = Path(video_path).stem

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 '{video_path}'")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("警告：无法读取视频的FPS，将默认使用30 FPS。")
            fps = 30

        frames_to_skip = int(round(fps * interval_sec))
        print(f"视频FPS: {fps:.2f}，每隔 {frames_to_skip} 帧截取一张图片。")

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frames_to_skip == 0:
                h, w, _ = frame.shape
                start_x = (w - initial_crop_size) // 2
                start_y = (h - initial_crop_size) // 2

                if start_x < 0 or start_y < 0:
                    print(f"警告：视频分辨率({w}x{h}) 小于初始裁剪尺寸({initial_crop_size}x{initial_crop_size})，跳过第 {frame_count} 帧。")
                    frame_count += 1
                    continue

                cropped_frame = frame[start_y: start_y + initial_crop_size, start_x: start_x + initial_crop_size]
                resized_frame = cv2.resize(cropped_frame, (final_size, final_size), interpolation=cv2.INTER_AREA)

                # 构建包含原始视频名和时间点的唯一文件名
                current_time_sec = frame_count / fps
                output_filename = f"{video_basename}{i}_frame_at_{current_time_sec:.2f}s.png"
                output_path = os.path.join(output_folder, output_filename)

                cv2.imwrite(output_path, resized_frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        i += 1
        print(f"--- 视频 '{video_path}' 处理完成，共保存了 {saved_count} 张图片。 ---")

    print("\n所有视频处理完毕！")


if __name__ == '__main__':
    # --- 用户需要修改的部分 ---

    # 1. 在这个列表中放入所有您想处理的MP4文件路径
    # input_video_files = [
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250717/sim140+tunevsd1+p100+cfg7.5+cottage+n4+i30000+orient0.6-100/2025-07-17-A-3D-model-of-an-adorable-cottage-with-a-thatched-roof-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_00_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250717/sim140+tunevsd1+p100+cfg7.5+cottage+n4+i30000+orient0.6-100/2025-07-17-A-3D-model-of-an-adorable-cottage-with-a-thatched-roof-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_01_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250717/sim140+tunevsd1+p100+cfg7.5+cottage+n4+i30000+orient0.6-100/2025-07-17-A-3D-model-of-an-adorable-cottage-with-a-thatched-roof-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_02_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250717/sim140+tunevsd1+p100+cfg7.5+cottage+n4+i30000+orient0.6-100/2025-07-17-A-3D-model-of-an-adorable-cottage-with-a-thatched-roof-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_03_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+cactus+n4+i30000+orient0.6-100/2025-05-24-A-rubbery-cactus-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0300_00_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+cactus+n4+i30000+orient0.6-100/2025-05-24-A-rubbery-cactus-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0300_01_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+cactus+n4+i30000+orient0.6-100/2025-05-24-A-rubbery-cactus-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0300_02_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+cactus+n4+i30000+orient0.6-100/2025-05-24-A-rubbery-cactus-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0300_03_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+lamp+n4+i30000+orient0.6-100/2025-05-24-A-lamp-casting-a-warm-glow-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0300_00_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+lamp+n4+i30000+orient0.6-100/2025-05-24-A-lamp-casting-a-warm-glow-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0300_01_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+lamp+n4+i30000+orient0.6-100/2025-05-24-A-lamp-casting-a-warm-glow-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0300_02_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+lamp+n4+i30000+orient0.6-100/2025-05-24-A-lamp-casting-a-warm-glow-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0300_03_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+mug+n4+i30000+orient0.6-100/2025-05-24-A-mug-filled-with-steaming-coffee-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_00_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+mug+n4+i30000+orient0.6-100/2025-05-24-A-mug-filled-with-steaming-coffee-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_01_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+mug+n4+i30000+orient0.6-100/2025-05-24-A-mug-filled-with-steaming-coffee-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_02_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+mug+n4+i30000+orient0.6-100/2025-05-24-A-mug-filled-with-steaming-coffee-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_03_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+brick+n4+i30000+orient0.6-100/2025-05-24-A-brick-house-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_00_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+brick+n4+i30000+orient0.6-100/2025-05-24-A-brick-house-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_01_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+brick+n4+i30000+orient0.6-100/2025-05-24-A-brick-house-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_02_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+brick+n4+i30000+orient0.6-100/2025-05-24-A-brick-house-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_03_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+vase+n4+i30000+orient0.6-100/2025-05-24-A-baby-bunny-sitting-on-top-of-a-stack-of-pancakes.-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_00_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+vase+n4+i30000+orient0.6-100/2025-05-24-A-baby-bunny-sitting-on-top-of-a-stack-of-pancakes.-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_01_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+vase+n4+i30000+orient0.6-100/2025-05-24-A-baby-bunny-sitting-on-top-of-a-stack-of-pancakes.-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_02_albedo_rgb.mp4",
    #     "/gpfs/share/home/2206192113/cvpr_code/lyb/sim_3d_reward/exp-weimin-250524/sim140+tunevsd1+p100+cfg7.5+vase+n4+i30000+orient0.6-100/2025-05-24-A-baby-bunny-sitting-on-top-of-a-stack-of-pancakes.-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-12500-finetune-dth-0.3-tet-256-lnorm-100.0/results/df_ep0400_03_albedo_rgb.mp4",
    # ]

    input_video_files = [
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/brick_p0_s250_stage1/2025-05-24-A-brick-house-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_00_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/brick_p0_s250_stage1/2025-05-24-A-brick-house-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_01_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/brick_p0_s250_stage1/2025-05-24-A-brick-house-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_02_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/brick_p0_s250_stage1/2025-05-24-A-brick-house-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_03_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/cactus_p0_s250_stage1/2025-05-24-A-rubbery-cactus-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_00_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/cactus_p0_s250_stage1/2025-05-24-A-rubbery-cactus-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_01_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/cactus_p0_s250_stage1/2025-05-24-A-rubbery-cactus-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_02_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/cactus_p0_s250_stage1/2025-05-24-A-rubbery-cactus-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_03_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/cottage_p0_s250_stage1/2025-07-18-A-3D-model-of-an-adorable-cottage-with-a-thatched-roof-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_00_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/cottage_p0_s250_stage1/2025-07-18-A-3D-model-of-an-adorable-cottage-with-a-thatched-roof-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_01_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/cottage_p0_s250_stage1/2025-07-18-A-3D-model-of-an-adorable-cottage-with-a-thatched-roof-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_02_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/cottage_p0_s250_stage1/2025-07-18-A-3D-model-of-an-adorable-cottage-with-a-thatched-roof-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_03_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/lamp_p0_s250_stage1/2025-05-24-A-lamp-casting-a-warm-glow-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_00_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/lamp_p0_s250_stage1/2025-05-24-A-lamp-casting-a-warm-glow-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_01_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/lamp_p0_s250_stage1/2025-05-24-A-lamp-casting-a-warm-glow-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_02_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/lamp_p0_s250_stage1/2025-05-24-A-lamp-casting-a-warm-glow-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_03_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/mug_p0_s250_stage1/2025-05-24-A-mug-filled-with-steaming-coffee-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_00_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/mug_p0_s250_stage1/2025-05-24-A-mug-filled-with-steaming-coffee-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_01_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/mug_p0_s250_stage1/2025-05-24-A-mug-filled-with-steaming-coffee-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_02_albedo_rgb.mp4",
        "/gpfs/share/home/2206192113/cvpr_code/prolificdreamer-main/exp_weimin/mug_p0_s250_stage1/2025-05-24-A-mug-filled-with-steaming-coffee-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-dth-0.2-tet-256/results/df_ep0250_03_albedo_rgb.mp4",
    ]

    # 2. 指定一个统一的文件夹来保存所有输出的PNG图片
    output_image_folder = "output_frames_all4brickbunny_kl"

    # --- 修改结束 ---

    process_multiple_videos(
        video_paths=input_video_files,
        output_folder=output_image_folder,
        interval_sec=0.1,
        initial_crop_size=700,
        final_size=512
    )