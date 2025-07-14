import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=30):
    """
    从视频中提取帧
    :param video_path: 视频文件路径
    :param output_dir: 输出目录
    :param fps: 每秒提取的帧数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 检查视频FPS是否有效
    if video_fps <= 0:
        print(f"警告：视频 {video_path} 的FPS无效 ({video_fps})，使用默认值30")
        video_fps = 30
    
    # 计算需要提取的帧间隔
    frame_interval = max(1, int(video_fps / fps))
    
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 使用tqdm创建进度条
    pbar = tqdm(total=total_frames, desc=f"处理视频 {video_name}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 每隔frame_interval帧保存一次
        if frame_count % frame_interval == 0:
            # 生成输出文件名
            output_path = os.path.join(output_dir, f"{video_name}_frame_{saved_count:04d}.jpg")
            # 保存帧
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
        pbar.update(1)
    
    # 释放资源
    cap.release()
    pbar.close()
    
    print(f"视频 {video_name} 处理完成！")
    print(f"视频FPS: {video_fps}")
    print(f"提取间隔: {frame_interval} 帧")
    print(f"总共提取了 {saved_count} 帧图片")

def main():
    # 视频文件列表
    video_files = ['a.mp4']
    
    # 输出目录
    output_dir = './dataset/sharp'
    
    # 处理每个视频文件
    for video_file in video_files:
        if os.path.exists(video_file):
            print(f"\n开始处理视频: {video_file}")
            extract_frames(video_file, output_dir)
        else:
            print(f"错误：找不到视频文件 {video_file}")

if __name__ == '__main__':
    main() 