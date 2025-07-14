import cv2
import os
import random

# 视频根目录
video_dir = '.'  # 当前目录
output_dir = './test_images'
os.makedirs(output_dir, exist_ok=True)

# 视频文件名列表
video_files = ['a.mp4', 'a1.mp4', 'a2.mp4', 'a3.mp4']

frames_per_video = 500

for video_name in video_files:
    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < frames_per_video:
        print(f"{video_name} 总帧数不足 {frames_per_video}，将全部提取")
        frame_indices = list(range(total_frames))
    else:
        frame_indices = sorted(random.sample(range(total_frames), frames_per_video))
    save_count = 0
    frame_id = 0
    idx = 0
    while cap.isOpened() and idx < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id == frame_indices[idx]:
            save_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_{frame_id:05d}.jpg")
            cv2.imwrite(save_path, frame)
            save_count += 1
            idx += 1
        frame_id += 1
    cap.release()
    print(f"{video_name} 提取了 {save_count} 张图片")