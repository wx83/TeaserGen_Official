"""
Divide Video into sliced videos
"""
import ffmpeg
from pathlib import Path
import utils

def split_video(video_path, clip_dir, num_clips):
    video_path = Path(video_path)
    # Get the total duration of the video
    probe = ffmpeg.probe(str(video_path))
    duration = float(probe['streams'][0]['duration'])

    # Calculate duration of each clip
    clip_length = duration / num_clips

    # Generate clips
    for i in range(num_clips):
        start_time = i * clip_length
        output_path = clip_dir / f"clip_{i}.mp4"
        
        # Command to cut the video
        ffmpeg.input(str(video_path), ss=start_time, t=clip_length).output(str(output_path), codec="copy").run(capture_stdout=True, capture_stderr=True)

def split_video_scale(video_names, input_dir, out_dir, num_clips, part="main"):
    video_urls = utils.load_txt(video_names)
    for video_name in video_urls:
        video_path = input_dir / video_name[0] / video_name / f"{part}.mp4"
        clip_dir = out_dir / video_name[0] / video_name 
        clip_dir.mkdir(parents = True, exist_ok = True)
        split_video(video_path, clip_dir, num_clips)

def split_video_scale_zeroshot(video_names, input_dir, out_dir, num_clips):
    video_urls = utils.load_txt(video_names)
    for video_name in video_urls:
        video_path = input_dir /  f"{video_name}.mp4"
        clip_dir = out_dir / video_name[0] / video_name 
        clip_dir.mkdir(parents = True, exist_ok = True)
        split_video(video_path, clip_dir, num_clips)

if __name__ == "__main__":
    # Pass
    # video_names = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_name.txt")
    # input_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshotvideo")
    # out_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshotvideo_clip")
    # num_clips = 10  # Number of clips to split the video into
    # split_video_scale_zeroshot(video_names, input_dir, out_dir, num_clips)


"""
Run zero-shot highlight detection with UniVTG
"""