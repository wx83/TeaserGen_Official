import pathlib
from pathlib import Path
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
from helper import load_txt
import numpy as np
from datetime import timedelta
from moviepy.editor import VideoFileClip
from pathlib import Path
import requests
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

"""
In this script, we first detect scene and then generate scene level description with pretrained model
"""
def parse_time(frame_timecode):
    # Assume get_seconds() returns time in seconds as a float
    total_seconds = frame_timecode.get_seconds()
    # print(f"total_seconds = {total_seconds}")
    return total_seconds

def scene_detect(video_name_path, input_dir, output_dir):
    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        print(f"vd = {vd}")
        output_path = output_dir / vd[0] / vd / "scene.npy"
        try:
            input_path = input_dir / vd[0] / vd / "intro.mp4"
            # total len of mp4
            clip = VideoFileClip(str(input_path))
            video_duration = clip.duration
            scene_list = detect(str(input_path), AdaptiveDetector())
            total_seconds = [parse_time(end) for _, end in scene_list]
            # add 0 at the beginning and total length at the end
            total_seconds = [0] + total_seconds + [video_duration] # pad the duration of video just for extrac information
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"output_path = {total_seconds}")
            np.save(output_path, total_seconds)
        #     break
        except Exception as e:
            print(f"Error: {e}, error file name = {vd}")
            continue    

def scene_description(video_name_path, image_input_dir, scene_input_dir, output_dir): # every 3 seconds
    video_name_list = load_txt(video_name_path)

    for vd in video_name_list:
        print(f"vd = {vd}")
        scene_description = pd.DataFrame(columns = ["Start Time", "End Time", "scene start description", "scene end description"])
        scene_input = scene_input_dir / vd[0] / vd / "scene.npy"
        scene_input_npy = np.load(scene_input)
        print(f"scene_input_npy = {scene_input_npy}")
        csv_output_dir = output_dir / vd[0] / vd 
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_output_dir / "scene_description.csv"
        if csv_path.exists():
            print(f"csv_path = {csv_path} exists")
            continue
        for i in range(len(scene_input_npy) - 1): # last one intentally for duration
            
            start = round(scene_input_npy[i])
            if start == 0:
                start = 1 # avoid 0 due to difference in frame and second
            end = round(scene_input_npy[i+1])
            # combine scene start and end time description
            start_frame_path = image_input_dir / vd[0] / vd / f"frame_{start:05}.jpg"
            end_frame_path = image_input_dir / vd[0] / vd / f"frame_{end:05}.jpg"
            start_description = image_caption(start_frame_path)
            end_description = image_caption(end_frame_path)
            scene_description = pd.concat([scene_description, pd.DataFrame([{"Start Time": start, "End Time": end, "scene start description": start_description, "scene end description": end_description}])], ignore_index=True)

        scene_description.to_csv(csv_path, index=False)


def image_caption(image_path):

    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    model = model.to("cuda")
    # Describe this image in detail: or an image of: less detail
    prompt = "<grounding>Describe this image in detail:"

    # url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
    # image = Image.open(requests.get(url, stream=True).raw)

    # # The original Kosmos-2 demo saves the image first then reload it. For some images, this will give slightly different image input and change the generation outputs.
    # image.save("new_image.jpg")
    image = Image.open(image_path)

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()} # all on cuda
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    processed_text, entities = processor.post_process_generation(generated_text)
    print(processed_text)
    return processed_text

if __name__ == "__main__":
    video_name_path = Path("/home/weihanx/videogpt/whisperX/good_file.txt")
    # input_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/sliced_video")
    image_input_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/body_frame_train")
    scene_input_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/scene_intro")
    output_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/scene_intro_description")
    # output_dir = Path("/home/weihanx/videogpt/deepx_data7/scene_intro")
    output_dir.mkdir(parents=True, exist_ok=True)
    # scene_detect(video_name_path, input_dir, output_dir)
    scene_description(video_name_path, image_input_dir, scene_input_dir, output_dir)
#     video_path = Path(input_dir) / video_name_path
#     scene_list = detect(video_path, AdaptiveDetector())
#     split_video_ffmpeg(video_path, scene_list, output_dir)
# scene_list = detect('/home/weihanx/videogpt/data_deepx/documentary/sliced_video/0/0_wV9LXjRf0/intro.mp4', AdaptiveDetector())
# # print(f"scene list = {scene_list}")
# # split_video_ffmpeg('/home/weihanx/videogpt/data_deepx/documentary/sliced_video/0/0_wV9LXjRf0/intro.mp4', scene_list)
# # print(f"type of scene list = {type(scene_list)}, inside = {type(scene_list[0][1])}")
# # from datetime import datetime



# split_points_seconds = [parse_time(end) for _, end in scene_list]
# print(f"split_points_seconds = {split_points_seconds}")
