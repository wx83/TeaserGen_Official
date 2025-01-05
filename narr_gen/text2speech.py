import torch
from TTS.api import TTS
import pathlib
from pathlib import Path
import json
from config import tts
import librosa
from helper import save_json, load_json
def load_json(filename):
    """Load data from a JSON file."""
    with open(filename, encoding="utf8") as f:
        return json.load(f)
#  no cloning tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path=OUTPUT_PATH)
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=YOUR_API_KEY,
)

def get_speech_clip_UGT(num_clips, json_path, out_dir, video_name, storylike, language="en"):
    queries_json = load_json(json_path)
    duration_json = {}
    for clip_num in range(num_clips):
        if f"video_{clip_num}" in queries_json.keys():
            sent_sum = queries_json[f"video_{clip_num}"]
            story_sum = " ".join(sent_sum)
            out_path = out_dir / video_name[0] / video_name / f"clip_{clip_num}_UGT_speech.wav"
            # Check if the file already exists
            if not out_path.exists():
                # Call your TTS function if the file does not exist
                tts.tts_to_file(text=story_sum, file_path=str(out_path))
                print(f"Audio file created: {out_path}")
            else:
                # Handle the case where the file already exists
                print(f"Audio file already exists: {out_path}")

            audio, sample_rate = librosa.load(str(out_path), sr=None)  # sr=None ensures original sample rate is used
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            duration_json[f"clip_{clip_num}"] = duration
        else:
            duration_json[f"clip_{clip_num}"] = 0 # no need for selection
    time_duration_json = out_dir / video_name[0] / video_name / f"UGT_duration.json"
    save_json(filename=time_duration_json, data=duration_json)
    return time_duration_json

def gen_tts(narr_json_path, group_index_list, out_dir, video_name, chunk_num):
    narr_dict = load_json(narr_json_path)
    sentence_script = []
    out_path = out_dir / video_name[0] / video_name / f"chunk_{chunk_num}_speech.wav"
    if not out_path.exists():
        for i in group_index_list: # clip index number
            sentence_script.append(narr_dict["storylike"][f"clip_{i}"])
        joined_sentence = " ".join(sentence_script) 
        print(f"joined_sentence = {joined_sentence}")
        # tts.tts_to_file(text=joined_sentence, file_path=str(out_path))
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=joined_sentence,
        )
        response.stream_to_file(out_path)
        audio, sample_rate = librosa.load(str(out_path), sr=None)
        duration = librosa.get_duration(y=audio, sr=sample_rate)

    else:
        print(f"Audio file already exists: {out_path}")
    
    return joined_sentence, duration

def get_speech_clip(num_clips, json_path, out_dir, video_name, storylike, language="en"):
    """
    speaker wav: should be the narrator's voice
    """
    queries_json = load_json(json_path)
    map_list = queries_json["map_list"]
    ending_q = queries_json["ending_question"]
    duration_json = {}
    chunk_num = 0

    if storylike == True: # num_clips = 10 == len of map list
        start_index = 0
        video_index = map_list[start_index]
        group = [start_index]
        for i in range(1, len(map_list)):
            if map_list[i] == map_list[start_index]:
                group.append(i) # index of sentences should be grouped together
            else:
                # video clip number: map_list[i]
                # the lip number is the last i

                # tuple: chunk i: (video_index, duration)
                script, duration = gen_tts(json_path, group, out_dir, video_name, chunk_num)
                duration_json[f"chunk_{chunk_num}"] = {
                    "video_index": video_index, 
                    "duration": duration,
                    "sentence": script
                }
                
                chunk_num += 1
                start_index = i
                video_index = map_list[start_index]
                group = [start_index] # last one is a new group
        # last chunk
        script, duration = gen_tts(json_path, group, out_dir, video_name, chunk_num)
        duration_json[f"chunk_{chunk_num}"] = {
            "video_index": video_index, 
            "duration": duration,
            "sentence": script
        }
    else:
        for clip_num in range(num_clips):
        # if it is not story like, no ending q is added
            sent_sum = queries_json["sentence_summary"][f"clip_{clip_num}"]
            out_path = out_dir / video_name[0] / video_name / f"clip_{clip_num}_speech.wav"

            # Check if the file already exists
            if not out_path.exists():
                # Call your TTS function if the file does not exist
                tts.tts_to_file(text=sent_sum, file_path=str(out_path))
                print(f"Audio file created: {out_path}")
            else:
                # Handle the case where the file already exists
                print(f"Audio file already exists: {out_path}")
            audio, sample_rate = librosa.load(str(out_path), sr=None)  # sr=None ensures original sample rate is used
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            queries_json["sumlen"][f"clip_{clip_num}"] = duration
    time_duration_json = out_dir / video_name[0] / video_name / f"duration.json"
    save_json(filename=time_duration_json, data=duration_json)
    return time_duration_json
            # print(f"num_clips = {clip_num}, input = {sent_sum}")



if __name__ == "__main__":
    print(f"text2speechdebug")
    json_path = Path("/home/weihanx/videogpt/deepx_data7/prelim_cut/1/1akcYVlAvjE/narr_script.json")
    out_dir = Path("/home/weihanx/videogpt/deepx_data7/prelim_cut")
    video_name = "1akcYVlAvjE"
    get_speech_clip(10, json_path,out_dir,video_name, True)
