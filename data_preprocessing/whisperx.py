import whisperx
import os
import torch
import pandas as pd
import pathlib
from pathlib import Path
import helper
from helper import load_txt, save_json, save_txt, load_json
audio_file =  "/home/weihanx/videogpt/data_deepx/documentary/separate_audio_intro/1/1akcYVlAvjE/dialog_intro.mp3"
from pydub import AudioSegment
# fix error: https://github.com/m-bain/whisperX/issues/237
device = "cuda"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
model_config = "large-v2"
cuda_index = 1


def group_by_speaker(transcript):
    speakers = {}
    for entry in transcript:
        if 'speaker' in entry:
            speaker = entry['speaker']
            if speaker not in speakers:
                speakers[speaker] = []
            # Append each entry as a tuple of (start, end, text)
            speakers[speaker].append((entry['start'], entry['end'], entry['text']))
        else:
            print("No speaker found in entry:", entry)
            continue # cannot identify speaker, exclude from output
    return speakers

def diarize_audio(video_name_path, input_wav_dir, output_dir, part, bitrate='192k'):
    video_name_list = load_txt(video_name_path)
    error_file = []
    df = pd.DataFrame(columns=['video_name', 'narrator_id'])
    for vd in video_name_list:
        print("Processing video:", vd)
        # set the destination folder
        output_path_csv = output_dir / vd[0] / vd / "diarize_segments.csv"
        output_path_csv.parent.mkdir(parents=True, exist_ok=True)

        output_path_json = output_dir / vd[0] / vd / "diarize_result.json"
        output_path_json.parent.mkdir(parents=True, exist_ok=True)

        # check existence:
        if output_path_csv.exists() and output_path_json.exists():
            print(f"Already processed video: {vd}") # already processed for botrh json and csv
            # load for narrator_id
            results = load_json(output_path_json)
            print(f"results = {results}")
            if results == {}:
                print(f"Empty results for video: {vd}")
                error_file.append(vd)
                continue
            diarize_segments = pd.read_csv(output_path_csv)
            narrator_id = get_narr_id(results, diarize_segments)
            df = pd.concat([df, pd.DataFrame({'video_name': [vd], 'narrator_id': [narrator_id]})])
            continue
        # try:
            # need process
        audio_file = input_wav_dir / vd[0] / vd / f"dialog_{part}.wav"
        model = whisperx.load_model(model_config, device, device_index=cuda_index, compute_type=compute_type) # device_index: the gpu that i will be using 
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        # # # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=device)


        diarize_segments = diarize_model(audio)
        diarize_model(audio)

        result = whisperx.assign_word_speakers(diarize_segments, result)


        diarize_segments.to_csv(output_path_csv)


        output = result['segments']
        if output == []:
            print(f"Empty output for video: {vd}")
            error_file.append(vd)
            continue
        print(f"output = {output}")

        grouped_output = group_by_speaker(output) # json is group by speaker
        save_json(output_path_json, grouped_output)


# Combining entries by speaker
def get_narr_id(output, diarize_segments):
    
    narrator_id = None
    row_num = 0
    first_row_speaker = diarize_segments.loc[row_num, 'speaker']
    while first_row_speaker not in output.keys():  # all speaker id
        row_num += 1
        first_row_speaker = diarize_segments.loc[row_num, 'speaker']
    narrator_id = first_row_speaker
    return narrator_id



def find_exact_matching_text(video_name_path,input_dir, output_dir, error_file_path):


    video_name_list = load_txt(video_name_path)
    error_list = load_txt(error_file_path)
    for vd in video_name_list:
        print("Processing video:", vd)
        if vd in error_list:
            continue
        data_frame_path = input_dir / vd[0] / vd / "diarize_segments.csv"
        transcripts_path = input_dir / vd[0] / vd / "diarize_result.json"
        if not data_frame_path.exists() and not transcripts_path.exists():
            continue
        data = load_json(transcripts_path)
        rows = []

        for speaker, segments in data.items():
            for segment in segments:
                start_time, end_time, text = segment
                text = text.strip().replace('"', '')
                rows.append([speaker, start_time, end_time, text])

        # Create DataFrame
        df = pd.DataFrame(rows, columns=["Speaker", "Start Time", "End Time", "Text"])

        # Sort the DataFrame by end Time
        df_sorted = df.sort_values(by=["End Time"]) # no overlap
        output_path = output_dir / vd[0] / vd / "script_timeline.csv"     
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_sorted.to_csv(output_path, index=False)

def narrator_selection(video_name_path, input_dir, output_dir):
    # load all narrtor txt and fild the longest as the model output
    df = pd.DataFrame(columns=['video_name', 'narrator_id'])
    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        # load all possible txt file in this folder
        print(f"processing ... {vd}")
        subfolder_path = input_dir / vd[0] / vd
        txt_files = [file for file in subfolder_path.glob('*.txt')] # remove the default narrator
        # load all txt_files and find what is the longest
        longest_txt_id = 0
        long_txt_len = 0
        
        for txt_file in txt_files:
            with open(txt_file, 'r') as file:
                content = file.read() # do not need to remove lines
            if len(content) > long_txt_len:
                long_txt_len = len(content)
                # no root path and no extension
                txt_file = Path(txt_file)

                longest_txt_id = txt_file.stem
                print(f"longest_txt_id = {longest_txt_id}")
        df = pd.concat([df, pd.DataFrame({'video_name': vd, 'narrator_id': [str(longest_txt_id)]})])
    df.to_csv(output_dir / "narrator_id_by_length.csv", index=False)

def remove_file(video_name_path, input_dir):
    # first remove all txt files
    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        try:
            print(f"processing ... {vd}")
            subfolder_path = input_dir / vd[0] / vd
            txt_files = [file for file in subfolder_path.glob('*.txt')]
            for txt_file in txt_files:
                txt_file.unlink()
        except:
            print(f"error processing ... {vd}")

def text_script(video_name_path, input_dir, output_dir):
    # first remove all txt files
    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        print(f"processing ... {vd}")
        input_json = input_dir / vd[0] / vd / "diarize_result.json"
        input_json_content = load_json(input_json)
        for speaker, content_list in input_json_content.items():
            content_str = ""
            for content in content_list:
                content_str += str(content[2])
            output_path = output_dir / vd[0] / vd / f"{speaker}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'a') as file:
                file.write(content_str)
if __name__ == "__main__":
    # diazier
    video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt") # in documentar/audio_separate_2
    input_wav_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/audio_separate_test")
    output_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_main_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    part = "main"
    diarize_audio(video_name_path, input_wav_dir, output_dir, part)

    # further notation
    video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    input_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_main_test")
    output_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_main_test")
    text_script(video_name_path, input_dir, input_dir)
    narrator_selection(video_name_path, input_dir, output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # input_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_intro_train")
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_intro_train")
    # error_file_path = Path("/home/weihanx/videogpt/whisperX/error_file.txt")
    # find_exact_matching_text(video_name_path, input_dir, output_dir, error_file_path)
    # video_name_path = Path("/home/weihanx/videogpt/processed.txt") # in documentar/audio_separate_2
    # input_wav_dir = Path("/home/weihanx/videogpt/deepx_data7/audio_extract1219_separate")
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_main_test")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # diarize_audio(video_name_path, input_wav_dir, output_dir)
