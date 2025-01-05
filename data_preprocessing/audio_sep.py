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

def diarize_audio(video_name_path, input_wav_dir, output_dir, bitrate='192k'):
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
        audio_file = input_wav_dir / vd[0] / vd / "dialog_intro.wav"
        model = whisperx.load_model(model_config, device, device_index=cuda_index, compute_type=compute_type) # device_index: the gpu that i will be using 
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        # # # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device=device)


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

        grouped_output = group_by_speaker(output)
        save_json(output_path_json, grouped_output)
        
        output_script_folder = output_dir / vd[0] / vd
        narrator_id = write_script(output, output_script_folder, diarize_segments)
        # remeber to add the narrator_id to the dataframe
        df = pd.concat([df, pd.DataFrame({'video_name': [vd], 'narrator_id': [narrator_id]})])
        # except Exception as e:
        #     print(f"Error processing video: {vd}")
        #     print(e)
        #     error_file.append(vd)
    # with open("error_file.txt", "w") as f:
    #     for item in error_file:
    #         f.write("%s\n" % item)
    output_path = output_dir / "narrator_id.csv"
    df.to_csv(output_path, index=False)
def combine_entries_by_speaker(transcript): # combine the entries by speaker into paragraph
    speakers = {}
    for entry in transcript:
        if 'speaker' not in entry:
            print("No speaker found in entry:", entry)
            continue
        speaker = entry['speaker']
        if speaker not in speakers:
            speakers[speaker] = {
                'start': entry['start'],
                'end': entry['end'],
                'text': entry['text'].strip()
            }
        else:
            # Update the end time
            speakers[speaker]['end'] = entry['end']
            # Combine text into a single paragraph, adding a space before the next sentence
            speakers[speaker]['text'] += " " + entry['text'].strip()
    return speakers

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

def write_script(output, output_folder, diarize_segments):
    combined_output = combine_entries_by_speaker(output) 
    narrator_id = get_narr_id(combined_output, diarize_segments)

    for speaker, details in combined_output.items():
        filename = output_folder / f"{speaker.replace(' ', '_').lower()}.txt"  # Formatting the filename to be lowercase and replace spaces with underscores

        with open(filename, 'w') as file:
            file.write(details['text'])
        print(f"Saved: {filename}")
        if str(speaker) == str(narrator_id): # only one key, it is the narrator
            filename = output_folder / f"narrator.txt"
            with open(filename, 'w') as file:
                file.write(details['text'])
            print(f"Saved narrator : {filename}")
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

if __name__ == "__main__":
    video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/intro_error.txt") # in documentar/audio_separate_2
    input_wav_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/separate_audio_intro")
    output_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_intro_train")
    output_dir.mkdir(parents=True, exist_ok=True)
    diarize_audio(video_name_path, input_wav_dir, output_dir)

    # input_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_intro_train")
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_intro_train")
    # error_file_path = Path("/home/weihanx/videogpt/whisperX/error_file.txt")
    # find_exact_matching_text(video_name_path, input_dir, output_dir, error_file_path)
    # video_name_path = Path("/home/weihanx/videogpt/processed.txt") # in documentar/audio_separate_2
    # input_wav_dir = Path("/home/weihanx/videogpt/deepx_data7/audio_extract1219_separate")
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_main_test")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # diarize_audio(video_name_path, input_wav_dir, output_dir)
