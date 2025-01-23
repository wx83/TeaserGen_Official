"""
Divide into chunks for GPT4 to process, save as a summary
"""
import nltk
import json
from nltk.tokenize import sent_tokenize
import utils
import numpy
import os
import pandas as pd
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
# Ensure you have the Punkt sentence tokenizer models downloaded
nltk.download('punkt')
def save_json(filename, data):
    """Save data as a JSON file."""
    with open(filename, "w", encoding="utf8") as f:
        json.dump(data, f)
def save_txt(filename, data):
    """Save a list to a TXT file."""
    with open(filename, "w", encoding="utf8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt(filename):
    """Load a TXT file as a list."""
    with open(filename, encoding="utf8") as f:
        return [line.strip() for line in f]



def split_into_chunks(text, n=10):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Calculate the number of sentences per chunk
    k, m = divmod(len(sentences), n)
    
    # Generate the chunks
    chunks = []
    for i in range(n):
        # Calculate start and end indices for each chunk
        start_index = i * k + min(i, m)
        end_index = start_index + k + (1 if i < m else 0)
        chunk = ' '.join(sentences[start_index:end_index])
        chunks.append(chunk)
    
    return chunks

# Example text



# Display the chunks
# for i, chunk in enumerate(chunks):
    # print(f"Chunk {i+1}:\n{chunk}\n")
def find_closest(sent, summary_text_list, device = "cuda:2"):
    assert len(summary_text_list) == 10 # suppose there are 10 video clips

    # Load the model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')
    model = model.to(device)
    # Encode all the sentences
    embeddings = model.encode(summary_text_list)
    given_sentence_embedding = model.encode([sent])

    # Calculate cosine similarities
    similarities = cosine_similarity(given_sentence_embedding, embeddings)

    # Find the index of the closest sentence
    closest_idx = np.argmax(similarities)

    # Output the closest sentence
    closest_sentence = summary_text_list[closest_idx]
    print(f"The closest sentence to '{sent}' is '{closest_sentence}'")

    return closest_idx


   # find the closest and return the index
   
def get_chunk_summary(transcript_path, output_path, output_csv_path, video_title):
    """
    input: original transcirpt
    # load_path, narr_path, output_csv_path, video_title
    """
    summarizer = {}

    summary_text = [] 
    if 'sentence_summary' not in summarizer:
        summarizer['sentence_summary'] = {}  # Or use another appropriate default value like list or int
    if 'storylike' not in summarizer:
        summarizer['storylike'] = {}  # Or use another appropriate default value like list or int

    with open(transcript_path, 'r', encoding='utf-8') as file:
      text = file.read()   
      chunks = split_into_chunks(text) # return a list of paragraph
      
    for index, content in enumerate(chunks):
      prompt = "Summarize the paragraph in one sentence:" + str(content)
      print(f"current prompt = {prompt}")
      system_prompt = "You are a narrator for this story."
      summary = call_gpt(prompt, system_prompt, "gpt-4o")
      summarizer["sentence_summary"][f"clip_{index}"] = summary
      summary_text.append(summary) 
    long_string = ' '.join(summary_text)
    
    # prompt2 = f"Rephrase the following paragraph to create a brief and engaging story opening within 10 sentences. Ensure the story includes the main conflicts and remains engaging and concise" + str(long_string)
    # 0824 prompts: 
    prompt2 = f"Rewrite the paragraph into an engaging story opening in 10 sentences or less, keeping all names and avoiding being replaced by pronouns." + str(long_string)
    story_summary = call_gpt(prompt2, system_prompt, "gpt-4o")
    summarizer["intro_script"] = story_summary

    prompt3 = f"Given the title {video_title} and the provided summary, formulate one thought-provoking and concise question that relate directly to the summary" + str(long_string)
    ending_q = call_gpt(prompt3, system_prompt, "gpt-4o")
    summarizer["ending_question"] = ending_q

    # truncate into sentences
    sentences = sent_tokenize(story_summary)
    sentences.append(ending_q)

    # prompt2 = f"Generate an appealing and short narration script for the following story" + str(long_string)
    # prompt2 = f"Generate an appealing and short documentary introductionary narration for the following story, including the main characters and main conflict:{str(long_string)}"
    # map_list = []
    # for index, sent in enumerate(sentences):
    #     summarizer["storylike"][f"clip_{index}"] = sent
    #     # find the most similar words
    #     closest_idx = find_closest(sent, summary_text)
    #     # summarizer["storylike"][f"clip_{index}_video"] = closest_idx # video query number
    #     # video query to get 
    #     map_list.append(int(closest_idx))
    # summarizer["map_list"] = map_list
    # prompt3: inspring questions

    # Create a list of dictionaries to store the id and sentences
    data = [{"id": index, "text": sent} for index, sent in enumerate(sentences)]

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    save_json(output_path, summarizer)


# # Split the text into chunks
# def split_into_chunks_scale(video_name, input_dir, out_dir):
#     video_urls = utils.load_txt(video_name)
#     for video_name in video_urls:
#       transcript_path = input_dir / video_name[0] / video_name / "dialog_main.txt"
#       output_path = out_dir /  video_name[0] / video_name / "dialog_main_summary.json"
#       get_chunk_summary(transcript_path, output_path) # one video one summary

"""
Call ChatGPT 4 to do summarization: should be one sentence summarization
"""

# lock = threading.lock()
@retry(wait= wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100)) # solve wait limit
def call_gpt(prompt, system_prompt, model_name = "gpt-3.5-turbo"):
  client = OpenAI(
      # This is the default and can be omitted
      api_key=YOUR_API_KEY,
  )

  chat_completion = client.chat.completions.create(
      messages=[
          {
              "role": "system",
              "content": system_prompt,
          }, 
          {
              "role": "user",
              "content": prompt,
          }
      ],
      model= model_name,
      seed = 42,
      max_tokens = 500,
      temperature=0.0, # probability distribution
  )

  # return the contents
  return chat_completion.choices[0].message.content

def prepare_narr_text(video_name, video_title, sample_num, input_dir=Path("/home/weihanx/videogpt/data_deepx/documentary/audio_dialog_text"), out_dir = Path("/home/weihanx/videogpt/deepx_data7/prelim_cut")):
    load_path = input_dir /  video_name[0] / video_name / "dialog_main.txt"
    narr_path = out_dir / video_name[0] / video_name / "narr_script.json"
    narr_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path = out_dir / video_name[0] / video_name / f"story_script_{sample_num}.csv"
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
#    if not output_csv_path.exists():
    # transcript_path, output_path, output_csv_path, video_title
    get_chunk_summary(load_path, narr_path, output_csv_path, video_title) # one video one summary
#    else:
#         # Handle the case where the file already exists
#     print(f"Audio file already exists: {narr_path}")

    return narr_path

   # step 2: split into chunks 
   # step 3: get one sentences for each chunk
   # step 4: prompt: generate a story based on this and split into 10 sentences
   # step 5: prompt:The title is: Generate an inspiring question that describe the main conflicts in this documentary based on the transcription and title
   # step 6: save at a json: tts_transcription.json
"""
Example
"""

if __name__ == "__main__":
    video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    video_names = load_txt(video_name_path)
    meta_csv = Path('/home/weihanx/videogpt/workspace/start_code/eval/test_metadata.csv') # test data title csv
    meta_csv_df = pd.read_csv(meta_csv)
    input_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/audio_dialog_text")
    output_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/narration_0824_sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in meta_csv_df.iterrows():

        VIDEO_NAME = row["VIDEO_NAME"]
        VIDEO_TITLE = row["VIDEO_TITLE"]
        if VIDEO_NAME not in video_names:
            continue
        narr_path = output_dir / VIDEO_NAME[0] / VIDEO_NAME / "narr_script.json"
        
        # if not narr_path.exists():
        prepare_narr_text(VIDEO_NAME, VIDEO_TITLE, 3, input_dir, output_dir) # 0824 replace with new prompts
            

    