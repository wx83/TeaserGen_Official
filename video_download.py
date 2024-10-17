import argparse
import pathlib
import numpy as np
import utils
import yt_dlp
import os
import logging
import subprocess
import torch
import csv
from pathlib import Path
import json
import pandas as pd
import organize
from organize import organize_file

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Youtube Video Download"
    )
    parser.add_argument("-p", "--playlist", type=pathlib.Path, help="playlist names")
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-m", "--meta_dir", type=pathlib.Path, help="metadata out directory"
    )
    parser.add_argument(
        "-e",
        "--ignore_exceptions",
        action="store_true",
        help="whether to ignore all exceptions",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="number of jobs",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def download_video(video_url, cookies_path='./cookies.txt', out_dir = "./raw_video"):
    ydl_opts = {
        'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
        'cookiefile': cookies_path,
        'outtmpl': out_dir + '/%(id)s.%(ext)s',
        'merge_output_format': 'mp4',
        'noplaylist': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        logging.info(f'Successfully downloaded video: {video_url}')
    except yt_dlp.utils.DownloadError as e:
        logging.error(f'Failed to download video: {video_url} - {str(e)}')
    
def download_playlist(filename, out_dir):
    """
    out_dir: to raw vidoes and then separate
    """
    # video_path = f"./data_deepx/{genre}/video"
    os.makedirs(out_dir, exist_ok=True)
    """
    Prepare all the video names in a txt file
    """

    with open(filename, 'r', encoding='utf-8') as file:
        video_urls = [line.strip() for line in file.readlines()]

    for video_url in video_urls:
        download_video(video_url, output_path=out_dir)

    organize_file(out_dir)
    
def get_playlist_urls_and_titles(playlist_url, csv_name):
    """
    Save urls and titles for file naming purpose;
    Used for playlist purpose
    """
    # Configuration for yt-dlp to extract playlist data
    ydl_opts = {
        'quiet': True,  # Keep the output clean
        'noplaylist': False,  # Ensure the playlist's videos are treated as a playlist
        'force_generic_extractor': False,  # Allow yt-dlp to use specific extractors for YouTube
        'geo_bypass': True,  # Bypass geo-restrictions
        'skip_download': True,  # Do not download the videos
        'youtube_include_dash_manifest': False,  
    }
    

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(playlist_url, download=False)  # Extract info without downloading
        
        if 'entries' in result and result['entries']:
            # Prepare to write to CSV
            with open(csv_name, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['title', 'url'])  
                for entry in result['entries']:
                    title = entry.get('title', 'No title available')  # Get the title, default to 'No title available'
                    url = entry.get('webpage_url', 'No URL available')  # Get the URL, default to 'No URL available'
                    writer.writerow([title, url])
            print(f"Data for all playlist videos has been written to {csv_name}.")
        else:
            print("No videos found in the playlist.")



def get_playlist_metadata(video_urls, out_dir):
    """
    Extract metadata for each video in a list of URLs and save as JSON.
    Args:
        video_urls (list of str): List of YouTube video URLs.
        out_dir (str): Directory where JSON files are saved.
    """

    yt_opts = {
        'quiet': True,
        'no_warnings': True,
        'force_generic_extractor': False,
    }


    os.makedirs(out_dir, exist_ok=True)


    for url in video_urls:
        try:

            video_id = url.split('=')[-1] 
            filename = f"{video_id}.json"
            file_path = os.path.join(out_dir, filename)


            if os.path.exists(file_path):
                print(f"Metadata for video ID {video_id} already exists.")
                continue


            with yt_dlp.YoutubeDL(yt_opts) as yt_dl:
                video_info = yt_dl.extract_info(url, download=False)


            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(video_info, f, ensure_ascii=False, indent=4)
            print(f"Metadata saved for video ID {video_id}.")
        except Exception as e:
            print(f"Failed to extract data for video {url}. Error: {e}")
            
def process_playlist(playlist_url, meta_dir, out_dir, playlist_name, download = False ):
    filename = f"./data_deepx/{playlist_name}.txt"
    if os.path.exists(filename):
        print(f"The file {filename} already exists. Skipping download.")
    else:
        video_urls = get_playlist_urls_and_titles(playlist_url) # get all urls inside that playlist
        get_playlist_metadata(video_urls, meta_dir)
        utils.save_as_txt(filename, video_urls)
        if download == True:
            download_playlist(filename, out_dir)




def main(playlist_url):
    args = parse_args()
    process_playlist(playlist_url, args.meta_dir, args.out_dir, args.playlist)

if __name__ == "__main__":
    """
    national_geo = "https://www.youtube.com/playlist?list=PLivjPDlt6ApSiD2mk9Ngp-5dZ9CDDn72O"
    pbs_geo = "https://www.youtube.com/playlist?list=PLzkQfVIJun2JiMRQzj6dBPClH0r5kDPIF"
    dw = "https://www.youtube.com/@DWDocumentary"
    """
    
    pass