import time
import os
import re
import pandas as pd
from pytube import Channel


def write_content(records_folder, video_ids_list, video_type):
    if video_type == "shorts":
        records_path = os.path.join(records_folder, "shorts_video_ID.csv" )
    elif video_type == "long":
        records_path = os.path.join(records_folder, "long_video_ID.csv" )
    else:
        records_path = os.path.join(records_folder, "others_video_ID.csv")

    df = pd.DataFrame(video_ids_list, columns=['videoId'])

    if not os.path.exists(records_path):
        df.to_csv(records_path, header=True, index=False)
    else:
        existing_df = pd.read_csv(records_path)
        new_df = df[~df['videoId'].isin(existing_df['videoId'])]

        if not new_df.empty:
            print(f"\n\nNew video IDs found for {video_type}:")
            print(new_df['videoId'].tolist())
            new_df.to_csv(records_path, mode='a', header=not os.path.exists(records_path), index=False)
        else:
            print("No New video IDs")

def collect_videoID(channel_html, records_folder):

    # Patterns to find shorts and regular watch YouTube URLs
    shorts_pattern = r'"url":"/shorts/([\w-]{11})"'
    watch_pattern = r'"url":"/watch\?v=([\w-]{11})"'

    # Find all matches in the HTML content
    shorts_matches = re.findall(shorts_pattern, channel_html)
    watch_matches = re.findall(watch_pattern, channel_html)

    # find the all the video
    videoID_pattern = r'"videoId":"([\w-]{11})"'
    all_videoID_matches = re.findall(videoID_pattern, channel_html)

    # Initialize sets to keep track of unique video IDs
    video_id_short_set = set(shorts_matches)
    video_id_long_set = set(watch_matches)
    video_id_others_set = set(all_videoID_matches) - video_id_short_set - video_id_long_set


    # Write unique shorts video IDs
    if video_id_short_set:
        print("Shorts video IDs found:")
        for video_id in video_id_short_set:
            print(video_id)
        write_content(records_folder, [{'videoId': video_id} for video_id in video_id_short_set], 'shorts')
    else:
        print("No shorts video IDs found.")

    # Write unique long video IDs
    if video_id_long_set:
        print("\nWatch video IDs found:")
        for video_id in video_id_long_set:
            print(video_id)
        write_content(records_folder, [{'videoId': video_id} for video_id in video_id_long_set], 'long')
    else:
        print("No watch video IDs found.")


    # Write unique other video IDs
    if video_id_others_set:
        print("\nOther types of video IDs found:")
        for video_id in video_id_others_set:
            print(video_id)
        write_content(records_folder, [{'videoId': video_id} for video_id in video_id_others_set], 'others')
    else:
        print("\nNo other types of video IDs found.")


if __name__ == "__main__":
    # long_links = "/home/data/chensun/affi_project/purl/data/yt_data/shorts_video_ID.csv"
    # df = pd.read_csv(long_links)
    # df = df.drop_duplicates(subset=['videoId'])
    # df.to_csv("/home/data/chensun/affi_project/purl/data/yt_data/shorts_video_ID_2.csv", index=False)
    
    
    channel_url = 'https://www.youtube.com/channel/UCkYQyvc_i9hXEo4xic9Hh2g'
   
    records_folder = os.path.abspath("../data/yt_data/")
    while True:
        try:
            channel = Channel(channel_url)
            html = channel.about_html
            collect_videoID(html, records_folder)
        except Exception as e:
            print(f"An error occurred: {e}")

        print("======= Next round checking ========")
        time.sleep(900)  # Wait for 15 minutes (900 seconds)
    
    #A_path = "/home/data/chensun/affi_project/purl/data/yt_data/new.csv"
    #dfA = pd.read_csv(A_path)
    
    #B_path = '/home/data/chensun/affi_project/purl/data/yt_data/long_video_ID.csv'
    #dfB = pd.read_csv(B_path)
    #new_records = dfA[~dfA.index.isin(dfB.index)]

    # Append new records to dfB
    #result_df = pd.concat([dfB, new_records])
    #result_df.to_csv("/home/data/chensun/affi_project/purl/data/yt_data/long_video_ID_2.csv", index=False)

    



