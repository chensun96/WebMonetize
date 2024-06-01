import pandas as pd
import os
from pytubefix import YouTube



def fetch_channelID(url):
    try:
        yt = YouTube(url)
        channelID = yt.channel_id
        print(f'Channel ID for {url}: {channelID}')
        return channelID
    except Exception as e:
        print(f'Failed to fetch details for {url}: {e}')
        return None

def update_and_fetch_new_videos(df, df_existing):
    # Merge both DataFrames to find non-existing entries in df_existing
    merged_df = pd.merge(df, df_existing, on='youtube_video', how='left', indicator=True)
    new_entries = merged_df[merged_df['_merge'] == 'left_only']

    # Fetch channel IDs for new entries
    if not new_entries.empty:
        new_entries['channelID'] = new_entries['youtube_video'].apply(fetch_channelID)
        # Append new entries to existing DataFrame
        updated_df = pd.concat([df_existing, new_entries[['youtube_video', 'channelID']]])
        return updated_df
    else:
        return df_existing

RESULT_DIR = "../../output/youtube_affiliate_evaluation/"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

result_name = 'video_channel_ID.csv'
result_path = os.path.join(RESULT_DIR, result_name)

file_path = '/home/data/chensun/affi_project/purl/output/youtube_affiliate_evaluation/unique_youtube_video.csv'
df = pd.read_csv(file_path)

if not os.path.exists(result_path):
    # Step 1: Fetch all the channelID if result_path does not exist
    df['channelID'] = df['youtube_video'].apply(fetch_channelID)
    df.to_csv(result_path, index=False)
else:
    # Step 2: If result_path exists, append new YouTube videos into df_result
    df_result = pd.read_csv(result_path)
    updated_df = update_and_fetch_new_videos(df, df_result)
    updated_df.to_csv(result_path, index=False)

print("Operation completed.")

  
# blocked in this machine  
# see the result in mac 2 "youtube_reddit_dataset/youtube_affiliate_evalution"