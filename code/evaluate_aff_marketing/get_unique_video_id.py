import pandas as pd
import os


def get_unique_aff_video_from_clicked(RESULT_DIR):

    file_path = '/home/data/chensun/affi_project/purl/output/domain_analysis/affiliate/yt_records_sort_by_landing_page.csv'  
    df = pd.read_csv(file_path)

    df_unique = df.drop_duplicates(subset=['parent_page_url'])

    df_unique.rename(columns={'parent_page_url': 'youtube_video'}, inplace=True)

    youtube_videos = df_unique[['youtube_video']]


    output_name = 'aff_unique_clicked.csv' 
    output_path = os.path.join(RESULT_DIR, output_name)
    youtube_videos.to_csv(output_path, index=False)

    print("File saved successfully.")
  
  
def get_unique_non_aff_video_from_clicked(RESULT_DIR):

    file_path = '/home/data/chensun/affi_project/purl/output/domain_analysis/others/yt_records_sort_by_landing_page.csv'  
    df = pd.read_csv(file_path)

    df_unique = df.drop_duplicates(subset=['parent_page_url'])

    df_unique.rename(columns={'parent_page_url': 'youtube_video'}, inplace=True)

    youtube_videos = df_unique[['youtube_video']]


    output_name = 'non_aff_unique_clicked.csv' 
    output_path = os.path.join(RESULT_DIR, output_name)
    youtube_videos.to_csv(output_path, index=False)

    print("File saved successfully.")  
    
    
def get_unique_ignored_video_from_yt_records(RESULT_DIR):
    unique_video = pd.DataFrame() 
    yt_records_path = '/home/data/chensun/affi_project/yt_records'  
   
    for each_folder in os.listdir(yt_records_path):
        if 'dedup_links.ipynb' in each_folder:
            continue
        each_folder_path = os.path.join(yt_records_path, each_folder)
        for each_table in os.listdir(each_folder_path):
            each_file_path = os.path.join(each_folder_path, each_table)
            if 'ignored' in each_table:
                print(f'processing {each_file_path}')
                df = pd.read_csv(each_file_path)

                df_unique = df.drop_duplicates(subset=['site'])
                
                df_unique.rename(columns={'site': 'youtube_video'}, inplace=True)

                youtube_videos = df_unique[['youtube_video']]
                
                unique_video = pd.concat([unique_video, youtube_videos], ignore_index=True)
          
    unique_video.to_csv(os.path.join(RESULT_DIR, 'unique_videos_from_yt_records_ignored.csv'), index=False)
                
       
def combine_unique_video(RESULT_DIR):
    unique_video = pd.DataFrame() 
    for each_file in os.listdir(RESULT_DIR):
        if 'csv' in each_file:
            
            each_file_path = os.path.join(RESULT_DIR, each_file)
            print(f'processing {each_file_path}')
            df = pd.read_csv(each_file_path)

            df_unique = df.drop_duplicates(subset=['youtube_video'])
                
            unique_video = pd.concat([unique_video, df_unique], ignore_index=True)
          
    unique_video.to_csv(os.path.join(RESULT_DIR, 'unique_videos_combined.csv'), index=False)
                
            
            
        
                
if __name__ == "__main__":
    RESULT_DIR = "../../output/youtube_affiliate_evaluation/"
    if os.path.exists(RESULT_DIR) == False:
        os.makedirs(RESULT_DIR)  
    
    
    # 1: get all the affiliate video from crawl
    get_unique_aff_video_from_clicked(RESULT_DIR)
    
    # get all the non-aff video from crawl
    get_unique_non_aff_video_from_clicked(RESULT_DIR)
    
    
    # get all the ignored video from yt_records
    get_unique_ignored_video_from_yt_records(RESULT_DIR)
    
    # 2: get all the affiliate video from 'youtube_ignored_high_freq_links.csv'
    # TODO: Need seperate aff and non-aff from this csv file
   
   
    
    
    # 3: get all the unique video id (crawl + ignored) and (aff + others)
    # (a). affiliate folder + others folder
    # (b). yt_records -- 'site' columns
    combine_unique_video(RESULT_DIR)
    
