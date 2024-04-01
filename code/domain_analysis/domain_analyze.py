import os
import pandas as pd



def sort_by_frequency(df_records):
  

    # (1) Group by 'url_domain' and 'landing page domain' and count the occurrences of each domain
    url_domain_counts = df_records['url_domain'].value_counts().rename('domain_counts')
    landing_page_domain_counts = df_records['landing_page_domain'].value_counts().rename('domain_counts')

    # (2) Merge the counts back into the original DataFrame
    url_df = df_records.merge(url_domain_counts.to_frame(),
                left_on='url_domain',
                right_index=True)
    
    landing_page_df = df_records.merge(landing_page_domain_counts.to_frame(),
                left_on='landing_page_domain',
                right_index=True)

    # Sort the DataFrame based on the count, then by 'url_domain' if counts are equal
    df_sorted_url = url_df.sort_values(by=['domain_counts', 'url_domain'], ascending=[False, True])
    df_sorted_landing_page = landing_page_df.sort_values(by=['domain_counts', 'landing_page_domain'], ascending=[False, True])
    return df_sorted_url, df_sorted_landing_page


if __name__ == "__main__":
    
    RESULT_DIR = "../../output/domain_analysis/"
    if os.path.exists(RESULT_DIR) == False:
        os.makedirs(RESULT_DIR)
    
    # fullGraph classification
    others_folder = "../../output/others"
    ads_folder = "../../output/ads"
    affiliate_folder = "../../output/affiliate"


    # get features
    df_others_phaseA_features_all = pd.DataFrame()
    df_others_phaseA_features_simple = pd.DataFrame()
    df_others_labels = pd.DataFrame()
    df_others_records = pd.DataFrame()
    df_others_url_features = pd.DataFrame()
    
    for crawl_id in os.listdir(others_folder):
        visited = ["crawl_aff_normal_10_2", "crawl_aff_normal_260", "crawl_aff_normal_120", "crawl_aff_normal_140"]
        if crawl_id not in visited:
            continue
        each_crawl =  os.path.join(others_folder, crawl_id)
        for filename in os.listdir(each_crawl):
            
            # phase A 
            if filename == "features_phase1.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_others_phaseA_features_all = df_others_phaseA_features_all.append(df)

    
            elif filename == "records.csv" or filename == "record.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_others_records = df_others_records.append(df)
           
            else:
                continue
        
    # remove duplicate clicked others

    print("Before dedepulicate: ", len(df_others_phaseA_features_all))
    df_others_records['parent_visit_id'] = df_others_records['visit_id'].str.split('_', expand=True)[0]
    merged_df = pd.merge(df_others_records, df_others_phaseA_features_all, on='visit_id', how='inner') 
    df_others_phaseA_features_deduplicate =merged_df.drop_duplicates(subset=["name", "num_nodes", "num_edges", "parent_domain", "parent_visit_id"])
    df_others_phaseA_features_all = df_others_phaseA_features_all.merge(df_others_phaseA_features_deduplicate[['visit_id']], on=["visit_id"])
    print("After dedepulicate: ", len(df_others_phaseA_features_deduplicate))
  
    df_others_url_sort, df_others_landing_page_sort = sort_by_frequency(df_others_records)
    RESULT_DIR_Others = os.path.join(RESULT_DIR, "others")
    df_others_url_sort.to_csv(os.path.join(RESULT_DIR_Others,"records_sort_by_url_02_28.csv"))
    df_others_landing_page_sort.to_csv(os.path.join(RESULT_DIR_Others,"records_sort_by_landing_page_02_28.csv"))

    """   
    # collect all the records.csv
    df_ads_all_records = pd.DataFrame()
    df_aff_all_records = pd.DataFrame()
    df_ads_phaseA_features_all = pd.DataFrame()
    df_aff_phaseA_features_all = pd.DataFrame()

    for crawl_id in os.listdir(ads_folder):
        if "unseen" in crawl_id:
            print("\tIgnore this folder, since it is for testing")
            continue
        if "old" in crawl_id:
             print("\tIgnore this folder, since it is the old data")
             continue
        each_crawl =  os.path.join(ads_folder, crawl_id)
        for filename in os.listdir(each_crawl):
        
            if filename == "records.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_ads_all_records = df_ads_all_records.append(df)
            # phase A 
            if filename == "features_phase1.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')

                df_ads_phaseA_features_all = df_ads_phaseA_features_all.append(df)



    # remove duplicate clicked ads

    print("Before dedepulicate: ", len(df_ads_all_records))
    df_ads_all_records['parent_visit_id'] = df_ads_all_records['visit_id'].str.split('_', expand=True)[0]
    merged_df = pd.merge(df_ads_all_records, df_ads_phaseA_features_all, on='visit_id', how='inner') 
    df_ads_phaseA_features_deduplicate =merged_df.drop_duplicates(subset=["name", "num_nodes", "num_edges", "parent_domain", "parent_visit_id"])
    #df_ads_phaseA_features_deduplicate =merged_ad_phaseA_df.drop_duplicates(subset=["name", "top_level_url", "parent_domain"])
    df_ads_all_records = df_ads_all_records.merge(df_ads_phaseA_features_deduplicate[['visit_id']], on=["visit_id"])
    print("After dedepulicate: ", len(df_ads_all_records))
    print(df_ads_all_records.columns)
   


    for crawl_id in os.listdir(affiliate_folder):
        if "unseen" in crawl_id:
                print("\tIgnore this folder, since it is for testing")
                continue
        each_crawl =  os.path.join(affiliate_folder, crawl_id)
        for filename in os.listdir(each_crawl):
        
            if filename == "records.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df['visit_id'] = df['visit_id'].astype(str)
                if len(df[df['visit_id'] == '3036213338850092_2'])> 0:
                    print(file_path)
                df_aff_all_records = df_aff_all_records.append(df)
            # phase A 
            if filename == "features_phase1.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df['visit_id'] = df['visit_id'].astype(str)
                df_aff_phaseA_features_all = df_aff_phaseA_features_all.append(df)


    # remove duplicate clicked affiliate
     
    print("Before dedepulicate: ", len(df_aff_all_records))
    # Split the DataFrame into two based on whether visit_id contains "_"
    df_aff_with_underscore = df_aff_all_records[df_aff_all_records['visit_id'].str.contains('_')]
    df_aff_without_underscore = df_aff_all_records[~df_aff_all_records['visit_id'].str.contains('_')]
    df_aff_with_underscore['parent_visit_id'] = df_aff_with_underscore['visit_id'].str.split('_', expand=True)[0]
    
    merged_df = pd.merge(df_aff_with_underscore, df_aff_phaseA_features_all, on='visit_id', how='inner') 
    df_aff_all_records_dedup =merged_df.drop_duplicates(subset=["name", "num_nodes", "num_edges", "parent_domain", "parent_visit_id"])
    #df_ads_phaseA_features_deduplicate =merged_ad_phaseA_df.drop_duplicates(subset=["name", "top_level_url", "parent_domain"])
    df_aff_with_underscore_dedup = df_aff_all_records.merge(df_aff_all_records_dedup[['visit_id']], on=["visit_id"])
    df_aff_without_underscore = df_aff_all_records.merge(df_aff_without_underscore[['visit_id']], on=["visit_id"])
    df_aff_all_records = pd.concat([df_aff_without_underscore, df_aff_with_underscore_dedup])
    print("After dedepulicate: ", len(df_aff_all_records))
    print(df_aff_all_records.columns)

    print("number of ads records: ", len(df_ads_all_records))
    print("number of aff records: ", len(df_aff_all_records))    

    df_aff_url_sort, df_aff_landing_page_sort = sort_by_frequency(df_aff_all_records)
    RESULT_DIR_aff = os.path.join(RESULT_DIR, "affiliate")
    df_aff_url_sort.to_csv(os.path.join(RESULT_DIR_aff, "records_sort_by_url_02_21.csv"))
    df_aff_landing_page_sort.to_csv(os.path.join(RESULT_DIR_aff, "records_sort_by_landing_page_02_21.csv"))
    
    df_ads_url_sort, df_ads_landing_page_sort = sort_by_frequency(df_ads_all_records)
    RESULT_DIR_ads = os.path.join(RESULT_DIR, "ads")
    df_ads_url_sort.to_csv(os.path.join(RESULT_DIR_ads,"records_sort_by_url_02_21.csv"))
    df_ads_landing_page_sort.to_csv(os.path.join(RESULT_DIR_ads,"records_sort_by_landing_page_02_21.csv"))


    """  
    



    
