import os
import pandas as pd



def sort_by_frequency(df_records):
  

    # (1) Group by 'url_domain' and 'landing page domain' and count the occurrences of each domain
    # url_domain_counts = df_records['url_domain'].value_counts().rename('domain_counts')
    landing_page_domain_counts = df_records['landing_page_domain'].value_counts().rename('domain_counts')

    # (2) Merge the counts back into the original DataFrame
    
    landing_page_df = df_records.merge(landing_page_domain_counts.to_frame(),
                left_on='landing_page_domain',
                right_index=True)

    # Sort the DataFrame based on the count, then by 'url_domain' if counts are equal
    # df_sorted_url = url_df.sort_values(by=['domain_counts', 'url_domain'], ascending=[False, True])
    df_sorted_landing_page = landing_page_df.sort_values(by=['domain_counts', 'landing_page_domain'], ascending=[False, True])
    return df_sorted_landing_page


if __name__ == "__main__":
    
    RESULT_DIR = "../../output/domain_analysis/"
    if os.path.exists(RESULT_DIR) == False:
        os.makedirs(RESULT_DIR)
    
    # fullGraph classification
    others_folder = "../../output/rule_based_others_yt"
    affiliate_folder = "../../output/rule_based_aff_yt"


    # get features
    df_others_phaseA_features_all = pd.DataFrame()
    df_others_phaseA_features_simple = pd.DataFrame()
    df_others_labels = pd.DataFrame()
    df_others_records = pd.DataFrame()
    df_others_url_features = pd.DataFrame()
    
    for crawl_id in os.listdir(others_folder):
        #if crawl_id not in visited:
        #    continue
        each_crawl =  os.path.join(others_folder, crawl_id)
        for filename in os.listdir(each_crawl):
            
            if filename == "records.csv" or filename == "record.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df['crawl_id'] = crawl_id 
                df_others_records = df_others_records.append(df)
           
            #else:
            #    continue
        

    df_aff_all_records = pd.DataFrame()
    df_aff_phaseA_features_all = pd.DataFrame()    

    for crawl_id in os.listdir(affiliate_folder):
        
        each_crawl =  os.path.join(affiliate_folder, crawl_id)
        for filename in os.listdir(each_crawl):
            
            if filename == "records.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df['crawl_id'] = crawl_id 
                #df['visit_id'] = df['visit_id'].astype(str)
                df_aff_all_records = df_aff_all_records.append(df)

    
    print("number of aff unique video: ", len(df_aff_all_records.drop_duplicates(subset=['parent_page_url'])))
    print("number of others unique video: ", len(df_others_records.drop_duplicates(subset=['parent_page_url'])))


    print("number of aff records: ", len(df_aff_all_records))    
    print("number of others records: ", len(df_others_records)) 
    

    df_aff_landing_page_sort = sort_by_frequency(df_aff_all_records)
    RESULT_DIR_others = os.path.join(RESULT_DIR, "affiliate")
    df_aff_landing_page_sort.to_csv(os.path.join(RESULT_DIR_others, "yt_records_sort_by_landing_page.csv"))
    
    
    df_others_landing_page_sort = sort_by_frequency(df_others_records)
    RESULT_DIR_ads = os.path.join(RESULT_DIR, "others")
    df_others_landing_page_sort.to_csv(os.path.join(RESULT_DIR_ads,"yt_records_sort_by_landing_page.csv"))
   
    # save domains with counts > 200
    aff_high_freq_domains = df_aff_landing_page_sort[df_aff_landing_page_sort['domain_counts'] > 200]
    aff_high_freq_domains = aff_high_freq_domains[['landing_page_domain', 'domain_counts']]
    aff_high_freq_domains = aff_high_freq_domains.drop_duplicates(subset=['landing_page_domain'])
    aff_high_freq_domains.to_csv(os.path.join(RESULT_DIR, "aff_high_freq_landing_pages_200.csv"))

    non_aff_high_freq_domains = df_others_landing_page_sort[df_others_landing_page_sort['domain_counts'] > 200]
    non_aff_high_freq_domains = non_aff_high_freq_domains[['landing_page_domain', 'domain_counts']]
    non_aff_high_freq_domains = non_aff_high_freq_domains.drop_duplicates(subset=['landing_page_domain'])
    non_aff_high_freq_domains.to_csv(os.path.join(RESULT_DIR, "non-aff_high_freq_landing_pages_200.csv"))


    