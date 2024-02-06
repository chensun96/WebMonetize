import os
import pandas as pd


def sort_by_frequency(df):
    # (1) Group by 'url_domain' and 'landing page domain' and count the occurrences of each domain
    url_domain_counts = df['url_domain'].value_counts().rename('domain_counts')
    landing_page_domain_counts = df['landing_page_domain'].value_counts().rename('domain_counts')

    # (2) Merge the counts back into the original DataFrame
    url_df = df.merge(url_domain_counts.to_frame(),
                left_on='url_domain',
                right_index=True)
    
    landing_page_df = df.merge(landing_page_domain_counts.to_frame(),
                left_on='landing_page_domain',
                right_index=True)

    # Sort the DataFrame based on the count, then by 'url_domain' if counts are equal
    df_sorted_url = url_df.sort_values(by=['domain_counts', 'url_domain'], ascending=[False, True])
    df_sorted_landing_page = landing_page_df.sort_values(by=['domain_counts', 'landing_page_domain'], ascending=[False, True])
    return df_sorted_url, df_sorted_landing_page


if __name__ == "__main__":
# fullGraph classification
    normal_folder = "../../output/normal"
    ads_folder = "../../output/ads"
    affiliate_folder = "../../output/affiliate"

    # collect all the records.csv
    
    df_ads_all_records = pd.DataFrame()
    df_aff_all_records = pd.DataFrame()
    for crawl_id in os.listdir(ads_folder):
        if "unseen" in crawl_id:
                print("\tIgnore this folder, since it is for testing")
                continue
        each_crawl =  os.path.join(ads_folder, crawl_id)
        for filename in os.listdir(each_crawl):
            if filename == "records.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_ads_all_records = df_ads_all_records.append(df)



    for crawl_id in os.listdir(affiliate_folder):
        if "unseen" in crawl_id:
                print("\tIgnore this folder, since it is for testing")
                continue
        each_crawl =  os.path.join(affiliate_folder, crawl_id)
        for filename in os.listdir(each_crawl):
            if filename == "records.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_aff_all_records = df_aff_all_records.append(df)

    print("number of ads records: ", len(df_ads_all_records))
    print("number of aff records: ", len(df_aff_all_records))    

    df_aff_url_sort, df_aff_landing_page_sort = sort_by_frequency(df_aff_all_records)
    df_aff_url_sort.to_csv("/home/data/chensun/affi_project/purl/output/domain_analysis/affiliate/records_sort_by_url.csv")
    df_aff_landing_page_sort.to_csv("/home/data/chensun/affi_project/purl/output/domain_analysis/affiliate/records_sort_by_landing_page.csv")
    
    df_ads_url_sort, df_ads_landing_page_sort = sort_by_frequency(df_ads_all_records)
    df_ads_url_sort.to_csv("/home/data/chensun/affi_project/purl/output/domain_analysis/ads/records_sort_by_url.csv")
    df_ads_landing_page_sort.to_csv("/home/data/chensun/affi_project/purl/output/domain_analysis/ads/records_sort_by_landing_page.csv")
    

    #df_aff_all_records.to_csv("/home/data/chensun/affi_project/purl/output/domain_analysis/affiliate/records.csv")
    #df_ads_all_records.to_csv("/home/data/chensun/affi_project/purl/output/domain_analysis/ads/records.csv")

    



    
