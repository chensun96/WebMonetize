import os
import pandas as pd
from urllib.parse import urlparse
import tldextract

def get_domain(url):
    try:
        if (isinstance(url, list)):
            domains = []
            for u in url:
                u = tldextract.extract(u)
                domains.append(u.domain+"."+u.suffix)
            return domains
        else:
            u = tldextract.extract(url)
            return u.domain+"."+u.suffix
    except:
        return None

def process_folder(folder_path, processed_paths, result_dir):
    all_data = pd.DataFrame(columns=['url', 'file_path', 'visit_id', 'domain', 'parent_page_url'])

    for crawl_id in os.listdir(folder_path):
        each_crawl = os.path.join(folder_path, crawl_id)
        for filename in os.listdir(each_crawl):
            if filename == "redirect_chains.csv":
                file_path = os.path.join(each_crawl, filename)
                if file_path in processed_paths:
                    continue  # Skip if already processed
                print("Processing: ", file_path)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                for index, row in df.iterrows():
                    domain = get_domain(row['url'])
                    all_data = all_data.append({
                        'url': row['url'],
                        'file_path': file_path,
                        'visit_id': row['visit_id'],
                        'domain': domain,
                        'parent_page_url': row['parent_page_url'],
                        'redirect_domain_total': row['redirect_domain_total']
                    }, ignore_index=True)

    # Group by domain and update each CSV file with new data only
    for domain, group in all_data.groupby('domain'):
        domain_file = f"{domain}.csv"
        domain_path = os.path.join(result_dir, domain_file)
        if os.path.exists(domain_path):
            existing_df = pd.read_csv(domain_path)
            combined_df = pd.concat([existing_df, group])
            combined_df.drop_duplicates(subset=['file_path'], keep='last', inplace=True)
            combined_df.to_csv(domain_path, index=False)
        else:
            group.to_csv(domain_path, index=False)

    return all_data

def count_links_by_domains(affiliate_folder, non_affiliate_folder, affiliate_result_dir, non_affiliate_result_dir, processed_file_path):
    
    
    if os.path.exists(processed_file_path):
        processed_df = pd.read_csv(processed_file_path)
        processed_paths = set(processed_df['file_path'].unique())
    else:
        processed_paths = set()

    # Process each folder and maintain unique paths
    all_aff_data = process_folder(affiliate_folder, processed_paths, affiliate_result_dir)
    all_non_aff_data = process_folder(non_affiliate_folder, processed_paths, non_affiliate_result_dir)

    # Update the processed file with new file paths
    new_paths = pd.concat([all_aff_data, all_non_aff_data])['file_path'].drop_duplicates().to_frame()
    if os.path.exists(processed_file_path):
        existing_paths = pd.read_csv(processed_file_path)
        updated_paths = pd.concat([existing_paths, new_paths]).drop_duplicates()
        updated_paths.to_csv(processed_file_path, index=False)
    else:
        new_paths.to_csv(processed_file_path, index=False)

    print("Updated CSV files for each domain in respective directories.")



def dedup_links_by_domains(affiliate_result_dir, non_affiliate_result_dir):
    for result_dir in [affiliate_result_dir, non_affiliate_result_dir]:

        for each_domain in os.listdir(result_dir):
            #if each_domain != "1.envato.market.csv":
            #    continue
            if each_domain.endswith("csv"):
                each_crawl = os.path.join(result_dir, each_domain)
                print(each_crawl)
                df = pd.read_csv(each_crawl)
                deduplicated_df = df.drop_duplicates(subset=['parent_page_url', 'url'])
                deduplicated_df.to_csv(each_crawl, index=False)
        


if __name__ == "__main__":

   
    #affiliate_folder = "../../output/affiliate_yt"
    #non_affiliate_folder = "../../output/others_yt"
    affiliate_folder = "../../output/rule_based_aff_yt"
    non_affiliate_folder = "../../output/rule_based_others_yt"

    processed_file_path = "../../output/domain_analysis/processed_each_domain_record.csv"
    
    affiliate_result_dir = "../../output/domain_analysis/aff_each_domain"
    non_affiliate_result_dir = "../../output/domain_analysis/non_aff_each_domain"
    for result_dir in [affiliate_result_dir, non_affiliate_result_dir]:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    
    count_links_by_domains(affiliate_folder, non_affiliate_folder, affiliate_result_dir, non_affiliate_result_dir, processed_file_path)
    #process_folder(affiliate_folder, processed_file_path, affiliate_result_dir)
    #process_folder(non_affiliate_folder, processed_file_path, non_affiliate_result_dir)

    dedup_links_by_domains(affiliate_result_dir, non_affiliate_result_dir)



    # see analysis_domain.ipynb for finding "all unique redirect chains" (focus on domain)
    # e.g., unique_aff_redirect_domain_chains.csv