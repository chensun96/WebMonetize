from urllib.parse import urlparse
import pandas as pd
import re
import os
import labelling_graph  

def process_crawl(df_redirect_chains):
    # Apply affiliate_rules to each URL and record the specific rule matched
    # record the match.group() in the new column
    def apply_rules(url):
        rules = labelling_graph.get_affiliate_rules()
        for rule in rules:
            matched, description = rule(url)
            if matched:
                return True, description
        return False, ''
    
    # Apply rules to each URL and store results

    df_redirect_chains['match_result'] = df_redirect_chains['url'].apply(apply_rules)
    df_redirect_chains['match'] = df_redirect_chains['match_result'].apply(lambda x: x[0])
    df_redirect_chains['rule_description'] = df_redirect_chains['match_result'].apply(lambda x: x[1])

    # Drop the temporary column and determine labels based on matches
    df_redirect_chains.drop(columns=['match_result'], inplace=True)
    
    # Map True/False to 'affiliate'/'others'
    df_redirect_chains['final_rules_based_label'] = df_redirect_chains.groupby('visit_id')['match'].transform(any).map({True: 'affiliate', False: 'others'})

    output_columns = ['visit_id', 'url', 'redirect_domain_total', 'match', 'rule_description', 'final_rules_based_label']
    df_rule_based_label = df_redirect_chains[output_columns]

    return df_rule_based_label
    

def apply_high_precision_rules(folder):
    for crawl_id in os.listdir(folder):
        each_crawl =  os.path.join(folder, crawl_id)

        # Check if this crawl complete, or still building
        #label_path = os.path.join(each_crawl, 'label.csv')
        #if not os.path.exists(label_path):
        #    print("This crawl is not complete. Ignore")
        #    continue

        rule_based_label_path = os.path.join(each_crawl, 'rule_based_label.csv')
        if not os.path.exists(rule_based_label_path):
            print("This crawl is still processing. Continue")
            continue
        
        if not crawl_id == "crawl_aff_normal_0":
            continue

        redirect_chain_path = os.path.join(each_crawl, 'redirect_chains.csv')
        print("\nProcessing : ",redirect_chain_path)
        df_redirect_chains = pd.read_csv(redirect_chain_path, on_bad_lines='skip')
        df_rule_based_label  = process_crawl(df_redirect_chains)   
        df_rule_based_label.to_csv(rule_based_label_path, index=False)


def seperate_data_based_on_rule_labels(folder, rule_based_aff_folder, rule_based_others_folder):
    os.makedirs(folder, exist_ok=True)
    other_files = ['features_phase1_simple.csv', 'features_phase1.csv', 'records.csv', 'redirect_chains.csv', 'url_features.csv', 'rule_based_label.csv']
    data_files = {}
    
    for crawl_id in os.listdir(folder):
        each_crawl = os.path.join(folder, crawl_id)
        
        if not crawl_id == "crawl_aff_normal_0":
            continue
        
        rule_based_label_path = os.path.join(each_crawl, 'rule_based_label.csv')
        if not os.path.exists(rule_based_label_path):
            print(f"This crawl {crawl_id} is not complete or rule_based_label.csv is missing. Ignoring.")
            continue

        df_labels = pd.read_csv(rule_based_label_path)
        if 'final_rules_based_label' not in df_labels.columns:
            print(f"Missing 'final_rules_based_label' in the data for crawl {crawl_id}. Ignoring.")
            continue

        # Pre-load other files into memory
        for file_name in other_files:
            file_path = os.path.join(each_crawl, file_name)
            if os.path.exists(file_path):
                data_files[file_name] = pd.read_csv(file_path)
        
        grouped = df_labels.groupby('visit_id')
        for visit_id, group in grouped:
            label = group['final_rules_based_label'].iloc[0]  # all rows in group have the same label
            
            if label == 'affiliate':
                destination_folder = os.path.join(rule_based_aff_folder, crawl_id)
            elif label == 'others':
                destination_folder = os.path.join(rule_based_others_folder, crawl_id)
            else:
                continue
            
            os.makedirs(destination_folder, exist_ok=True)
            
            # Handle files for this visit_id
            for file_name, df in data_files.items():
                df_filtered = df[df['visit_id'] == visit_id]
                # Check if any data exists for this visit_id
                if not df_filtered.empty:
                    other_destination_path = os.path.join(destination_folder, file_name)
                    df_filtered.to_csv(other_destination_path, index=False)  # Overwrite the original file


def delete_incomplete_affiliate_link(subfolder, rule_based_aff_folder):
    aff_network = ['rstyle.me', 'linksynergy.com', 'awin1.com', 'skimresources.com', 'howl.me'\
                   'shop-links.co', 'bam-x.com', 'narrativ.com', 'ztk5.net', 'narrativ.com']
    
    
    # Check if this crawl complete, or still building
    label_path = os.path.join(rule_based_aff_folder, subfolder, 'rule_based_label.csv')

    df_label = pd.read_csv(label_path, on_bad_lines='skip')
    # Step 1: Extract the last domain and add it as a new column
    df_label['last_domain'] = df_label['redirect_domain_total'].apply(lambda x: re.split(r' \|\| ', x)[-1].strip())

    # Step 2: Filter out rows where the last_domain is in the aff_network list
    df_filtered = df_label[~df_label['last_domain'].isin(aff_network)]

    df_filtered.to_csv(label_path, index=False)
    print(f"Updated the label file at {df_label} after removing incomplete affiliate links.")


if __name__ == "__main__":

    # step 1, apply high_precision_rules to all data
    affiliate_folder = "../../output/rule_based_aff_yt"
    others_folder = "../../output/rule_based_others_yt"

    apply_high_precision_rules(affiliate_folder)
    apply_high_precision_rules(others_folder)
    
    
    
    # step 2. Seperate data into two folders: "rule_based_aff", "rule_based_others" folder 
    rule_based_aff_folder = "../../output/rule_based_aff_yt"
    rule_based_others_folder = "../../output/rule_based_others_yt"
    seperate_data_based_on_rule_labels(affiliate_folder, rule_based_aff_folder, rule_based_others_folder)
    seperate_data_based_on_rule_labels(others_folder, rule_based_aff_folder, rule_based_others_folder)
  
      
      
