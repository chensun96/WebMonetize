import json
import pandas as pd
import networkx as nx
import numpy as np
import re
import networkx as nx
import os
import graph_scripts as gs
import matplotlib.pyplot as plt
import ast



# Filter relevant nodes and edges
def filter_data(df):
    # Filter nodes
    nodes = df[(df['graph_attr'] == 'Node') & (df['type'] == 'Document') & (df['is_in_phase1'])]
    # Filter edges
    edges = df[(df['graph_attr'] == 'Edge') & (df['is_in_phase1'])]
    # Ensure that both src and dst are in the valid nodes
    valid_nodes = set(nodes['name'])
    edges = edges[edges['src'].isin(valid_nodes) & edges['dst'].isin(valid_nodes)]
    return nodes, edges

# Construct network graph
def build_graph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes['name'])
    G.add_edges_from(edges[['src', 'dst']].itertuples(index=False, name=None))
    return G

def collect_redirect_chains(G, node_types):

    # find all the start nodes, no in-coming edges
    start_nodes = []
    chains = []
    for node in G.nodes:
        # print("\nnode: ", node)
        if G.in_degree(node) == 0:
            start_nodes.append(node)
        else:
            # Check if all incoming nodes are non-'Document' type
            all_non_document = True
            for pred in G.predecessors(node):
                # print("pred: ", pred)
                if node_types.get(pred, '') == 'Document':
                    if pred != node: # prevent self loop
                        all_non_document = False
                        break
            if all_non_document:
                start_nodes.append(node)


    for start in start_nodes:
        current = start
        chain = [current]
        visited = {current}
        while G.out_degree(current):
            successors = [node for node in G.successors(current) if node != current and node not in visited]
            if not successors:
                break
            next_node = successors[0]
            chain.append(next_node)
            visited.add(next_node)
            current = next_node
        
        # Generate a flat list of tuples for each URL in the chain
        for url in chain:
            chains.append((url, ' -> '.join(chain)))
    return chains
    


def process_by_visit_id(df):
    results = []
    grouped = df.groupby('visit_id')
    for visit_id, group in grouped:
        #print(f"\nProcessing network for visit_id: {visit_id}")
        nodes, edges = filter_data(group)
        if nodes.empty or edges.empty:
            print("No valid nodes or edges for this visit_id.")
            continue
        node_types = {row['name']: row['type'] for index, row in nodes.iterrows()}
        G = build_graph(nodes, edges)
        chains = collect_redirect_chains(G, node_types)
        for url, chain in chains:
            results.append({'visit_id': visit_id, 'url': url, 'redirect_chain': chain})

    return pd.DataFrame(results)

def extract_redirect_domains(redirect_chain):
    # Split the chain on the '->' and strip any leading/trailing whitespace
    urls = [url.strip() for url in redirect_chain.split('->')]
    domains = []
    for url in urls:
        domains.append(gs.get_domain(url)) 
    return ' || '.join(domains)

def process_redirect_domain(df):
    # add url domain
    df["url_domain"] = df["url"].apply(gs.get_domain)
    # redirect_domain_by_component
    df["redirect_domain_by_component"] = df["redirect_chain"].apply(extract_redirect_domains)
    # redirect_domain_total

    result_df = df.groupby('visit_id')['url_domain'].agg(lambda x: ' || '.join(x)).reset_index()
    result_df.columns = ['visit_id', 'redirect_domain_total']
    merged_df = pd.merge(df, result_df, on='visit_id')

    return merged_df


def extract_redirect_chains(output_folder):
    # Check if this crawl already build redirect chain
    redirect_chains_path = os.path.join(output_folder, 'redirect_chains.csv')
    if os.path.exists(redirect_chains_path):
        print("Already contain redirect chains. Ignore")


    # Step 1: Extract redirect url chain
    graph_file_path = os.path.join(output_folder, 'graph.csv')
    if os.path.exists(graph_file_path):
        df_graph = pd.read_csv(graph_file_path, on_bad_lines='skip')
        results = process_by_visit_id(df_graph)
        redirect_chains_df = pd.DataFrame(results)
        redirect_chains_df.to_csv(redirect_chains_path, index=False)
        print(f"Redirect chains processed for {graph_file_path}")

    
    # Step 2: Merge with parent_page_url
    records_file_path = os.path.join(output_folder, 'records.csv')
    if os.path.exists(records_file_path) and os.path.exists(redirect_chains_path):
        records_df = pd.read_csv(records_file_path)
        redirect_chains_df = pd.read_csv(redirect_chains_path)
        merged_df = pd.merge(redirect_chains_df, records_df[['visit_id', 'parent_page_url']], on='visit_id', how='left')
        merged_df.to_csv(redirect_chains_path, index=False)
        print(f"Merged data saved to {redirect_chains_path}")

    # Step 3: Extract domain from redirect chain
    if os.path.exists(redirect_chains_path):
        df = pd.read_csv(redirect_chains_path, on_bad_lines='skip')
        processed_df = process_redirect_domain(df)
        processed_df.to_csv(redirect_chains_path, index=False)
        print(f"Domain extraction complete for {redirect_chains_path}")
    return processed_df

    


def process_crawls(folder_path):
    for crawl_id in os.listdir(folder_path):
        each_crawl = os.path.join(folder_path, crawl_id)


        # Check if this crawl complete, or still building
        #label_path = os.path.join(each_crawl, 'label.csv')
        #if not os.path.exists(label_path):
        #    print("This crawl is not complete. Ignore")
        #    continue

        # Check if this crawl already build redirect chain
        redirect_chains_path = os.path.join(each_crawl, 'redirect_chains.csv')
        if os.path.exists(redirect_chains_path):
            print("Already contain redirect chains. Ignore")
            continue
        
        # Step 1: Extract redirect url chain
        graph_file_path = os.path.join(each_crawl, 'graph.csv')
        if os.path.exists(graph_file_path):
            df_graph = pd.read_csv(graph_file_path, on_bad_lines='skip')
            results = process_by_visit_id(df_graph)
            redirect_chains_df = pd.DataFrame(results)
            redirect_chains_df.to_csv(redirect_chains_path, index=False)
            print(f"Redirect chains processed for {graph_file_path}")

        
        # Step 2: Merge with parent_page_url
        records_file_path = os.path.join(each_crawl, 'records.csv')
        if os.path.exists(records_file_path) and os.path.exists(redirect_chains_path):
            records_df = pd.read_csv(records_file_path)
            redirect_chains_df = pd.read_csv(redirect_chains_path)
            merged_df = pd.merge(redirect_chains_df, records_df[['visit_id', 'parent_page_url']], on='visit_id', how='left')
            merged_df.to_csv(redirect_chains_path, index=False)
            print(f"Merged data saved to {redirect_chains_path}")

        # Step 3: Extract domain from redirect chain
        if os.path.exists(redirect_chains_path):
            df = pd.read_csv(redirect_chains_path, on_bad_lines='skip')
            processed_df = process_redirect_domain(df)
            processed_df.to_csv(redirect_chains_path, index=False)
            print(f"Domain extraction complete for {redirect_chains_path}")

        
        # step 4: correct the label file, make the label file only include phase 1 url
        redirect_chains_path = os.path.join(each_crawl, 'redirect_chains.csv')
        label_file_path = os.path.join(each_crawl, 'label.csv')
        df_redirect = pd.read_csv(redirect_chains_path, on_bad_lines='skip')
        features_to_keep = ['visit_id', 'url', 'redirect_domain_total'] 
        df_filtered = df_redirect.filter(items=features_to_keep)
        df_filtered['label'] = "others"
        df_label = df_filtered.rename(
        columns={
            "redirect_domain_total": "name",
        }
        )
        df_label.to_csv(label_file_path, index=False)
        print(f"processed for {label_file_path}")
        

       
def delete_incomplete_affiliate_link(affiliate_folder):
    aff_network = ['rstyle.me', 'linksynergy.com', 'awin1.com', 'skimresources.com', 'howl.me'\
                   'shop-links.co', 'bam-x.com', 'narrativ.com', 'ztk5.net', 'narrativ.com']
    for crawl_id in os.listdir(affiliate_folder):
        each_crawl = os.path.join(affiliate_folder, crawl_id)

        # Check if this crawl complete, or still building
        label_path = os.path.join(each_crawl, 'rule_based_label.csv')
        if not os.path.exists(label_path):
            print("This crawl is not complete. Ignore")
            continue
        
        #if not crawl_id == "crawl_aff_normal_15000":
        #    continue

        df_label = pd.read_csv(label_path, on_bad_lines='skip')
        # Step 1: Extract the last domain and add it as a new column
        df_label['last_domain'] = df_label['redirect_domain_total'].apply(lambda x: re.split(r' \|\| ', x)[-1].strip())

        # Step 2: Filter out rows where the last_domain is in the aff_network list
        df_filtered = df_label[~df_label['last_domain'].isin(aff_network)]
  
        df_filtered.to_csv(label_path, index=False)
        print(f"Updated the label file at {df_label} after removing incomplete affiliate links.")



if __name__ == "__main__":
    url_features_columns = []
    # fullGraph classification
    others_folder = "../output/others_yt"
    affiliate_folder = "../output/affiliate_yt"

    # step 1
    #process_crawls(affiliate_folder)
    #process_crawls(others_folder)

    # step 2
    rule_based_aff_folder = "../output/rule_based_aff_yt"
    delete_incomplete_affiliate_link(rule_based_aff_folder)
    
   
     
    



