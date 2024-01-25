import graph_scripts as gs
import labelling_scripts as ls
from tqdm import tqdm
import pandas as pd
import gc
import re
from yaml import full_load
import feature_scripts_cookies as fs
import graph_scripts as gs
from checking_affiliate import apply_affiliate_rule
from networkx.readwrite import json_graph
import json
# import leveldb
import argparse
import multiprocessing as mp
import numpy as np
import timeit
from resource import getrusage, RUSAGE_SELF
import traceback
import time
import os
import tldextract

import networkx as nx
import matplotlib.pyplot as plt


def modify_type(orig_type):
    """
    Function to process type of a node in a better format.

    Args:
        orig_attr: Original type.
    Returns:
        new_attr: New type.

    This functions does the following:

    1. Processes the original type.
    """
    orig_type = list(set(orig_type))
    if len(orig_type) == 1:
      return orig_type[0]
    else:
      new_type = "Request"
      if "Script" in orig_type:
        new_type = "Script"
      elif "Document" in orig_type:
        new_type = "Document"
      elif "Element" in orig_type:
        new_type = "Element"
      return new_type

def modify_attr(orig_attr):
    """
    Function to process attributes of a node in a better format.

    Args:
        orig_attr: Original attribute.
    Returns:
        new_attr: New attribute.

    This functions does the following:

    1. Processes the original attribute.
    """
    new_attr = {}
    orig_attr = np.array(list(set(orig_attr)))
    if 'Cookie' in orig_attr:
        return 'Cookie'
    if 'HTTPCookie' in orig_attr:
        return 'HTTPCookie'

    for item in orig_attr:
        try:
            d = json.loads(item)
            new_attr.update(d)
        except:
            continue
    return json.dumps(new_attr)

def build_graph(pdf):

    """
    Function to build a networkX graph from a Pandas DataFrame.

    Args:
        pdf: DataFrame of nodes and edges.
    Returns:
        G: networkX graph.

    This functions does the following:

    1. Selects nodes and edges.
    2. Processes node attributes.
    3. Creates graph from edges.
    4. Updates node attributes in graph.
    """

    df_nodes = pdf[(pdf["graph_attr"] == "Node") | (pdf["graph_attr"] == "NodeWG")]
    df_edges = pdf[(pdf["graph_attr"] == "Edge") | (pdf["graph_attr"] == "EdgeWG")]
    df_nodes = df_nodes.groupby(['visit_id', 'name'], \
        as_index=False).agg({'type': lambda x: list(x), \
        'attr': lambda x: list(x), 'domain' : lambda x: list(x)[0], \
        'top_level_domain' : lambda x: list(x)[0]})

    df_nodes['type'] = df_nodes['type'].apply(modify_type)
    df_nodes['attr'] = df_nodes['attr'].apply(modify_attr)
    G = nx.from_pandas_edgelist(df_edges, source='src', target='dst', edge_attr=True, create_using=nx.DiGraph())
    node_dict = df_nodes.set_index('name').to_dict("index")
    nx.set_node_attributes(G, node_dict)
    return G


def create_graph(df):
    """Function to build a graph on each visit_id/site. Complete as required.
    :param df: pandas dataframe of nodes and edges.
    :return: graph object.
    :rtype: Graph
    """
    G = build_graph(df)
    df.to_csv()
    
    # try to visualize the graph
    # plt.figure(figsize=(12, 12))  # Set the size of the figure
    # pos = nx.spring_layout(G)  # Positions for all nodes using one of the layout algorithms
    # nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", linewidths=0.25, font_size=10, font_weight="bold", edge_color="gray")
    # plt.title("Graph Visualization")
    # plt.savefig('/home/ubuntu/purl/graph/graph.png')

    tqdm.write(f"Built graph of {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

def load_features_info(filename):
    """Load features from features.yaml file
    :param filename: yaml file name containing feature names
    :return: list of features to use.
    """
    with open(filename) as file:
        return full_load(file)


def has_ad_keywords(url):
    keyword_raw = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect",
               "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban",
               "delivery", "promo", "tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc", "google_afs"]

    for key in keyword_raw:
        if re.search(key, url, re.I):
            print("\n URL:", url)
            print("Found a match ads keyword: ", key)
            return True
    return False

def extract_url_features(G): # only apply to phaseA
    nodes = G.nodes(data=True)
    has_affiliate_keyword = 0
    number_affiliate_keyword = 0
    #has_ads_keyword = 0
    #number_ads_keyword = 0
    for node in nodes:
    
        if ("type" in node[1]):
            if (node[1]["type"] == "Document" or node[1]["type"] == "Request" or node[1]["type"] == "Script"):
                # print("\nnode[0]: ", node[0])
                # apply check affiliate function
                if apply_affiliate_rule(node[0]):
                    #print("\nnode[0]: ", node[0])
                    #print("This is an affiliate url")
                    has_affiliate_keyword = 1
                    number_affiliate_keyword = number_affiliate_keyword + 1
                #if has_ad_keywords(node[0]):
                #    has_ads_keyword = 1
                #    number_ads_keyword = number_ads_keyword + 1

    url_feature = [has_affiliate_keyword, number_affiliate_keyword]
    url_feature_name =  ['has_affiliate_keyword', 'number_affiliate_keyword']
   
    return url_feature, url_feature_name


def extract_graph_node_features(
    G,
    df_graph,
    G_indirect,
    df_indirect_graph,
    G_indirect_all,
    node,
    ldb,
    selected_features,
    vid,
    node_type,
):
    all_features = []
    all_feature_names = ["visit_id", "name", "node_type"]
    content_features = []
    structure_features = []
    dataflow_features = []
    additional_features = []
    content_feature_names = []
    structure_feature_names = []
    dataflow_feature_names = []
    additional_feature_names = []

    # calcualte time it takes to get each content feature
    """
    start = time.time()
    if "content" in selected_features:
        content_features, content_feature_names = fs.get_content_features(
            G, df_graph, node
        )
    end = time.time()
    # print("Time to get content features:", end - start)
    """
    start = time.time()
    if "structure" in selected_features:
        structure_features, structure_feature_names = fs.get_structure_features(
            G, df_graph, node, ldb
        )
    end = time.time()
    print("Time to get structure features:", end - start)

    print("structure_features: ", structure_features)
    print("structure_feature_names: ", structure_feature_names)
    
    """
    start = time.time()
    if "dataflow" in selected_features:
        dataflow_features, dataflow_feature_names = fs.get_dataflow_features(
            G, df_graph, node, G_indirect, G_indirect_all, df_indirect_graph
        )
    end = time.time()
    # print("Time to get dataflow features:", end - start)

    start = time.time()
    if "additional" in selected_features:
        additional_features, additional_feature_names = fs.get_additional_features(
            G, df_graph, node
        )
    end = time.time()
    # print("Time to get additional features:", end - start)
    """
    all_features = (
        content_features + structure_features + dataflow_features + additional_features
    )
    all_feature_names += (
        content_feature_names
        + structure_feature_names
        + dataflow_feature_names
        + additional_feature_names
    )

    df = pd.DataFrame([[vid] + [node] + [node_type] + all_features], columns=all_feature_names)

    return df


def extract_graph_features(df_graph, G, vid, ldb, feature_config, tag):
    """
    Function to extract features.

    Args:
      df_graph_vid: DataFrame of nodes/edges for.a site
      G: networkX graph of site
      vid: Visit ID
      ldb: Content LDB
      feature_config: Feature config
    Returns:
      df_features: DataFrame of features for each URL in the graph

    This functions does the following:

    1. Reads the feature config to see which features we want.
    2. Creates a graph of indirect edges if we want to calculate dataflow features.
    3. Performs feature extraction based on the feature config. Feature extraction is per node of graph.
    """

    exfil_columns = [
        "visit_id",
        "src",
        "dst",
        "dst_domain",
        "attr",
        "time_stamp",
        "direction",
        "type",
    ]

    df_features = []
    nodes = G.nodes(data=True)

    G_indirect = nx.DiGraph()
    G_indirect_all = nx.DiGraph()
    df_indirect_graph = pd.DataFrame()

    df_graph["src_domain"] = df_graph["src"].apply(fs.get_domain)
    df_graph["dst_domain"] = df_graph["dst"].apply(fs.get_domain)

    selected_features = feature_config["features_to_extract"]

    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for node in nodes:
            if (node[1]["type"] == "Document" or node[1]["type"] == "Request" or node[1]["type"] == "Script"):
                result = pool.apply_async(
                    extract_graph_node_features,
                    args=(
                        G,
                        df_graph,
                        G_indirect,
                        df_indirect_graph,
                        G_indirect_all,
                        node[0],
                        ldb,
                        selected_features,
                        vid,
                        node[1]["type"],
                    ),
                )
                results.append(result)

        for result in tqdm(results):
            df_features.append(result.get())

        df_features = pd.concat(df_features)
        #df_features.to_csv("/home/data/chensun/affi_project/purl/output/affiliate/fullGraph/df_features_est.csv")
    return df_features
    
        
def get_features(pdf, G, visit_id, features_file, ldb_file, tag):
    """Getter to generate the features of each node in a graph.
    :param pdf: pandas df of nodes and edges in a graph.
    :param G: Graph object representation of the pdf.
    :return: dataframe of features per node in the graph
    """
    # Generate features for each node in our graph
    feature_config = load_features_info(features_file)
    # ldb = leveldb.LevelDB(ldb_file)
    # ldb = plyvel.DB(ldb_file)
    ldb = None

    df_features = extract_graph_features(pdf, G, visit_id, ldb, feature_config, tag)
    print("here")
    return df_features
    

def apply_tasks(
    df,
    visit_id,
    ldb_file,
    graph_columns,
    graph_feature_columns,
    graph_feature_columns_simple,
    graph_path
):
    try:

        df_top_domain = gs.extract_url_domain(df)
        df_top_domain = df_top_domain.drop_duplicates(subset=['visit_id', 'top_level_url']) 

        # get phase1 level and graph level features
        phase1_df = df[df['is_in_phase1'] == True]
        dataframes = {'phase1': phase1_df, 'fullGraph': df}
        
        for key, dataframe in dataframes.items():
            all_feature_names = ["visit_id"]
            start = time.time()
            # print(df.columns)

            #print(f"get the {i} level features")
            G = create_graph(dataframe)
            node_feature, node_feature_name, connectivity_feature, connectivity_feature_names, simpler_feature, simpler_feature_name = gs.extract_full_graph_features(G, visit_id, key, graph_path)
            
            #df.to_csv("/home/data/chensun/affi_project/purl/output/affiliate/fullGraph/df_features_1.csv")
            # crate graph for this single df (visit id)
            #G = create_graph(df) # fullGraph

            #df_features = get_features(df, G, visit_id, features_file, ldb_file, tag)
            #url_feature, url_feature_name = extract_url_features(phase1_G)

            all_features = (
            node_feature + connectivity_feature 
            )
            all_feature_names += (
                node_feature_name
                + connectivity_feature_names
            )

            df_features = pd.DataFrame([[visit_id] + all_features], columns=all_feature_names)
            #print("df_features: ", df_features.columns)

            df_features = df_features.merge(df_top_domain, on='visit_id', how='left')
            #print("df_features", df_features.columns)
            

            # get the graph simpler features
            simpler_feature_names = ["visit_id"] + simpler_feature_name
            df_features_simpler = pd.DataFrame([[visit_id] + simpler_feature] , columns= simpler_feature_names)
            df_features_simpler = df_features_simpler.merge(df_top_domain, on='visit_id', how='left')
           
            if key == 'phase1':
                df_features_path = graph_path.replace('graph', 'features_phase1')
                df_features_simpler_path = graph_path.replace('graph', 'features_phase1_simple')
                print("df_features_simpler_phase1_path: ", df_features_simpler_path)
                #df_features_path = "/home/data/chensun/affi_project/purl/output/ads/fullGraph/graph_level_features_phase1.csv"
            else:
                df_features_path = graph_path.replace('graph', 'features_fullGraph')
                df_features_simpler_path = graph_path.replace('graph', 'features_fullGraph_simple')
                print("df_features_simpler_fullGraph_path: ", df_features_simpler_path)

            if not os.path.exists(df_features_path):
                df_features.reindex(columns=graph_feature_columns).to_csv(df_features_path)
            else:
                df_features.reindex(columns=graph_feature_columns).to_csv(
                    df_features_path, mode="a", header=False
            )    

            if not os.path.exists(df_features_simpler_path):
                df_features_simpler.reindex(columns=graph_feature_columns_simple).to_csv(df_features_simpler_path)
            else:
                df_features_simpler.reindex(columns=graph_feature_columns_simple).to_csv(
                    df_features_simpler_path, mode="a", header=False
            )
            end = time.time()
            print("Extracted features:", end - start)
            
    except Exception as e:
        print("Errored in pipeline:", e)
        traceback.print_exc()


def pipeline(df, visit_id, graph_columns, graph_path):
   
    print(f"Building graph lebel features in {graph_path}")
    
    graph_feature_columns = [
        "visit_id",
        "name",
        "top_level_url",
        "num_nodes",
        "num_edges",
        "average_degree",
        "average_in_degree",
        "average_out_degree",
        "median_in_degree",
        "median_out_degree",
        "max_in_degree", 
        "max_out_degree", 
        "density",
        "avg_clustering_coefficient",
        "transitivity",
        "number_of_ccs",
        "average_size_cc",
        "largest_cc",
        "smallest_cc",
        "max_avg_path_length",
        "average_degree_centrality",
        "median_degree_centrality",
        "max_degree_centrality",
        "min_degree_centrality",
        "std_dev_degree_centrality",
        "average_closeness_centrality",
        "median_closeness_centrality",
        "max_closeness_centrality",
        "min_closeness_centrality",
        "std_dev_closeness_centrality",
        "average_closeness_centrality_outward", 
        "median_closeness_centrality_outward",
        "max_closeness_centrality_outward", 
        "min_closeness_centrality_outward", 
        "std_dev_closeness_centrality_outward",
        "average_path_length_for_largest_cc"
    ]

    graph_feature_columns_simple = [
        "visit_id",
        "name",
        "top_level_url",
        "num_nodes", 
        "num_edges", 
        "max_in_degree", 
        "max_out_degree", 
        "nodes_div_by_edges", 
        "edges_div_by_nodes", 
        "density", 
        "largest_cc", 
        "number_of_ccs", 
        "transitivity", 
        "average_path_length_for_largest_cc"
    ]

    df.groupby(["visit_id"]).apply(
                    apply_tasks,
                    visit_id,
                    None,
                    graph_columns,
                    graph_feature_columns,
                    graph_feature_columns_simple,
                    graph_path
    )

            
if __name__ == "__main__":
    graph_columns = [
        "visit_id",
        "name",
        "top_level_url",
        "type",
        "attr",
        "domain",
        "document_url",
        "setter",
        "setting_time_stamp",
        "top_level_domain",
        "setter_domain",
        "graph_attr",
        "party",
        "src",
        "dst",
        "action",
        "time_stamp",
        "reqattr",
        "respattr",
        "response_status",
        "content_hash",
        "post_body",
        "post_body_raw",
        "is_in_phase1"
    ]

    """
    
    url_type = ["affiliate"]
    #url_type = ['ads']
    for type_name in url_type:

        full_graph_folder = f"../output/{type_name}/fullGraph"
        #phaseA_folder = f"../output/{type_name}/phaseA"

        for filename in os.listdir(full_graph_folder):
            if filename.startswith("graph_") and filename.endswith(".csv"):
                # get the graph_{i}.csv 
                fullGraph_path = os.path.join(full_graph_folder, filename)
                df_Graph = pd.read_csv(fullGraph_path)
                visit_ids = df_Graph['visit_id'].unique()
                #print(visit_ids)
                for visit_id in visit_ids:
                    df = df_Graph[df_Graph['visit_id'] == visit_id]
                    pipeline(df, visit_id, graph_columns, fullGraph_path)
                    
    """
    fullGraph_path = "/home/data/chensun/affi_project/purl/output/graph_0.csv"
    df_Graph = pd.read_csv(fullGraph_path)
    visit_ids = df_Graph['visit_id'].unique()
    #print(visit_ids)
    for visit_id in visit_ids:
        df = df_Graph[df_Graph['visit_id'] == visit_id]
        pipeline(df, visit_id, graph_columns, fullGraph_path)

