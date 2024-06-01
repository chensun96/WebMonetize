import graph_scripts as gs
import labelling_scripts as ls
from tqdm import tqdm
import pandas as pd
import gc
from yaml import full_load

#from checking_affiliate import check_affiliate_link
from feature_extraction import extract_graph_features
from feature_extraction import extract_graph_features_phase1
from networkx.readwrite import json_graph
import graph_approach_features
import json
# import leveldb
import argparse

import timeit
from resource import getrusage, RUSAGE_SELF
import traceback
import time
import os
import tldextract

import networkx as nx
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None, "display.max_columns", None)


def read_sites_visit(db_file, conn):
    """Read the list of sites visited by crawler and their information
    :return: pandas df of site_visits table in scraper SQL file.
    """
    # conn = gs.get_local_db_connection(db_file)
    # Return a dataframe of the sites visited (stored in sites_visits of scraper SQL)
    return gs.get_sites_visit(conn)


def create_graph(df):
    """Function to build a graph on each visit_id/site. Complete as required.
    :param df: pandas dataframe of nodes and edges.
    :return: graph object.
    :rtype: Graph
    """
    G = gs.build_graph(df)
    
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


def get_features(pdf, G, features_file):
    """Getter to generate the features of each node in a graph.
    :param pdf: pandas df of nodes and edges in a graph.
    :param G: Graph object representation of the pdf.
    :return: dataframe of features per node in the graph
    """
    # Generate features for each node in our graph
    feature_config = load_features_info(features_file)
    df_features = extract_graph_features(pdf, G, pdf.visit_id[0], None, feature_config)
    return df_features


def get_features_phase1(pdf, G, features_file):
    """Getter to generate the features of each node in a graph.
    :param pdf: pandas df of nodes and edges in a graph.
    :param G: Graph object representation of the pdf.
    :return: dataframe of features per node in the graph
    """
    # Generate features for each node in our graph
    feature_config = load_features_info(features_file)
    df_features = extract_graph_features_phase1(pdf, G, pdf.visit_id[0], None, feature_config)
    return df_features

def find_setter_domain(setter):
    try:
        domain = gs.get_domain(setter)
        return domain
    except:
        return None


def find_domain(row):
    domain = None

    try:
        node_type = row["type"]
        if (
            (node_type == "Document")
            or (node_type == "Request")
            or (node_type == "Script")
        ):
            domain = gs.get_domain(row["name"])
        elif node_type == "Element":
            return domain
        else:
            return row["domain"]
        return domain
    except Exception as e:
        traceback.print_exc()
        return None


def find_tld(top_level_url):
    try:
        if top_level_url:
            tld = gs.get_domain(top_level_url)
            return tld
        else:
            return None
    except:
        return None


def update_storage_names(row):
    name = row["name"]
    try:
        if row["type"] == "Storage":
            name = name + "|$$|" + row["domain"]
    except Exception as e:
        return name
    return name


def find_setters(
    df_all_storage_nodes,
    df_http_cookie_nodes,
    df_all_storage_edges,
    df_http_cookie_edges,
):
    df_setter_nodes = pd.DataFrame(
        columns=[
            "visit_id",
            "name",
            "type",
            "attr",
            "top_level_url",
            "domain",
            "setter",
            "setting_time_stamp",
        ]
    )

    try:
        df_storage_edges = pd.concat([df_all_storage_edges, df_http_cookie_edges])
        if len(df_storage_edges) > 0:
            df_storage_sets = df_storage_edges[
                (df_storage_edges["action"] == "set")
                | (df_storage_edges["action"] == "set_js")
            ]
            df_setters = gs.get_original_cookie_setters(df_storage_sets)
            df_storage_nodes = pd.concat([df_all_storage_nodes, df_http_cookie_nodes])
            df_setter_nodes = df_storage_nodes.merge(
                df_setters, on=["visit_id", "name"], how="outer"
            )

    except Exception as e:
        print("Error getting setter:", e)
        traceback.print_exc()

    return df_setter_nodes


def get_party(row):
    if row["type"] == "Storage":
        if row["domain"] and row["top_level_domain"]:
            if row["domain"] == row["top_level_domain"]:
                return "first"
            else:
                return "third"
    return "N/A"


def read_sql_crawl_data(visit_id, db_file, conn):
    """Read SQL data from crawler for a given visit_ID.
    :param visit_id: visit ID of a crawl URL.
    :return: Parsed information (nodes and edges) in pandas df.
    """

    #Directory where CSV files will be saved
    #dir_name = f"../graph_data/visit_data_{visit_id}/"
    #os.makedirs(dir_name, exist_ok=True)

    # Function to save DataFrame as CSV in the directory
    #def save_df(df, filename):
    #    df.to_csv(os.path.join(dir_name, filename + ".csv"), index=False)

    # Read tables from DB and store as DataFrames
    df_requests, df_responses, df_redirects, call_stacks, javascript = gs.read_tables(
        conn, visit_id
    )

    # Read only redirect phase data
    df_requests_phase1, df_responses_phase1, df_redirects_phase1, call_stacks_phase1, javascript_phase1 = gs.read_tables_phase1(
        conn, visit_id
    )

    # Create a column "is_in_phase1"
    # Mark or differentiate the data in df_http_requests_phase1 from df_http_requests
    df_requests, df_responses, df_redirects, call_stacks, javascript = gs.add_marker_column(
        df_requests, df_responses, df_redirects, call_stacks, javascript,
        df_requests_phase1, df_responses_phase1, df_redirects_phase1, call_stacks_phase1, javascript_phase1
    )
    
    #save_df(df_requests, "df_requests")
    #save_df(df_responses, "df_responses")
    #save_df(df_redirects, "df_redirects")
    #save_df(call_stacks, "call_stacks")
    #save_df(javascript, "javascript")
    

    try:
        df_js_nodes, df_js_edges = gs.build_html_components(javascript)
        if df_js_nodes.empty:
            raise ValueError("Invalid data type due to empty javascript table.")
        #else:
            #save_df(df_js_nodes, "df_js_nodes")
            #save_df(df_js_edges, "df_js_edges")
    except ValueError as e:
        print(e)
        df_all_graph = pd.DataFrame()
        return df_all_graph
    

    try:
        df_request_nodes, df_request_edges = gs.build_request_components(
            df_requests, df_responses, df_redirects, call_stacks
        )
        if df_request_edges.empty:
            raise ValueError("df_request_edges returned from get_cs_edges is empty.")
        else:
            #save_df(df_request_nodes, "df_request_nodes")
            #save_df(df_request_edges, "df_request_edges")

            df_decoration_nodes, df_decoration_edges = gs.build_decoration_components(
                df_request_nodes
            )

            #save_df(df_decoration_nodes, "df_decoration_nodes")
            #save_df(df_decoration_edges, "df_decoration_edges")

            df_all_storage_nodes, df_all_storage_edges = gs.build_storage_components(javascript)

            #save_df(df_all_storage_nodes, "df_all_storage_nodes")
            #save_df(df_all_storage_edges, "df_all_storage_edges")

            df_http_cookie_nodes, df_http_cookie_edges = gs.build_http_cookie_components(
                    df_request_edges, df_request_nodes
            )

            #save_df(df_http_cookie_nodes, "df_http_cookie_nodes")
            #save_df(df_http_cookie_edges, "df_http_cookie_edges")

            df_storage_node_setter = find_setters(
                df_all_storage_nodes,
                df_http_cookie_nodes,
                df_all_storage_edges,
                df_http_cookie_edges,
            )

            #save_df(df_storage_node_setter, "df_storage_node_setter")

            # Concatenate to get all nodes and edges
            df_request_nodes["domain"] = None
            df_decoration_nodes["domain"] = None


            df_all_nodes = pd.concat(
                [df_js_nodes, df_request_nodes, df_storage_node_setter, df_decoration_nodes]
            )
            df_all_nodes["domain"] = df_all_nodes.apply(find_domain, axis=1)
            df_all_nodes["top_level_domain"] = df_all_nodes["top_level_url"].apply(find_tld)
            df_all_nodes["setter_domain"] = df_all_nodes["setter"].apply(find_setter_domain)
            # df_all_nodes['name'] = df_all_nodes.apply(update_storage_names, axis=1)
            df_all_nodes = df_all_nodes.drop_duplicates()
            df_all_nodes["graph_attr"] = "Node"

            df_all_edges = pd.concat(
                [
                    df_js_edges,
                    df_request_edges,
                    df_all_storage_edges,
                    df_http_cookie_edges,
                    df_decoration_edges,
                ]
            )
            df_all_edges = df_all_edges.drop_duplicates()
            df_all_edges["top_level_domain"] = df_all_edges["top_level_url"].apply(find_tld)
            df_all_edges["graph_attr"] = "Edge"

            df_all_graph = pd.concat([df_all_nodes, df_all_edges])
            df_all_graph = df_all_graph.astype(
                {"type": "category", "response_status": "category"}
            )

            #save_df(df_all_nodes, "df_all_nodes")
            #save_df(df_all_edges, "df_all_edges")
            #save_df(df_all_graph, "df_all_graph")

    except ValueError as e:
        print(e)
        df_all_graph = pd.DataFrame()

    return df_all_graph


def read_sql_crawl_data_for_ads(visit_id, db_file, conn, tab_id):
    """Read SQL data from crawler for a given visit_ID.
    :param visit_id: visit ID of a crawl URL.
    :return: Parsed information (nodes and edges) in pandas df.
    """

    #Directory where CSV files will be saved
    #dir_name = f"../graph_data/visit_data_{visit_id}/"
    #os.makedirs(dir_name, exist_ok=True)

    # Function to save DataFrame as CSV in the directory
    #def save_df(df, filename):
    #    df.to_csv(os.path.join(dir_name, filename + ".csv"), index=False)

    # Read tables from DB and store as DataFrames
    df_requests, df_responses, df_redirects, call_stacks, javascript, max_request_id, min_request_id = gs.read_tables_for_ads(
        conn, visit_id, tab_id
    )
    #print("df_requests visit_id: ", df_requests['visit_id'].dtype)
    #print("df_responses visit_id: ", df_responses['visit_id'].dtype)
    #print("df_redirects visit_id: ", df_redirects['visit_id'].dtype)
    #print("call_stacks visit_id: ", call_stacks['visit_id'].dtype)
    
    #print("df_requests request_id: ", df_requests['request_id'].dtype)
    #print("df_responses request_id: ", df_responses['request_id'].dtype)
    #print("df_redirects visit_id: ", df_redirects['old_request_id'].dtype)
    #print("call_stacks request_id: ", call_stacks['request_id'].dtype)
    #print("javascript time stamp: ", javascript['time_stamp'].dtype)
    
    df_requests['request_id'] = df_requests['request_id'].astype('int64')
    df_responses['request_id'] = df_responses['request_id'].astype('int64')
    df_redirects['old_request_id'] = df_redirects['old_request_id'].astype('int64')
    call_stacks['visit_id'] = call_stacks['visit_id'].astype('int64')
    call_stacks['request_id'] = call_stacks['request_id'].astype('int64')
    javascript['time_stamp'] = pd.to_datetime(javascript['time_stamp'], errors='coerce')
    

    # Read only redirect phase data
    df_requests_phase1, df_responses_phase1, df_redirects_phase1, call_stacks_phase1, javascript_phase1 = gs.read_tables_phase1_for_ads(
        conn, visit_id, tab_id,  max_request_id, min_request_id
    )

    #print("df_requests_phase1 visit_id: ", df_requests_phase1['visit_id'].dtype)
    #print("df_responses_phase1 visit_id: ", df_responses_phase1['visit_id'].dtype)
    #print("df_redirects_phase1 visit_id: ", df_redirects_phase1['visit_id'].dtype)
    #print("call_stacks_phase1 visit_id: ", call_stacks_phase1['visit_id'].dtype)
    
    #print("df_requests_phase1 request_id: ", df_requests_phase1['request_id'].dtype)
    #print("df_responses_phase1 request_id: ", df_responses_phase1['request_id'].dtype)
    #print("df_redirects_phase1 visit_id: ", df_redirects_phase1['old_request_id'].dtype)
    #print("call_stacks_phase1 request_id: ", call_stacks_phase1['request_id'].dtype)
    #print("len of df_requests_phase1: ", len(df_requests_phase1))    
    #print("len of df_responses_phase1: ", len(df_responses_phase1))    
    #print("len of df_redirects_phase1: ", len(df_redirects_phase1))    
    #print("len of call_stacks_phase1: ", len(call_stacks_phase1))    
    #print("len of javascript_phase1: ", len(javascript_phase1))    

    #print("javascript_phase1 time stamp: ", javascript_phase1['time_stamp'].dtype)

    df_requests_phase1['request_id'] = df_requests_phase1['request_id'].astype('int64')
    df_responses_phase1['request_id'] = df_responses_phase1['request_id'].astype('int64')
    df_redirects_phase1['old_request_id'] = df_redirects_phase1['old_request_id'].astype('int64')
    call_stacks_phase1['visit_id'] = call_stacks_phase1['visit_id'].astype('int64')
    call_stacks_phase1['request_id'] = call_stacks_phase1['request_id'].astype('int64')
    javascript_phase1['time_stamp'] = pd.to_datetime(javascript_phase1['time_stamp'], errors='coerce')

   

    # Create a column "is_in_phase1"
    # Mark or differentiate the data in df_http_requests_phase1 from df_http_requests
    df_requests, df_responses, df_redirects, call_stacks, javascript = gs.add_marker_column(
        df_requests, df_responses, df_redirects, call_stacks, javascript,
        df_requests_phase1, df_responses_phase1, df_redirects_phase1, call_stacks_phase1, javascript_phase1
    )

    javascript['time_stamp'] = javascript['time_stamp'].astype(str)
  

    #print("df_requests visit_id: ", df_requests['visit_id'].dtype)
    #print("df_responses visit_id: ", df_responses['visit_id'].dtype)
    #print("df_redirects visit_id: ", df_redirects['visit_id'].dtype)
    #print("call_stacks visit_id: ", call_stacks['visit_id'].dtype)
    
    #print("df_requests request_id: ", df_requests['request_id'].dtype)
    #print("df_responses request_id: ", df_responses['request_id'].dtype)
    #print("df_redirects visit_id: ", df_redirects['old_request_id'].dtype)
    #print("call_stacks request_id: ", call_stacks['request_id'].dtype)
    #print("javascript time stamp: ", javascript['time_stamp'].dtype)
    
    

    #save_df(df_requests, "df_requests")
    #save_df(df_responses, "df_responses")
    #save_df(df_redirects, "df_redirects")
    #save_df(call_stacks, "call_stacks")
    #save_df(javascript, "javascript")
    

    try:
        df_js_nodes, df_js_edges = gs.build_html_components(javascript)
        if df_js_nodes.empty:
            raise ValueError("Invalid data type due to empty javascript table.")
        #else:
            #save_df(df_js_nodes, "df_js_nodes")
            #save_df(df_js_edges, "df_js_edges")
    except ValueError as e:
        print(e)
        df_all_graph = pd.DataFrame()
        return df_all_graph
    

    try:
        df_request_nodes, df_request_edges = gs.build_request_components(
            df_requests, df_responses, df_redirects, call_stacks
        )
        if df_request_edges.empty:
            raise ValueError("df_request_edges returned from get_cs_edges is empty.")
        else:
            #save_df(df_request_nodes, "df_request_nodes")
            #save_df(df_request_edges, "df_request_edges")

            df_decoration_nodes, df_decoration_edges = gs.build_decoration_components(
                df_request_nodes
            )

            #save_df(df_decoration_nodes, "df_decoration_nodes")
            #save_df(df_decoration_edges, "df_decoration_edges")

            df_all_storage_nodes, df_all_storage_edges = gs.build_storage_components(javascript)

            #save_df(df_all_storage_nodes, "df_all_storage_nodes")
            #save_df(df_all_storage_edges, "df_all_storage_edges")

            df_http_cookie_nodes, df_http_cookie_edges = gs.build_http_cookie_components(
                    df_request_edges, df_request_nodes
            )

            #save_df(df_http_cookie_nodes, "df_http_cookie_nodes")
            #save_df(df_http_cookie_edges, "df_http_cookie_edges")

            df_storage_node_setter = find_setters(
                df_all_storage_nodes,
                df_http_cookie_nodes,
                df_all_storage_edges,
                df_http_cookie_edges,
            )

            #save_df(df_storage_node_setter, "df_storage_node_setter")

            # Concatenate to get all nodes and edges
            df_request_nodes["domain"] = None
            df_decoration_nodes["domain"] = None


            df_all_nodes = pd.concat(
                [df_js_nodes, df_request_nodes, df_storage_node_setter, df_decoration_nodes]
            )
            df_all_nodes["domain"] = df_all_nodes.apply(find_domain, axis=1)
            df_all_nodes["top_level_domain"] = df_all_nodes["top_level_url"].apply(find_tld)
            df_all_nodes["setter_domain"] = df_all_nodes["setter"].apply(find_setter_domain)
            # df_all_nodes['name'] = df_all_nodes.apply(update_storage_names, axis=1)
            df_all_nodes = df_all_nodes.drop_duplicates()
            df_all_nodes["graph_attr"] = "Node"

            df_all_edges = pd.concat(
                [
                    df_js_edges,
                    df_request_edges,
                    df_all_storage_edges,
                    df_http_cookie_edges,
                    df_decoration_edges,
                ]
            )
            df_all_edges = df_all_edges.drop_duplicates()
            df_all_edges["top_level_domain"] = df_all_edges["top_level_url"].apply(find_tld)
            df_all_edges["graph_attr"] = "Edge"

            df_all_graph = pd.concat([df_all_nodes, df_all_edges])
            df_all_graph = df_all_graph.astype(
                {"type": "category", "response_status": "category"}
            )

            #save_df(df_all_nodes, "df_all_nodes")
            #save_df(df_all_edges, "df_all_edges")
            #save_df(df_all_graph, "df_all_graph")

    except ValueError as e:
        print(e)
        df_all_graph = pd.DataFrame()

    return df_all_graph


def load_features_info(filename):
    """Load features from features.yaml file
    :param filename: yaml file name containing feature names
    :return: list of features to use.
    """
    with open(filename) as file:
        return full_load(file)


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
    return df_features

def get_features_phase1(pdf, G, features_file):
    """Getter to generate the features of each node in a graph.
    :param pdf: pandas df of nodes and edges in a graph.
    :param G: Graph object representation of the pdf.
    :return: dataframe of features per node in the graph
    """
    # Generate features for each node in our graph
    feature_config = load_features_info(features_file)
    df_features = extract_graph_features_phase1(pdf, G, pdf.visit_id[0], None, feature_config)
    return df_features

def get_features_phase1(pdf, G, visit_id, features_file, ldb_file, tag):
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

    df_features = extract_graph_features_phase1(pdf, G, visit_id, ldb, feature_config, tag)
    return df_features

def most_frequent_url_per_visit(df):
    # Group by 'visit_id' and then find the most frequent 'top_level_url' for each group
    most_frequent_url = df.groupby('visit_id')['top_level_url'].agg(lambda x: x.value_counts().idxmax())
    most_frequent_url_df = most_frequent_url.reset_index()
    most_frequent_url_df.columns = ['visit_id', 'top_level_url']
    return most_frequent_url



def label_url_type(df, url_type):
    df_document = df[(df["type"] == "Document")]
    df_edges = df[df["graph_attr"] == "Edge"]

    # Filter edges to include only rows with names that are in df_document
    df_edges = df_edges[df_edges['src'].isin(df_document['name'])]

    # Sort the edges by 'src' 'time_stamp', to get the first url triggrer the graph
    df_edges['time_stamp'] = pd.to_datetime(df_edges['time_stamp'], errors='coerce')
    df_edges_sorted = df_edges.sort_values(by=['src', 'time_stamp'])
    # Drop duplicates, keeping the first occurrence of each 'src' within each 'visit_id'
    df_edges_sorted = df_edges_sorted.drop_duplicates(subset=['visit_id', 'src'])

    # df_url_sorted: contains all the main frame url ("Document" type) within order
    df_url_sorted = df_edges_sorted[
        [
            "visit_id",
            "src",
            "top_level_domain",
            "attr",
            "time_stamp",
            "is_in_phase1",
        ]
    ]
    df_url_sorted = df_url_sorted.sort_values(by=['visit_id', 'time_stamp'])
    df_url_sorted = df_url_sorted.rename(
        columns={
            "src": "url",
            "top_level_domain": "domain"
        }
    )

    # Column[Domain_list] contains all the main frame url's domain
    # Group by 'visit_id' and aggregate 'domain' into an ordered list 
    domain_list = df_url_sorted.groupby('visit_id')['domain'].agg(lambda x: ' || '.join(x))
    # Map the aggregated domain list back to a new column in the original dataframe
    df_url_sorted['domain_list'] = df_url_sorted['visit_id'].map(domain_list)
    df_url_sorted = df_url_sorted.rename(
        columns={
            "domain_list": "name",
        }
    )
    df_url_sorted['label'] = str(url_type)


    # get the top_level_url 
    # this step can remove when sqlite is exist
    df_top_level_url = most_frequent_url_per_visit(df_edges)
    # print(df_top_level_url.head())
    df_url_sorted = df_url_sorted.merge(df_top_level_url, on='visit_id', how='left')


    return df_url_sorted

def apply_tasks(
    df,
    visit_id,
    features_file,
    final_page_url,
    ldb_file,
    graph_columns,
    graph_folder
):
    try:
        start = time.time()
        graph_fname = "graph.csv"
        
        graph_path = os.path.join(graph_folder,graph_fname)
        if not os.path.exists(graph_path):
            df.reindex(columns=graph_columns).to_csv(graph_path)
        else:
            df.reindex(columns=graph_columns).to_csv(
                graph_path, mode="a", header=False
            )
        # G = create_graph(df)

        graph_approach_features.pipeline(df, visit_id, final_page_url, graph_columns, graph_path)


    except Exception as e:
        print("Errored in pipeline:", e)
        traceback.print_exc()


def pipeline(db_file, features_file, ldb_file, graph_folder, graph_type):
    start = time.time()
    conn = gs.get_local_db_connection(db_file)
    try:
        sites_visits = read_sites_visit(db_file, conn)
    except Exception as e:
        tqdm.write(f"Problem reading the sites_visits or the scraper data: {e}")
        exit()

    end = time.time()
    print("read site visits", end - start)

    fail = 0
    start = time.time()

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
    
    
    for i, row in tqdm(
        sites_visits.iterrows(),
        total=len(sites_visits),
        position=0,
        leave=True,
        ascii=True,
    ):
        # For each visit, grab the visit_id and the site_url
        visit_id = row["visit_id"]
        site_url = row["site_url"]
        tqdm.write("")
           
        tqdm.write(f"• Visit ID: {visit_id} | Site URL: {site_url}")
        try:
            start = time.time()

            #if visit_id != 335396880693770:
            #    continue
            

            """
            # In case crawl failed in middle, like pipeline broken. Restart the building.

            storage_features_fname = "storage.csv"
            features_path = os.path.join(graph_folder,storage_features_fname)
            df_features = pd.read_csv(features_path)
            # if visit_id in df_features["visit_id"], continue
            if visit_id in df_features['visit_id'].values:
                print("continue since already include")
                continue
            """

            # this cannot be parallelized as it is reading from the sqlite file, only one process at a time can do that
            pdf = read_sql_crawl_data(visit_id, db_file, conn)
            if pdf.empty:
                print("Fail to crawl this link since empty callstacks. Ignore")
                # TODO: add these failed to crawl link to OpenWPM and prepare for second time crawl
                continue
            
            final_page_url = gs.get_final_page_url(conn, visit_id)
            print("final_page_url: ", final_page_url)
        
            end = time.time()
            print("Built graph of shape: ", pdf.shape, "in :", end - start)
            pdf = pdf[pdf["top_level_domain"].notnull()]
            
            
            """
            # Below is procesing. Check if a link is affiliate or not
            normal_link = "/home/data/chensun/affi_project/purl/code/normal_potential.csv"
            affiliate_link = "/home/data/chensun/affi_project/purl/code/affiliate_potential.csv"
            
            df_features_normal = pd.read_csv(normal_link)
            df_features_affiliate = pd.read_csv(affiliate_link)
            # if the site_url already in either two file, continue
            if site_url in df_features_normal['site_url'].values or site_url in df_features_affiliate['site_url'].values:
                print("continue since already include")
                continue

            # Check if the url is affiliate link or not.
            if check_affiliate_link(visit_id, site_url, conn):
                print("Affiliate link.")
                
            else:
                print("Not affiliate link")

                columns = ["site_url"]
                df = pd.DataFrame([[site_url]], columns=columns)
                if not os.path.exists(normal_link):
                    df.to_csv(normal_link, index=False)
                else:
                    df.to_csv(normal_link, mode='a', header=False, index=False)
            """

            
            """
            # extract graph features
            
            pdf.groupby(["visit_id"]).apply(
                apply_tasks,
                visit_id,
                features_file,
                final_page_url,
                ldb_file,
                graph_columns,
                graph_folder
            )
            """
            graph_approach_features.pipline_only_extract_storage_features(pdf, visit_id, final_page_url, graph_columns, graph_folder)
            
            end = time.time()
            print("Finished processing graph: ", row["visit_id"], "in :", end - start)

            

            # Collect site url domain
            print("Collecting site url domain")
            features_path = os.path.join(graph_folder,"features_phase1.csv")
            df_features = pd.read_csv(features_path)
            if visit_id not in df_features['visit_id'].values:
                continue   # ignore since this visit_id is failed to build graph

            records_fname = "records.csv"
            records_path = os.path.join(graph_folder, records_fname)
            print(records_path)
            site_url_domain = gs.get_domain(site_url)
            landing_page_domain = gs.get_domain(final_page_url)
            df = pd.DataFrame({
                'visit_id': [visit_id],
                'url_domain': [site_url_domain],
                'url': [site_url],
                'landing_page_domain': [landing_page_domain],
                'landing_page_url': [final_page_url],
            })
            if not os.path.exists(records_path):
                df.to_csv(records_path, index=False)
            else:
                df.to_csv(records_path, mode="a", header=False, index=False)
            

        except Exception as e:
            fail += 1
            tqdm.write(f"Fail: {fail}")
            tqdm.write(f"Error: {e}")
            traceback.print_exc()
            pass
        
    
    # Label the graph
    print("Labelling the graph")
    label_fname = "label.csv"
    labels_path = os.path.join(graph_folder, label_fname)
    print(labels_path)

    graph_fname = "graph.csv"     
    graph_path = os.path.join(graph_folder,graph_fname)
    print(graph_path)

    df_Graph = pd.read_csv(graph_path)
    df_labels = label_url_type(df_Graph, graph_type)

    df_labels.to_csv(labels_path, index=False)



def pipeline_for_ads(db_file, features_file, ldb_file, graph_folder, graph_type):
    start = time.time()
    conn = gs.get_local_db_connection(db_file)
    try:
        sites_visits = read_sites_visit(db_file, conn)
    except Exception as e:
        tqdm.write(f"Problem reading the sites_visits or the scraper data: {e}")
        exit()

    end = time.time()
    print("read site visits", end - start)

    fail = 0
    start = time.time()

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
    
    
    for i, row in tqdm(
        sites_visits.iterrows(),
        total=len(sites_visits),
        position=0,
        leave=True,
        ascii=True,
    ):
        # For each visit, grab the visit_id and the site_url
        visit_id = row["visit_id"]
        site_url = row["site_url"]
        print(f"Visit ID: {visit_id} | Site URL: {site_url}")

        try:

            #if visit_id != 3182746237897655:
            #    continue
            
            """
            # In case crawl failed in middle, like pipeline broken. Restart the building.

            storage_features_fname = "storage.csv"
            features_path = os.path.join(graph_folder,storage_features_fname)
            df_features = pd.read_csv(features_path)
            # if visit_id in df_features["visit_id"], continue
            if visit_id in df_features['visit_id'].values:
                print("continue since already include")
                continue
            """
          
            unique_ad_tab_ids, first_urls = gs.unique_ad_tab_ids_fake(conn, visit_id)

            #unique_ad_tab_ids = 2
            #first_urls = 'https://l.instagram.com/?u=http%3A%2F%2Fwww.glaminatrixcosmetics.com.au%2F&e=AT35cLZBV-2c_Rdq6VFrryIJc4HXIvcDDgr6XvYv1O_EIu-iM0PT3xSYN3eC7fOBxY1jCjEDbkDSET2IAKluZCx7VYOsuq8g4v3Snx2WCFZ-ieuG'
            if len(unique_ad_tab_ids) == 0:
                
                continue
            print("len of graph: ", len(unique_ad_tab_ids))
            for i in range(len(unique_ad_tab_ids)):

                ad_url = first_urls[(visit_id, unique_ad_tab_ids[i])]

                tqdm.write("")
           
                tqdm.write(f"• Visit ID: {visit_id} | TAG ID: {unique_ad_tab_ids[i]} | AD URL: {ad_url}")

                start = time.time()
                pdf = read_sql_crawl_data_for_ads(visit_id, db_file, conn, unique_ad_tab_ids[i])
                
                # this cannot be parallelized as it is reading from the sqlite file, only one process at a time can do that
                #pdf = read_sql_crawl_data(visit_id, db_file, conn)
                if pdf.empty:
                    print("Fail to crawl this link since empty callstacks. Ignore")
                    # TODO: add these failed to crawl link to OpenWPM and prepare for second time crawl
                    continue
            
                final_page_url = gs.get_final_page_url_for_ads(conn, visit_id, unique_ad_tab_ids[i])
                print("final_page_url: ", final_page_url)
        
                end = time.time()
                print("Built graph of shape: ", pdf.shape, "in :", end - start)
                pdf = pdf[pdf["top_level_domain"].notnull()]

                # concate new visit id
                pdf['visit_id'] = pdf['visit_id'].astype(str) + "_" + str(unique_ad_tab_ids[i])
                visit_id = str(visit_id)   + "_" + str(unique_ad_tab_ids[i])    

                # extract graph features   
                pdf.groupby(["visit_id"]).apply(
                    apply_tasks,
                    visit_id,
                    features_file,
                    final_page_url,
                    ldb_file,
                    graph_columns,
                    graph_folder
                )
            
                #graph_approach_features.pipline_only_extract_storage_features(pdf, visit_id, final_page_url, graph_columns, graph_folder)
            
                end = time.time()
                print("Finished processing graph: ", row["visit_id"], " with tab: ", unique_ad_tab_ids[i], " in :", end - start)


                # Collect site url domain
                print("Collecting site url domain")
                #features_path = os.path.join(graph_folder,"features_phase1.csv")
                #df_features = pd.read_csv(features_path)
                #if visit_id not in df_features['visit_id'].values:
                #    continue   # ignore since this visit_id is failed to build graph

                records_fname = "records.csv"
                records_path = os.path.join(graph_folder, records_fname)
                print(records_path)
            
                ad_url_domain = gs.get_domain(ad_url)   
                landing_page_domain = gs.get_domain(final_page_url)
                parent_page_url = site_url
                parent_domain =  gs.get_domain(parent_page_url)
                
                df = pd.DataFrame({
                    'visit_id': [visit_id],
                    'url_domain': [ad_url_domain],
                    'url': [ad_url],
                    'landing_page_domain': [landing_page_domain],
                    'landing_page_url': [final_page_url],
                    'parent_page_url': [parent_page_url],
                    'parent_domain': [parent_domain]
                })
                if not os.path.exists(records_path):
                    df.to_csv(records_path, index=False)
                else:
                    df.to_csv(records_path, mode="a", header=False, index=False)
            

        except Exception as e:
            fail += 1
            tqdm.write(f"Fail: {fail}")
            tqdm.write(f"Error: {e}")
            traceback.print_exc()
            pass
        
    
    # Label the graph
    print("Labelling the graph")
    label_fname = "label.csv"
    labels_path = os.path.join(graph_folder, label_fname)
    print(labels_path)

    graph_fname = "graph.csv"     
    graph_path = os.path.join(graph_folder,graph_fname)
    print(graph_path)

    df_Graph = pd.read_csv(graph_path)
    df_labels = label_url_type(df_Graph, graph_type)

    df_labels.to_csv(labels_path, index=False)
    

    



if __name__ == "__main__":

    # get the features file, the dataset folder and the output folder from the command line\
    parser = argparse.ArgumentParser(
        description="Process the Graph features for Link Dedecorator."
    )
    parser.add_argument(
        "--features", type=str, default="features_new.yaml", help="the features file"
    )
    parser.add_argument("--folder", type=str, default="data", help="the dataset folder")
    parser.add_argument(
        "--output", type=str, default="output", help="the output folder"
    )
    parser.add_argument(
        "--tag", type=str, default="", help="the tag for the output file"
    )
    args = parser.parse_args()

    FEATURES_FILE = args.features
    FOLDER = args.folder
    OUTPUT = args.output
    TAG = args.tag

    # DB_FILE = os.path.join(FOLDER, "crawl-data.sqlite")
    # LDB_FILE = os.path.join(FOLDER, "content.ldb")da

    # pipeline(DB_FILE, FEATURES_FILE, LDB_FILE, TAG)

    #for i in range(0, 3000, 1000):
    #print("Processing:", i)
    # DB_FILE = os.path.join(FOLDER, f"test-database/crawl-data_2.sqlite")
    # LDB_FILE = os.path.join(FOLDER, f"datadir_ads_unseen_data_0/content.ldb")
    # LDB_FILE = ''

    DB_FILE = "/home/data/chensun/affi_project/purl_test/OpenWPM_2/datadir_youtube_aff-normal-50000/crawl-data.sqlite"
    LDB_FILE = ""

    print(DB_FILE, LDB_FILE)

    subfolder = "crawl_"+ TAG
    graph_type = "ads" #!! change to affiliate/ads/normal !!
    graph_folder = os.path.join(OUTPUT, graph_type, subfolder)

    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)


    
    #graph_folder = os.path.abspath('../output/affiliate')  
    #print(graph_folder)
    #TAG = str(0)
    pipeline_for_ads(DB_FILE, FEATURES_FILE, LDB_FILE, graph_folder, graph_type)  # change this
        
    """
    print("Get landing page domain into records file")
    records_fname = "records.csv"
    records_path = os.path.join(graph_folder, records_fname)
    feature_fname = "features_phase1.csv"
    feature_path = os.path.join(graph_folder, feature_fname)

    df_records = pd.read_csv(records_path)
    df_feature = pd.read_csv(feature_path)
    df_feature['landing_page_domain'] = df_feature['top_level_url'].apply(lambda x: gs.get_domain(x))
    df_feature['landing_page_url'] = df_feature['top_level_url']
    columns_to_keep = ['visit_id', 'landing_page_domain', 'landing_page_url']
    new_df = df_feature[columns_to_keep]
    df_records_2 = pd.merge(df_records, new_df, on='visit_id', how='inner')
    records_2_path = os.path.join(graph_folder, "records_2.csv")
    df_records_2.to_csv(records_2_path, index=False)
    """
        
    """
    print("Populate records file with parent scraped page information")
    records_fname = "records.csv"
    records_path = os.path.join(graph_folder, records_fname)

    parent_url_path = "/home/data/chensun/affi_project/purl/urls/ads/scraped_ad_urls.csv" # change this!

    df_parent_page = pd.read_csv(parent_url_path)
    df_parent_page['parent_domain'] = df_parent_page['parent_page_url'].apply(lambda x: gs.get_domain(x))
    
    df_records = pd.read_csv(records_path)


    df_records_2 = pd.merge(df_records, df_parent_page, on='url', how='inner')
    #df_records = pd.merge(df_records, df_parent_page, on='url', how='inner')
    df_records_2 = df_records_2.drop_duplicates(['url', 'visit_id'])
    records_path = os.path.join(graph_folder, "records_2.csv")
    df_records_2.to_csv(records_path, index=False)
    """
        
    
    

    # folders = os.listdir(FOLDER)
    # print(folders)
    # for folder in folders:
    #     tag = folder[-1:]
    #     print("Processing:", folder, tag)
    #     DB_FILE = os.path.join(FOLDER, folder, "crawl-data.sqlite")
    #     ldb_file = os.path.join(FOLDER, folder, "content.ldb")
    #     pipeline(DB_FILE, FEATURES_FILE, ldb_file, tag)
    #     first = True