import pandas as pd
import networkx as nx
import traceback
from .utils import *


def find_common_name(name):
    parts = name.split("|$$|")
    if len(parts) == 3:
        return name.rsplit("|$$|", 1)[0]
    return name


def get_storage_features(df_graph, node):
    num_get_storage = 0
    num_set_storage = 0
    num_get_storage_ls = 0
    num_set_storage_ls = 0
    num_ls_gets = 0
    num_ls_sets = 0
    num_ls_gets_js = 0
    num_ls_sets_js = 0
    num_cookieheader_exfil = 0

    try:
        request_nodes = df_graph[
            (df_graph["dst"] == node) & (df_graph["type"] == "Request")
        ]

        for request_node in request_nodes.iterrows():
            node_name = request_node["name"]
            cookie_js_get = df_graph[
                (df_graph["dst"] == node_name) & (df_graph["action"] == "get_js")
            ]

            cookie_js_set = df_graph[
                (df_graph["dst"] == node_name) & (df_graph["action"] == "set_js")
            ]

            cookie_get = df_graph[
                (df_graph["dst"] == node_name)
                & ((df_graph["action"] == "get") | (df_graph["action"] == "get_js"))
            ]

            cookie_header = df_graph[
                (df_graph["dst"] == node_name) & (df_graph["action"] == "get")
            ]

            cookie_set = df_graph[
                (df_graph["dst"] == node_name)
                & ((df_graph["action"] == "set") | (df_graph["action"] == "set_js"))
            ]

            localstorage_get = df_graph[
                (df_graph["dst"] == node_name)
                & (df_graph["action"] == "get_storage_js")
            ]

            localstorage_set = df_graph[
                (df_graph["dst"] == node_name)
                & (df_graph["action"] == "set_storage_js")
            ]

            num_get_storage += len(cookie_get) + len(localstorage_get)
            num_set_storage += len(cookie_set) + len(localstorage_set)
            num_get_storage_js += len(cookie_js_get)
            num_set_storage_js += len(cookie_js_set)

            df_graph_gets = df_graph[
                (df_graph["action"] == "get")
                | (df_graph["action"] == "get_js")
                | (df_graph["action"] == "get_storage_js")
            ].copy()
            df_graph_sets = df_graph[
                (df_graph["action"] == "set")
                | (df_graph["action"] == "set_js")
                | (df_graph["action"] == "set_storage_js")
            ].copy()
            df_graph_gets["new_dst"] = df_graph_gets["dst"].apply(find_common_name)
            df_graph_sets["new_dst"] = df_graph_sets["dst"].apply(find_common_name)

            num_ls_gets += len(df_graph_gets[df_graph_gets["new_dst"] == node_name])
            num_ls_sets += len(df_graph_sets[df_graph_sets["new_dst"] == node_name])

            df_graph_gets_js = df_graph[
                (df_graph["action"] == "get_js")
                | (df_graph["action"] == "get_storage_js")
            ].copy()
            df_graph_sets_js = df_graph[
                (df_graph["action"] == "set_js")
                | (df_graph["action"] == "set_storage_js")
            ].copy()
            df_graph_gets_js["new_dst"] = df_graph_gets_js["dst"].apply(
                find_common_name
            )
            df_graph_sets_js["new_dst"] = df_graph_sets_js["dst"].apply(
                find_common_name
            )

            num_ls_gets_js = len(
                df_graph_gets_js[df_graph_gets_js["new_dst"] == node_name]
            )
            num_ls_sets_js = len(
                df_graph_sets_js[df_graph_sets_js["new_dst"] == node_name]
            )

        num_cookieheader_exfil = len(cookie_header)

        storage_features = [
            num_get_storage,
            num_set_storage,
            num_get_storage_js,
            num_set_storage_js,
            num_ls_gets,
            num_ls_sets,
            num_ls_gets_js,
            num_ls_sets_js,
            num_cookieheader_exfil,
        ]
        storage_feature_names = [
            "parent_num_get_storage",
            "parent_num_set_storage",
            "parent_num_get_storage_js",
            "parent_num_set_storage_js",
            "parent_num_get_storage_ls",
            "parent_num_set_storage_ls",
            "parent_num_get_storage_ls_js",
            "parent_num_set_storage_ls_js",
            "parent_num_cookieheader_exfil",
        ]
    except:
        storage_features = [
            num_get_storage,
            num_set_storage,
            num_get_storage_ls,
            num_set_storage_ls,
            num_ls_gets,
            num_ls_sets,
            num_ls_gets_js,
            num_ls_sets_js,
            num_cookieheader_exfil,
        ]
        storage_feature_names = [
            "parent_num_get_storage",
            "parent_num_set_storage",
            "parent_num_get_storage_js",
            "parent_num_set_storage_js",
            "parent_num_get_storage_ls",
            "parent_num_set_storage_ls",
            "parent_num_get_storage_ls_js",
            "parent_num_set_storage_ls_js",
            "parent_num_cookieheader_exfil",
        ]

    return storage_features, storage_feature_names

def extract_storage_features(df_graph, key, final_page_url, graph_path):
    num_get_storage = 0
    num_set_storage = 0
    num_get_storage_js = 0
    num_set_storage_js = 0
    num_all_gets = 0
    num_all_sets = 0
    num_get_storage_in_product_node = 0
    num_set_storage_in_product_node = 0
    num_get_storage_js_in_product_node = 0
    num_set_storage_js_in_product_node = 0
    num_all_gets_in_product_node = 0
    num_all_sets_in_product_node = 0

    try:
       
        # get the storage information for the graph
        # "get_storage_js" get the storage related actions from the javascript table

        #df_graph.to_csv("/home/data/chensun/affi_project/purl/output/affiliate/storage_features_test.csv")

        cookie_js_get = df_graph[(df_graph["action"] == "get_js")]
        

        cookie_js_set = df_graph[(df_graph["action"] == "set_js")]

        cookie_get = df_graph[((df_graph["action"] == "get") | (df_graph["action"] == "get_js"))]

        cookie_header = df_graph[(df_graph["action"] == "get")]

        cookie_set = df_graph[((df_graph["action"] == "set") | (df_graph["action"] == "set_js"))]

        localstorage_get = df_graph[(df_graph["action"] == "get_storage_js")]

        localstorage_set = df_graph[(df_graph["action"] == "set_storage_js")]

        num_get_storage = len(cookie_get) + len(localstorage_get)
        num_set_storage = len(cookie_set) + len(localstorage_set)
        num_get_storage_js = len(cookie_js_get)
        num_set_storage_js = len(cookie_js_set)
        df_graph_gets = df_graph[
            (df_graph["action"] == "get")
            | (df_graph["action"] == "get_js")
            | (df_graph["action"] == "get_storage_js")
        ].copy()
        df_graph_sets = df_graph[
            (df_graph["action"] == "set")
            | (df_graph["action"] == "set_js")
            | (df_graph["action"] == "set_storage_js")
        ].copy()
        num_all_gets = len(df_graph_gets)
        # print("num_all_gets: ", num_all_gets)
        num_all_sets = len(df_graph_sets)
        # print("num_all_sets: ", num_all_sets)



        # get the storage information from the final product node
        cookie_js_get_in_product_node = df_graph[(df_graph["action"] == "get_js") & (df_graph['src'] == final_page_url)]

        cookie_js_set_in_product_node = df_graph[(df_graph["action"] == "set_js" ) & ( df_graph['src'] == final_page_url)]

        cookie_get_in_product_node = df_graph[( df_graph['src'] == final_page_url) & ((df_graph["action"] == "get") | (df_graph["action"] == "get_js")) ]

        cookie_header_in_product_node = df_graph[( df_graph['src'] == final_page_url) & (df_graph["action"] == "get")]

        cookie_set_in_product_node = df_graph[( df_graph['src'] == final_page_url) & ((df_graph["action"] == "set") | (df_graph["action"] == "set_js"))]

        localstorage_get_in_product_node = df_graph[( df_graph['src'] == final_page_url) & (df_graph["action"] == "get_storage_js")]

        localstorage_set_in_product_node = df_graph[( df_graph['src'] == final_page_url) & (df_graph["action"] == "set_storage_js")]

        num_get_storage_in_product_node = len(cookie_get_in_product_node) + len(localstorage_get_in_product_node)
        num_set_storage_in_product_node = len(cookie_set_in_product_node) + len(localstorage_set_in_product_node)
        num_get_storage_js_in_product_node = len(cookie_js_get_in_product_node)
        num_set_storage_js_in_product_node = len(cookie_js_set_in_product_node)
        df_graph_gets = df_graph[
            (df_graph["action"] == "get")
            | (df_graph["action"] == "get_js")
            | (df_graph["action"] == "get_storage_js")
        ].copy()
        df_graph_sets = df_graph[
            (df_graph["action"] == "set")
            | (df_graph["action"] == "set_js")
            | (df_graph["action"] == "set_storage_js")
        ].copy()

        
        num_all_gets_in_product_node = len(df_graph_gets[(df_graph_gets['src'] == final_page_url)])
        num_all_sets_in_product_node = len(df_graph_sets[(df_graph_sets['src'] == final_page_url)])

        # print("num_all_sets_in_product_node: ", num_all_sets_in_product_node)
        # print("num_all_gets_in_product_node: ", num_all_gets_in_product_node)

        storage_features = [
            num_get_storage,
            num_set_storage,
            num_get_storage_js,
            num_set_storage_js,
            num_all_gets,
            num_all_sets,
            num_get_storage_in_product_node,
            num_set_storage_in_product_node,
            num_get_storage_js_in_product_node,
            num_set_storage_js_in_product_node,
            num_all_gets_in_product_node,
            num_all_sets_in_product_node,
        ]
        storage_feature_names = [
            "num_get_storage",
            "num_set_storage",
            "num_get_storage_js",
            "num_set_storage_js",
            "num_all_gets",
            "num_all_sets",
            "num_get_storage_in_product_node",
            "num_set_storage_in_product_node",
            "num_get_storage_js_in_product_node",
            "num_set_storage_js_in_product_node",
            "num_all_gets_in_product_node",
            "num_all_sets_in_product_node",
        ]  
    except:
        storage_features = [
            num_get_storage,
            num_set_storage,
            num_get_storage_js,
            num_set_storage_js,
            num_all_gets,
            num_all_sets,
            num_get_storage_in_product_node,
            num_set_storage_in_product_node,
            num_get_storage_js_in_product_node,
            num_set_storage_js_in_product_node,
            num_all_gets_in_product_node,
            num_all_sets_in_product_node,
            ]
        storage_feature_names = [
            "num_get_storage",
            "num_set_storage",
            "num_get_storage_js",
            "num_set_storage_js",
            "num_all_gets",
            "num_all_sets",
            "num_get_storage_in_product_node",
            "num_set_storage_in_product_node",
            "num_get_storage_js_in_product_node",
            "num_set_storage_js_in_product_node",
            "num_all_gets_in_product_node",
            "num_all_sets_in_product_node",
        ]
    return storage_features, storage_feature_names

            