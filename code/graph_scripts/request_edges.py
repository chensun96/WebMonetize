import pandas as pd
import json
import re
import multidict
import traceback
import numpy as np


def get_attr(cpt, tlu):
    """Function to set attributes for HTTP nodes."""

    record = {"content_policy_type": cpt, "top_level_url": tlu}
    return json.dumps(record)


def convert_type(orig_type, attr):
    """Function to set high level category for nodes (Script/Document/HTTP)."""

    attr = json.loads(attr)
    new_type = orig_type
    if attr["content_policy_type"] == "script":
        new_type = "Script"
    if attr["content_policy_type"] == "main_frame":
        new_type = "Document"
    return new_type


def get_key(v1, v2):
    return str(int(v1)) + "_" + str(int(v2))


def process_attr(respattr):
    """Function to build set headers as edge attributes."""

    attr = {}
    try:
        respattr = json.loads(respattr)
        for item in respattr:
            if item[0] == "Content-Length":
                attr["clength"] = int(item[1])
            if item[0] == "Content-Type":
                attr["ctype"] = item[1]
        return json.dumps(attr)
    except:
        return None


def process_redirects(df):
    header_list = df["respattr1"].append(df.iloc[-1:]["headers"], ignore_index=True)
    status_list = df["response_status_x"].append(
        df.iloc[-1:]["response_status_y"], ignore_index=True
    )
    edges = df[
        [
            "visit_id",
            "old_request_url",
            "new_request_url",
            "top_level_url_x",
            "reqattr2",
            "time_stamp_x",
            "is_in_phase1"
        ]
    ]
    # edges.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/process_redirects.csv', index=False)
    edges = edges.rename(
        columns={
            "old_request_url": "src",
            "time_stamp_x": "time_stamp",
            "new_request_url": "dst",
            "reqattr2": "reqattr",
            "top_level_url_x": "top_level_url",
        }
    )
    first_row = df.iloc[0]
    data = []
    new_entry = {
        "visit_id": first_row["visit_id"],
        "src": first_row["top_level_url_x"],
        "dst": first_row["old_request_url"],
        "top_level_url": first_row["top_level_url_x"],
        "reqattr": first_row["reqattr1"],
        "time_stamp": first_row["time_stamp_x"],
        "is_in_phase1": first_row["is_in_phase1"],
    }
    data.insert(0, new_entry)
    edges = pd.concat([pd.DataFrame(data), edges], ignore_index=True)
    edges["respattr"] = header_list
    edges["response_status"] = status_list
    return edges


def get_redirect_edges(df_requests, df_redirects, df_responses):

    df_reqheaders = df_requests[
        ["visit_id", "request_id", "url", "headers", "top_level_url", "time_stamp", "is_in_phase1"]
    ]

    
    # df_reqheaders.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_reqheaders.csv', index=False)

    df_red = df_redirects[
        [
            "visit_id",
            "old_request_id",
            "old_request_url",
            "new_request_url",
            "headers",
            "response_status",
            "is_in_phase1"
        ]
    ]
    # df_red.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_red.csv', index=False)

    x1 = pd.merge(
        df_red,
        df_reqheaders,
        left_on=["visit_id", "old_request_id", "old_request_url", "is_in_phase1"],
        right_on=["visit_id", "request_id", "url", "is_in_phase1"],
    )

    # x1.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/x1.csv', index=False)

    x2 = pd.merge(
        x1,
        df_requests,
        left_on=["visit_id", "old_request_id", "new_request_url", "is_in_phase1"],
        right_on=["visit_id", "request_id", "url", "is_in_phase1"],
    )

    # x2.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/x2.csv', index=False)

    x2 = x2.rename(
        columns={
            "headers_x": "respattr1",
            "headers_y": "reqattr1",
            "headers": "reqattr2",
        }
    )

    # x2.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/x2_rename.csv', index=False)
    x3 = pd.merge(
        x2,
        df_responses,
        left_on=["visit_id", "old_request_id", "new_request_url", "is_in_phase1"],
        right_on=["visit_id", "request_id", "url", "is_in_phase1"],
        how="outer",
    )
    # x3.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/x3.csv', index=False)

    df_redirect_edges = (
        x3.groupby(["visit_id", "old_request_id"], as_index=False)
        .apply(process_redirects)
        .reset_index()
    )

    # df_redirect_edges.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_redirect_edges.csv', index=False)
    df_redirect_edges = df_redirect_edges[
        [
            "visit_id",
            "src",
            "dst",
            "top_level_url",
            "reqattr",
            "respattr",
            "response_status",
            "time_stamp",
            "is_in_phase1",
        ]
    ]
    df_redirect_edges["content_hash"] = pd.NA
    df_redirect_edges["post_body"] = pd.NA
    df_redirect_edges["post_body_raw"] = pd.NA

    completed_ids = x3["key_x"].unique().tolist()

    # df_redirect_edges.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_redirect_edges_final.csv', index=False)
    # print("completed_ids: ", completed_ids)
    return df_redirect_edges, completed_ids

def process_call_stack(row):
    """Function to process call stack information."""
    """ DEBUG - TO REMOVE
    node_t = "https://kraken.rambler.ru/cnt/"
    node_comp = "https://music.yandex.ru/api/v2.1/index/music.yandex.ru"
    if row['name'] == node_t or row['name'] == node_comp:
        print("--------")
        print(row['name'])
        print(cs_lines)
        print("DONE HERE")
    """
    cs_lines = row["call_stack"].split()
    urls = []
    new_urls = []
    min_len = 5
    for line in cs_lines:
        """OLD CODE WITH BUG
        components = re.split ('[@;]', line)
        if row['name'] == node_t or row['name'] == node_comp:
            print("********")
            print("LINE:" + str(line))
            print("COMPONENT:" + str(components))
        if len(components) >= 2:
            if row['name'] == node_t or row['name'] == node_comp:
                print("LEN COMPONENTS > 2")
                print("interval: " +str(components[1].rsplit(":", 2)))
                print("Appending: "+ str(components[1].rsplit(":", 2)[0]))
            urls.append(components[1].rsplit(":", 2)[0])
        """
        url_parsing = re.search("(?P<url>https?://[^\s:]+)", line)
        if url_parsing != None:
            urls.append(url_parsing.group("url"))
    urls = urls[::-1]
    urls = list(set(urls))
    for url in urls:
        if len(new_urls) == 0:
            new_urls.append(url)
        else:
            if new_urls[-1] != url:
                new_urls.append(url)
    edge_data = []
    if len(new_urls) > 1:
        for i in range(0, len(new_urls) - 1):
            src_cs = new_urls[i]
            dst_cs = new_urls[i + 1]
            reqattr_cs = "CS"
            respattr_cs = "CS"
            status_cs = 999
            post_body_cs = "CS"
            post_body_raw_cs = "CS"
            edge_data.append(
                [
                    src_cs,
                    dst_cs,
                    reqattr_cs,
                    respattr_cs,
                    status_cs,
                    row["time_stamp"],
                    row["visit_id"],
                    row["content_hash"],
                    post_body_cs,
                    post_body_raw_cs,
                ]
            )
    if len(new_urls) > 0:
        edge_data.append(
            [
                new_urls[-1],
                row["name"],
                row["reqattr"],
                row["respattr"],
                row["response_status"],
                row["time_stamp"],
                row["visit_id"],
                row["content_hash"],
                row["post_body"],
                row["post_body_raw"],
            ]
        )
    """ DEBUG - TO REMOVE
    if row['name'] == node_t or row['name'] == node_comp:
        print("Edge Data")
        print(len(new_urls))
        print(edge_data)
        print("-----------------------")
    """
    # print("edge_data ", edge_data)
    return edge_data


def get_cs_edges(df_requests, df_responses, call_stacks):
    """DEBUG - TO REMOVE
    node = "https://kraken.rambler.ru/cnt/"
    print("----Hello----")
    print(df_requests[df_requests.url==node])
    print(df_responses[df_responses.url==node])
    print("----Hello----")
    """
    try:
        if call_stacks.empty:
            raise ValueError("======== callstacks table is empty =========")

        else:

            # print("df_requests visit_id: ", df_requests['visit_id'].dtype)
            # print("df_responses visit_id: ", df_responses['visit_id'].dtype)
            # print("call_stacks visit_id: ", call_stacks['visit_id'].dtype)

            # print("df_requests request_id: ", df_requests['request_id'].dtype)
            # print("df_responses request_id: ", df_responses['request_id'].dtype)
            # print("call_stacks request_id: ", call_stacks['request_id'].dtype)

            df_merge = pd.merge(
                df_requests, df_responses, on=["visit_id", "request_id"], how="inner"
            )
            # df_merge.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_merge_first.csv', index=False)
            

            call_stack_nodup = (
                call_stacks[["visit_id", "request_id", "call_stack", "is_in_phase1"]].drop_duplicates().copy()
            )
            # call_stack_nodup.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/call_stack_nodup.csv', index=False)
            
            df_merge = pd.merge(
                df_merge, call_stack_nodup, on=["visit_id", "request_id"], how="inner"
            )
            # df_merge.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_merge_call_stack_nodup.csv', index=False)
            

            df_merge = df_merge[
                [
                    "visit_id",
                    "url_x",
                    "top_level_url",
                    "headers_x",
                    "headers_y",
                    "time_stamp_x",
                    "response_status",
                    "post_body",
                    "post_body_raw",
                    "content_hash",
                    "call_stack",
                    "key_x",
                    "is_in_phase1",
                ]
            ]
            df_merge = df_merge.rename(
                columns={
                    "url_x": "name",
                    "headers_x": "reqattr",
                    "headers_y": "respattr",
                    "time_stamp_x": "time_stamp",
                    "key_x": "key",
                }
            )
            
            # df_merge.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_merge.csv', index=False)
            
            # no data can be merged
            if df_merge.empty:
                df_cs_edges = pd.DataFrame()
                completed_ids = []
                print("======== df_cs_edges is empty =========")
                return df_cs_edges, completed_ids
            
            df_merge["cs_edges"] = df_merge.apply(process_call_stack, axis=1)
            # df_merge.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_merge_2.csv', index=False)
            

            df = df_merge[["top_level_url", "cs_edges", "is_in_phase1"]]
            df = df.explode("cs_edges").dropna()
            df["src"] = df["cs_edges"].apply(lambda x: x[0])
            df["dst"] = df["cs_edges"].apply(lambda x: x[1])
            df["reqattr"] = df["cs_edges"].apply(lambda x: x[2])
            df["respattr"] = df["cs_edges"].apply(lambda x: x[3])
            df["response_status"] = df["cs_edges"].apply(lambda x: x[4])
            df["time_stamp"] = df["cs_edges"].apply(lambda x: x[5])
            df["visit_id"] = df["cs_edges"].apply(lambda x: x[6])
            df["content_hash"] = df["cs_edges"].apply(lambda x: x[7])
            df["post_body"] = df["cs_edges"].apply(lambda x: x[8])
            df["post_body_raw"] = df["cs_edges"].apply(lambda x: x[9])
            df_cs_edges = df.drop(columns=["cs_edges"]).reset_index()
            del df_cs_edges["index"]
            # df_cs_edges.to_csv('/home/ubuntu/df_cd_edges.csv', index=False)

            completed_ids = df_merge["key"].unique().tolist()

    except ValueError as e:
        print("Error occur during get_cs_edges: ", e)  # Optionally log the exception message
        df_cs_edges = pd.DataFrame()  # Create an empty DataFrame
        completed_ids = []  # Create an empty list for completed_ids

    return df_cs_edges, completed_ids


def get_normal_edges(df_requests, df_responses, completed_ids):
    """Function to build edges that are not redirect edges."""
    df_remaining = df_requests[~df_requests["key"].isin(completed_ids)]
    df_remaining = pd.merge(df_remaining, df_responses, on=["key"])
    df_normal_edges = df_remaining[
        [
            "visit_id_x",
            "top_level_url",
            "url_x",
            "headers_x",
            "headers_y",
            "response_status",
            "post_body",
            "post_body_raw",
            "time_stamp_x",
            "content_hash",
            "is_in_phase1_y",
        ]
    ]
    df_normal_edges = df_normal_edges.rename(
        columns={
            "visit_id_x": "visit_id",
            "top_level_url": "src",
            "url_x": "dst",
            "headers_x": "reqattr",
            "headers_y": "respattr",
            "time_stamp_x": "time_stamp",
            "is_in_phase1_y": "is_in_phase1",
        }
    )
    df_normal_edges["top_level_url"] = df_normal_edges["src"]
    # df_normal_edges.to_csv('/home/ubuntu/df_normal.csv', index=False)
    return df_normal_edges

 
def build_request_components(df_requests, df_responses, df_redirects, call_stacks):
    df_request_nodes = pd.DataFrame()
    df_http_edges = pd.DataFrame()

    try:
        """Function to build HTTP nodes and edges from the OpenWPM HTTP data."""
        df_requests["key"] = df_requests[["visit_id", "request_id"]].apply(
            lambda x: get_key(*x), axis=1
        )
        df_responses["key"] = df_responses[["visit_id", "request_id"]].apply(
            lambda x: get_key(*x), axis=1
        )

        df_requests["type"] = "Request"
        df_requests["attr"] = df_requests[["resource_type", "top_level_url"]].apply(
            lambda x: get_attr(*x), axis=1
        )
        # df_requests.to_csv('/home/ubuntu/df_requests_new.csv', index=False)

        # Request nodes. To be inserted
        df_request_nodes = (
            df_requests[["visit_id", "url", "type", "top_level_url", "attr", "is_in_phase1"]]
            .drop_duplicates()
            .copy()
        )
        df_request_nodes["type"] = df_requests[["type", "attr"]].apply(
            lambda x: convert_type(*x), axis=1
        )
        df_request_nodes = df_request_nodes.rename(columns={"url": "name"})


        # Redirect edges. To be inserted
        if len(df_redirects) > 0:
            df_redirects["old_request_id"] = df_redirects["old_request_id"].apply(
                lambda x: int(x)
            )
            df_redirects["key"] = df_redirects[["visit_id", "old_request_id"]].apply(
                lambda x: get_key(*x), axis=1
            )
            df_redirect_edges, completed_ids_red = get_redirect_edges(
                df_requests, df_redirects, df_responses
            )
         
        else:
            print("======== http_redirecs table is empty =========")
            completed_ids_red = []
            df_redirect_edges = pd.DataFrame()

        try:
            # check if df_cs_edges is empty or not. 
            # If it is empty, maybe due to no data in call stack table. Or no data after merge call stack and request/response table
            df_cs_edges, completed_ids_cs = get_cs_edges(
                df_requests, df_responses, call_stacks
            )

            # Other edges. To be inserted
            completed_ids = set(completed_ids_red + completed_ids_cs)
            df_normal_edges = get_normal_edges(df_requests, df_responses, completed_ids)

            df_http_edges = pd.concat(
                [df_redirect_edges, df_cs_edges, df_normal_edges]
            ).reset_index()
            del df_http_edges["index"]
            df_http_edges["action"] = pd.NA

            """
            if df_cs_edges.empty:
                raise ValueError("======== df_cs_edges is empty =========")
            else:
                # Other edges. To be inserted
                completed_ids = set(completed_ids_red + completed_ids_cs)
                df_normal_edges = get_normal_edges(df_requests, df_responses, completed_ids)

                df_http_edges = pd.concat(
                    [df_redirect_edges, df_cs_edges, df_normal_edges]
                ).reset_index()
                del df_http_edges["index"]
                df_http_edges["action"] = pd.NA
            """

        except ValueError as e1:
            print(e1)  # Optionally log the exception message

    except Exception as e:
        print("Error in request_components:", e)
        traceback.print_exc()

    return df_request_nodes, df_http_edges

