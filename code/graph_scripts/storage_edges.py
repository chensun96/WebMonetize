import pandas as pd
from .cookies import *
import json
import re
import multidict
from .utils import *
import traceback
import numpy as np


def get_storage_name(arg):
    """Function to get the storage names from the javascript table."""
    try:
        return json.loads(arg)[0]
    except Exception as e:
        return ""


def get_storage_attr(arg):
    """Function to get the storage related values from the javascript table."""

    try:
        return json.dumps({"value": json.loads(arg)[1]})
    except Exception as e:
        return ""


def get_storage_action(symbol):
    """Function to get the storage related actions from the javascript table."""

    try:
        result = re.search("Storage.(.*)Item", symbol)
        return result.group(1) + "_storage_js"
    except Exception as e:
        return ""


def get_cookie_name(val, operation):
    """Function to process cookie names from the javascript table."""

    nameval_list = []
    try:
        if operation == "get":
            cookies = val.split(";")
            for cookie in cookies:
                nameval_list.append(cookie.strip().split("="))
        elif operation == "set":
            name = ""
            value = ""
            cookie = val.split(";", 1)
            if len(cookie) > 0:
                name = cookie[0].split("=", 1)[0]
                if len(cookie) > 1:
                    value = cookie[0].split("=", 1)[1] + "; "
                    value += cookie[1]
                nameval_list.append([name, value])
    except Exception as e:
        return nameval_list
    return nameval_list


def process_cookie_call_stack(row):
    """Function to process cookie call stacks."""

    cs = row["call_stack"]
    operation = row["operation"]
    visit_id = row["visit_id"]
    ts = row["time_stamp"]
    cookie_name = row["cookie_name"]
    edge_data = []

    try:
        operation = operation + "_js"
        cs_lines = cs.split()
        urls = []
        new_urls = []
        for line in cs_lines:
            components = re.split("[@;]", line)
            if len(components) >= 2:
                urls.append(components[1].rsplit(":", 2)[0])
        urls = urls[::-1]
        for url in urls:
            if len(new_urls) == 0:
                new_urls.append(url)
            else:
                if new_urls[-1] != url:
                    new_urls.append(url)

        if len(new_urls) > 1:
            for i in range(0, len(new_urls) - 1):
                src = new_urls[i]
                dst = new_urls[i + 1]
                attr = "CS"
                op = "CS"
                edge_data.append([src, dst, op, attr, visit_id, ts])
        if len(new_urls) > 0 and len(cookie_name) > 1:
            if operation == "set_js":
                val_dict = {}
                cookie_info = cookie_name[1].split(";")
                val_dict["value"] = cookie_info[0]
                for ci in cookie_info[1:]:
                    params = ci.split("=")
                    if len(params) == 2:
                        val_dict[params[0]] = params[1]
                edge_data.append(
                    [
                        new_urls[-1],
                        cookie_name[0],
                        operation,
                        json.dumps(val_dict),
                        visit_id,
                        ts,
                    ]
                )
            else:
                edge_data.append(
                    [
                        new_urls[-1],
                        cookie_name[0],
                        operation,
                        json.dumps({"value": cookie_name[1]}),
                        visit_id,
                        ts,
                    ]
                )
    except Exception as e:
        print("Error in processing cookie call stacks", e)

    return edge_data


def build_storage_components(df_javascript):
    df_all_storage_nodes = pd.DataFrame()
    df_all_storage_edges = pd.DataFrame()

    try:
        df_storage_nodes = pd.DataFrame()
        df_storage_edges = pd.DataFrame()
        df_js_cookie_nodes = pd.DataFrame()
        df_js_cookie_edges = pd.DataFrame()

        df_storage = df_javascript[
            df_javascript["symbol"].str.contains("Storage.")
        ].copy()
        if len(df_storage) > 0:
            df_storage["name"] = df_storage["arguments"].apply(get_storage_name)
            df_storage["type"] = "Storage"
            df_storage["node_attr"] = pd.NA
            df_storage["attr"] = df_storage["arguments"].apply(get_storage_attr)
            df_storage["action"] = df_storage["symbol"].apply(get_storage_action)

            # df_owners_storage = utils.get_original_cookie_setters(df_storage)
            # if len(df_owners_storage) > 0:
            #     df_owners_storage = df_owners_storage.rename(
            #         columns={"script_url": "setter", "time_stamp": "setting_time_stamp"})
            #     df_owners_storage = df_owners_storage[[
            #         'visit_id', 'name', 'setter', 'setting_time_stamp']]

            # df_storage = df_storage.merge(
            #     df_owners_storage, on=['visit_id', 'name'], how='outer')

            # To be inserted
            df_storage["domain"] = df_storage["document_url"].apply(get_domain)
            df_storage["storage_key"] = df_storage[["name", "domain"]].apply(
                lambda x: get_cookiedom_key(*x), axis=1
            )
            df_storage["storage_key"] = df_storage["storage_key"].apply(
                lambda x: str(x) + "|$$|LS"
            )
            df_storage_nodes = df_storage[
                [
                    "visit_id",
                    "storage_key",
                    "type",
                    "node_attr",
                    "document_url",
                    "domain",
                    "top_level_url",
                    "is_in_phase1",
                ]
            ].drop_duplicates()
            df_storage_nodes = df_storage_nodes.rename(
                columns={"node_attr": "attr", "storage_key": "name"}
            )
            df_storage_edges = df_storage[
                [
                    "visit_id",
                    "script_url",
                    "storage_key",
                    "top_level_url",
                    "action",
                    "attr",
                    "time_stamp",
                    "is_in_phase1",
                ]
            ]
            df_storage_edges = df_storage_edges.rename(columns={"storage_key": "name"})
            
        df_js_cookie = df_javascript[
            df_javascript["symbol"] == "window.document.cookie"
        ].copy()
        #df_js_cookie.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_js_cookie.csv', index=False)
            

        if len(df_js_cookie) > 0:
            df_js_cookie["cookie_name"] = df_js_cookie[["value", "operation"]].apply(
                lambda x: get_cookie_name(*x), axis=1
            )
            df_js_cookie = df_js_cookie.explode("cookie_name")[
                [
                    "call_stack",
                    "cookie_name",
                    "operation",
                    "time_stamp",
                    "document_url",
                    "top_level_url",
                    "visit_id",
                    "is_in_phase1"
                ]
            ]
            #df_js_cookie.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_js_cookie_before_process_cookie_call_stack.csv', index=False)
        
            df_js_cookie["cs_edges"] = df_js_cookie.apply(
                process_cookie_call_stack, axis=1
            )

            if (df_js_cookie['cs_edges'].apply(lambda x: x == []).any()):
                print("There is at least one row with an empty list in 'cs_edges'.")
                df_js_cookie_nodes = pd.DataFrame()
                df_js_cookie_edges = pd.DataFrame()
            else:

                #df_js_cookie.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_js_cookie_after_process_cookie_call_stack.csv', index=False)
            
                df_js_cookie = (
                    df_js_cookie[["cs_edges", "document_url", "top_level_url", "is_in_phase1"]]
                    .explode("cs_edges")
                    .dropna()
                )
                df_js_cookie["script_url"] = df_js_cookie["cs_edges"].apply(lambda x: x[0])
                df_js_cookie["name"] = df_js_cookie["cs_edges"].apply(lambda x: x[1])
                df_js_cookie["action"] = df_js_cookie["cs_edges"].apply(lambda x: x[2])
                df_js_cookie["attr"] = df_js_cookie["cs_edges"].apply(lambda x: x[3])
                df_js_cookie["visit_id"] = df_js_cookie["cs_edges"].apply(lambda x: x[4])
                df_js_cookie["time_stamp"] = df_js_cookie["cs_edges"].apply(lambda x: x[5])

                #df_js_cookie.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_js_cookie_2.csv', index=False)
            

                # df_owners = utils.get_original_cookie_setters(df_js_cookie)
                # if len(df_owners) > 0:
                #     df_owners = df_owners.rename(
                #         columns={"script_url": "setter", "time_stamp": "setting_time_stamp"})
                #     df_owners = df_owners[['visit_id', 'name', 'setter', 'setting_time_stamp']]

                # df_js_cookie = df_js_cookie.merge(
                #     df_owners, on=['visit_id', 'name'], how='outer')

                # To be inserted
                df_js_cookie_edges = df_js_cookie[
                    [
                        "visit_id",
                        "script_url",
                        "document_url",
                        "top_level_url",
                        "name",
                        "action",
                        "attr",
                        "time_stamp",
                        "is_in_phase1",
                    ]
                ]
                df_js_cookie_edges["domain"] = df_js_cookie_edges["document_url"].apply(
                    get_domain
                )

                #df_js_cookie_edges.to_csv('/home/data/chensun/affi_project/purl/output/ads/crawl_test_buildGraph/df_js_cookie_before_cookie_key.csv', index=False)
            
                df_js_cookie_edges["cookie_key"] = df_js_cookie_edges[
                    ["name", "domain"]
                ].apply(lambda x: get_cookiedom_key(*x), axis=1)

                # To be inserted
                df_js_cookie_nodes = df_js_cookie_edges[
                    df_js_cookie_edges["action"] != "CS"
                ][
                    ["visit_id", "cookie_key", "top_level_url", "document_url", "domain", "is_in_phase1"]
                ].drop_duplicates()
                df_js_cookie_nodes = df_js_cookie_nodes.rename(
                    columns={"cookie_key": "name"}
                )
                df_js_cookie_nodes["type"] = "Storage"
                df_js_cookie_nodes["attr"] = "Cookie"

                # [Added]
                # remove all the row from js_cookie_edges where js_cookie_edges["action"] == CS
                df_js_cookie_edges = df_js_cookie_edges[df_js_cookie_edges["action"] != "CS"]

                df_js_cookie_edges = df_js_cookie_edges[
                    [
                        "visit_id",
                        "script_url",
                        "cookie_key",
                        "top_level_url",
                        "action",
                        "attr",
                        "time_stamp",
                        "is_in_phase1",
                    ]
                ]
                df_js_cookie_edges = df_js_cookie_edges.rename(
                    columns={"cookie_key": "name"}
                )
            
        df_all_storage_nodes = pd.concat([df_storage_nodes, df_js_cookie_nodes])
        df_all_storage_edges = pd.concat([df_storage_edges, df_js_cookie_edges])
        df_all_storage_edges = df_all_storage_edges.rename(
            columns={"script_url": "src", "name": "dst"}
        )
        df_all_storage_edges["reqattr"] = pd.NA
        df_all_storage_edges["respattr"] = pd.NA
        df_all_storage_edges["response_status"] = pd.NA
        df_all_storage_edges["post_body"] = pd.NA
        df_all_storage_edges["post_body_raw"] = pd.NA
        # df_all_storage_edges['time_stamp'] = pd.NA
        # df_all_storage_edges['attr'] = pd.NA

    except Exception as e:
        traceback.print_exc()
        print("Error in storage_components:", e)

    return df_all_storage_nodes, df_all_storage_edges
