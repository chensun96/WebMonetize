import re
import json
import tldextract
from urllib.parse import urlparse
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=64)
import os
import requests

# get the top_level_url for a graph
def most_frequent_url_per_visit(df):
    # Group by 'visit_id' and then find the most frequent 'top_level_url' for each group
    most_frequent_url = df.groupby('visit_id')['top_level_url'].agg(lambda x: x.value_counts().idxmax())
    most_frequent_url_df = most_frequent_url.reset_index()
    most_frequent_url_df.columns = ['visit_id', 'top_level_url']
    return most_frequent_url



def extract_url_domain(df):
    df_document = df[(df["type"] == "Document")]
    #print("df_document:\n", df_document)
    df_edges = df[df["graph_attr"] == "Edge"]

    # Filter edges to include only rows with names that are in df_document
    df_edges = df_edges[df_edges['src'].isin(df_document['name'])]

    # Sort the edges by 'src' 'time_stamp', to get the first url triggrer the graph
    df_edges['time_stamp'] = pd.to_datetime(df_edges['time_stamp'], errors='coerce')
    df_edges_sorted = df_edges.sort_values(by=['src', 'time_stamp'])
    #print("df_edges_sorted:\n", df_edges_sorted)
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
    #print("df_url_sorted 2:\n", df_url_sorted)
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
    

    # get the top_level_url 
    # this step can remove when sqlite is exist
    df_top_level_url = most_frequent_url_per_visit(df_edges)
    df_url_sorted = df_url_sorted.merge(df_top_level_url, on='visit_id', how='left')
    df_top_domain = df_url_sorted[["visit_id", "name", "top_level_url"]]
    #print("df_top_domain: ", df_top_domain)
    return df_top_domain



