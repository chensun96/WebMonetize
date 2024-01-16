import json
import pandas as pd
import networkx as nx
import numpy as np
import re
import os
import D3_graph_html as d3


def read_html(html_path):
    if os.path.getsize(html_path) > 0:  # Check if the file is not empty
        with open(html_path, 'r', encoding='utf-8') as file:  # You can change encoding if needed
            content = file.read()
            #print(content)  # Print the content to check
            return content
    else:
        print("The file is empty.")
        return ""
    

def generate_html_from_json(json_data, D3_html):
    # Your logic to insert JSON data into the HTML template goes here
    # For example, you might replace some placeholder in the template
    # with data extracted from json_data
    # Example: template = template.replace('<!-- JSON_PLACEHOLDER -->', str(json_data))
    return D3_html  # Return the modified HTML content


def build_d3_graph(pdf, visit_id, des_file_path, html_path):
    """
    Function to prepare data for D3.js visualization from a Pandas DataFrame.

    Args:
        pdf: DataFrame of nodes and edges.
    Returns:
        graph_json: JSON data for nodes and links suitable for D3.js.

    This function does the following:

    1. Selects nodes and edges.
    2. Processes node attributes.
    3. Prepares JSON data for D3.js.
    """
    # Processing nodes
    df_nodes = pdf[(pdf["graph_attr"] == "Node") | (pdf["graph_attr"] == "NodeWG")]
    df_nodes = df_nodes.groupby(['visit_id', 'name'], as_index=False)\
                       .agg({'type': lambda x: list(x),
                             'attr': lambda x: list(x),
                             'domain': lambda x: list(x)[0],
                             'top_level_domain': lambda x: list(x)[0],
                             'is_in_phase1': lambda x: x.iloc[0]})
    df_nodes = df_nodes.rename(columns={'name': 'id'})
    df_nodes['type'] = df_nodes['type'].apply(modify_type)
    df_nodes['attr'] = df_nodes['attr'].apply(modify_attr)
    # Replace NaN in 'domain' with None (which becomes null in JSON)
    df_nodes['domain'] = np.where(df_nodes['domain'].isna(), None, df_nodes['domain'])


    # Converting nodes to a list of dictionaries
    nodes = df_nodes.to_dict('records')

    # Processing edges
    df_edges = pdf[(pdf["graph_attr"] == "Edge") | (pdf["graph_attr"] == "EdgeWG")]
    df_edges = df_edges.rename(columns={'src': 'source', 'dst': 'target'})
    # Converting edges to a list of dictionaries
    # links = df_edges[['source', 'target', 'is_in_phase1']].to_dict('records')
    df_edges = df_edges[['source', 'target', 'is_in_phase1']]
    links = df_edges.to_dict('records')

    # Combine nodes and links into a single dictionary
    graph = {'nodes': nodes, 'links': links}

    # Convert to JSON
    graph_json = json.dumps(graph)

    filename = "graph_full.json"
    fullGraph_path = os.path.join(des_file_path, filename)
    print("fullGraph_path: ", fullGraph_path)
    with open(fullGraph_path, 'w') as file:
        file.write(graph_json)

    # Construct HTML
    html_name = 'graph_full.html'
    D3_html_path = os.path.join(html_path, html_name)
    print("D3_html_path: ", D3_html_path)

    D3_html = read_html(D3_html_path)

    #html_content = generate_html_from_json(graph_json, D3_html)
    html_file_name = filename.replace('.json', '.html')
    html_path = os.path.join(des_file_path, html_file_name)

    with open(html_path, 'w') as html_file:
        html_file.write(D3_html)

    return graph_json

def build_d3_redirect(pdf, visit_id, des_file_path, html_path):
    """
    Function to prepare data for D3.js visualization from a Pandas DataFrame.

    Args:
        pdf: DataFrame of nodes and edges.
    Returns:
        graph_json: JSON data for nodes and links suitable for D3.js.

    This function does the following:

    1. Selects nodes and edges.
    2. Processes node attributes.
    3. Prepares JSON data for D3.js.
    """
    # Processing nodes
    df_nodes = pdf[(pdf["graph_attr"] == "Node") | (pdf["graph_attr"] == "NodeWG")]
    df_nodes = df_nodes.groupby(['visit_id', 'name'], as_index=False)\
                       .agg({'type': lambda x: list(x),
                             'attr': lambda x: list(x),
                             'domain': lambda x: list(x)[0],
                             'top_level_domain': lambda x: list(x)[0],
                             'is_in_phase1': lambda x: x.iloc[0]})
    df_nodes = df_nodes.rename(columns={'name': 'id'})
    df_nodes['type'] = df_nodes['type'].apply(modify_type)
    df_nodes['attr'] = df_nodes['attr'].apply(modify_attr)
    # Replace NaN in 'domain' with None (which becomes null in JSON)
    df_nodes['domain'] = np.where(df_nodes['domain'].isna(), None, df_nodes['domain'])
    df_nodes = df_nodes[df_nodes['is_in_phase1'] == True]

    # Converting nodes to a list of dictionaries
    nodes = df_nodes.to_dict('records')

    # Processing edges
    df_edges = pdf[(pdf["graph_attr"] == "Edge") | (pdf["graph_attr"] == "EdgeWG")]
    df_edges = df_edges.rename(columns={'src': 'source', 'dst': 'target'})
    # Converting edges to a list of dictionaries
    # links = df_edges[['source', 'target', 'is_in_phase1']].to_dict('records')
    df_edges = df_edges[['source', 'target', 'is_in_phase1']]
    df_edges = df_edges[df_edges['is_in_phase1'] == True]
    links = df_edges.to_dict('records')

    # Combine nodes and links into a single dictionary
    graph = {'nodes': nodes, 'links': links}

    # Convert to JSON
    graph_json = json.dumps(graph)

    filename = "graph_redirect.json"
    phaseA_path = os.path.join(des_file_path, filename)
    print("phaseA_path: ", phaseA_path)
    with open(phaseA_path, 'w') as file:
        file.write(graph_json)

     # Construct HTML
    html_name = 'graph_redirect.html'
    D3_html_redirect_path = os.path.join(html_path, html_name)
    print("D3_html_path: ", D3_html_redirect_path)

    D3_html = read_html(D3_html_redirect_path)

    #html_content = generate_html_from_json(graph_json, D3_html)
    html_file_name = filename.replace('.json', '.html')
    html_path = os.path.join(des_file_path, html_file_name)

    with open(html_path, 'w') as html_file:
        html_file.write(D3_html)

    return graph_json

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


def apply_tasks(
    df,
    visit_id,
    graph_folder,
    html_path,
):
    
    # Call build_graph function with the filtered DataFrame
    data_folder = "D3_graph/visit_data_" + str(visit_id)
    des_file_path = os.path.join(graph_folder, data_folder)
    os.makedirs(des_file_path, exist_ok=True)

    build_d3_graph(df, visit_id, des_file_path,html_path)

    build_d3_redirect(df, visit_id, des_file_path,html_path)



if __name__ == "__main__":
    html_path = 'D3_graph_html'

    url_type = ["affiliate"]
    #url_type = ['ads']
    for type_name in url_type:

        full_graph_folder = f"../output/{type_name}/fullGraph"
        #phaseA_folder = f"../output/{type_name}/phaseA"

        for filename in os.listdir(full_graph_folder):
            if filename.startswith("graph_") and filename.endswith(".csv"):
                # get the graph_{i}.csv 
                fullGraph_path = os.path.join(full_graph_folder, filename)
                
                df_graph = pd.read_csv(fullGraph_path)
                #print("df_graph: ", df_graph)

                visit_ids_list = df_graph['visit_id'].unique().tolist()
                for visit_id in visit_ids_list:
                    print("\n=================== visit_id: " + str(visit_id) + " ==============================")
                    df_one_graph = df_graph[df_graph['visit_id'] == visit_id]
                    df_one_graph.groupby(["visit_id"]).apply(
                                apply_tasks,
                                visit_id,
                                full_graph_folder,
                                html_path,
                            )
