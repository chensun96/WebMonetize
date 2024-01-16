import json
import networkx as nx
import numpy as np    
from statistics import median
from statistics import stdev

def graph_info(G, visit_id, max_in_degree,max_out_degree,
               connected_components, largest_cc, smallest_cc, 
               max_avg_path_length, component_with_max_path, 
                degree_centrality, max_degree_centrality, min_degree_centrality,
                closeness_centrality, max_closeness_centrality, min_closeness_centrality,
                closeness_centrality_outward, max_closeness_centrality_outward, min_closeness_centrality_outward,
                graph_type, graph_path):
    number_of_ccs = len(connected_components)

    if graph_type == 'phase1':
        txt_path = graph_path.replace("graph_", "graph_analysis_phase1_").replace(".csv", ".txt")
    else:
        txt_path = graph_path.replace("graph_", "graph_analysis_fullGraph_").replace(".csv", ".txt")

    with open(txt_path, 'a') as file:
        file.write(f"\n=============== Graph Analysis for Visit ID: {visit_id} =====================\n")


        # Find nodes with in and out degrees
        file.write(f"\nMaximum In-Degree: {max_in_degree}\n")
        for node, degree in G.in_degree():
            if degree == max_in_degree:
                file.write(f"\tNode: {node}\n")
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    node_attr = G.nodes[node].get('attr', 'N/A')  
                    file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")

        file.write(f"\nMaximum Out-Degree: {max_out_degree}\n")
        for node, degree in G.out_degree():
            if degree == max_out_degree:
                file.write(f"\tNode: {node}\n")
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    node_attr = G.nodes[node].get('attr', 'N/A')  
                    file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")

        file.write(f"\nMaximum Closeness Centrality (inward): {max_closeness_centrality}\n")
        for node, centrality in closeness_centrality.items():
            if centrality == max_closeness_centrality:
                file.write(f"\tNode: {node}\n")
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    node_attr = G.nodes[node].get('attr', 'N/A')  
                    file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")

        file.write(f"\nMinimum Closeness Centrality (inward): {min_closeness_centrality}\n")    
        for node, centrality in closeness_centrality.items():
            if centrality == min_closeness_centrality:
                file.write(f"\tNode: {node}\n")
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    node_attr = G.nodes[node].get('attr', 'N/A')  
                    file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")


        file.write(f"\nMaximum Closeness Centrality (outward): {max_closeness_centrality_outward}\n")
        for node, centrality in closeness_centrality_outward.items():
            if centrality == max_closeness_centrality_outward:
                file.write(f"\tNode: {node}\n")
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    node_attr = G.nodes[node].get('attr', 'N/A')  
                    file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")


        file.write(f"\nMinimum Closeness Centrality (outward): {min_closeness_centrality_outward}\n")    
        for node, centrality in closeness_centrality_outward.items():
            if centrality == min_closeness_centrality_outward:
                file.write(f"\tNode: {node}\n")
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    node_attr = G.nodes[node].get('attr', 'N/A')  
                    file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")


        file.write(f"\nMaximum Degree Centrality: {max_degree_centrality}\n")
        for node, centrality in degree_centrality.items():
            if centrality == max_degree_centrality:
                file.write(f"\tNode: {node}\n")
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    node_attr = G.nodes[node].get('attr', 'N/A')  
                    file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")


        file.write(f"\nMinimum Degree Centrality: {min_degree_centrality}\n")    
        for node, centrality in degree_centrality.items():
            if centrality == min_degree_centrality:
                file.write(f"\tNode: {node}\n")
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    node_attr = G.nodes[node].get('attr', 'N/A')  
                    file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")


        file.write(f"\nNumber of connected components: {number_of_ccs}\n")
        file.write("\nComponent with Maximum Average Shortest Path Length:\n")
        file.write(f"\t\tLength: {max_avg_path_length}), Component Size: {len(component_with_max_path)}\n")

        file.write(f"\nLargest Connected Component (Size: {len(largest_cc)}):\n")
        for node in largest_cc:
            file.write(f"\tNode: {node}\n")
            node_type = G.nodes[node].get('type', 'N/A')  
            if node_type == "Document" or node_type == "Request" or node_type == "Script":
                node_attr = G.nodes[node].get('attr', 'N/A')  
                file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")


        if number_of_ccs > 1:
            file.write(f"\nSmallest Connected Component (Size: {len(smallest_cc)}):\n")
            for node in smallest_cc:
                file.write(f"\tNode: {node}\n")
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    node_attr = G.nodes[node].get('attr', 'N/A')  
                    file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")

            file.write("\nOther Connected Components:\n")
            for cc in connected_components:
                if cc != largest_cc and cc != smallest_cc:
                    file.write(f"\nConnected Component (Size: {len(cc)}):\n")
                    for node in cc:
                        file.write(f"\tNode: {node}\n")
                        node_type = G.nodes[node].get('type', 'N/A')  
                        if node_type == "Document" or node_type == "Request" or node_type == "Script":
                            node_attr = G.nodes[node].get('attr', 'N/A')  
                            file.write(f"\t\tType: {node_type}, Attr: {node_attr}\n")
                    file.write("\n")
        file.write("\n")

def extract_full_graph_features(G, visit_id, graph_type, graph_path): 
    node_feature = []
    connectivity_feature = []
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    in_degrees = [d for n, d in G.in_degree()]  # List of in-degrees of all nodes
    out_degrees = [d for n, d in G.out_degree()] # List of out-degrees of all nodes

    average_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    average_in_degree = sum(d for n, d in G.in_degree()) / G.number_of_nodes()
    average_out_degree = sum(d for n, d in G.out_degree()) / G.number_of_nodes()

    median_in_degree = np.median(in_degrees)
    median_out_degree = np.median(out_degrees)

    # Calculate maximum and minimum degrees
    max_in_degree = max(in_degrees)
    max_out_degree = max(out_degrees)
   

    node_feature = [num_nodes, num_edges, average_degree, \
                    average_in_degree, average_out_degree, median_in_degree, median_out_degree, \
                    max_in_degree, max_out_degree]
    node_feature_name = ['num_nodes', 'num_edges', 'average_degree', \
                    'average_in_degree', 'average_out_degree', 'median_in_degree', 'median_out_degree',\
                     'max_in_degree', 'max_out_degree']
    
   
    """
    try:
        if nx.is_connected(G.to_undirected()):
            avg_path_length = nx.average_shortest_path_length(G)
            print("Average Path Length:", avg_path_length)
        else:
            print("Graph is not connected, average path length is undefined.")
    except Exception as e:
        print("Error calculating average path length:", e)

    #diameter = nx.diameter(G)  # failed since the graph is not connected
    """

    density = nx.density(G)

    G_undirected = G.to_undirected()

    avg_clustering_coefficient = nx.average_clustering(G)
    transitivity = nx.transitivity(G)


    #print("Number of nodes:", num_nodes)
    #print("Number of edges:", num_edges)
    #print("average_degree: ", average_degree)
    #print("density: ", density)
    #print("diameter: ", diameter)
    
    #print("avg_clustering_coefficient: ", avg_clustering_coefficient)
    #print("transitivity: ", transitivity)

    # need take an undirected graph for connect_component
    connected_components = list(nx.connected_components(G_undirected))

    number_of_ccs = len(connected_components)
    average_size_cc = sum(len(c) for c in connected_components) / number_of_ccs

    largest_cc = len(max(connected_components, key=len))
    largest_cc_nodes = max(connected_components, key=len)

    smallest_cc = len(min(connected_components, key=len))
    smallest_cc_nodes = min(connected_components, key=len)

    cc_sizes = [len(c) for c in connected_components]
    
    average_path_length_list = []
    for component in connected_components:
        subgraph = G.subgraph(component)
        average_path_length = nx.average_shortest_path_length(subgraph.to_undirected()) if len(component) > 1 else 0
        average_path_length_list.append((average_path_length, component))

    # Find the component with maximum average shortest path length
    max_avg_path_length, component_with_max_path = max(average_path_length_list, key=lambda x: x[0])

    degree_centrality = nx.degree_centrality(G)
    average_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)
    median_degree_centrality = median(degree_centrality.values())
    max_degree_centrality = max(degree_centrality.values())
    min_degree_centrality = min(degree_centrality.values())
    std_dev_degree_centrality = stdev(degree_centrality.values())

    # inward distance
    closeness_centrality = nx.closeness_centrality(G)
    average_closeness_centrality = sum(closeness_centrality.values()) / len(closeness_centrality)
    median_closeness_centrality = median(closeness_centrality.values())
    max_closeness_centrality = max(closeness_centrality.values())
    min_closeness_centrality = min(closeness_centrality.values())
    std_dev_closeness_centrality = stdev(closeness_centrality.values())

    # outward distance
    closeness_centrality_outward = nx.closeness_centrality(G.reverse() )
    average_closeness_centrality_outward = sum(closeness_centrality_outward.values()) / len(closeness_centrality_outward)
    median_closeness_centrality_outward = median(closeness_centrality_outward.values())
    max_closeness_centrality_outward = max(closeness_centrality_outward.values())
    min_closeness_centrality_outward = min(closeness_centrality_outward.values())
    std_dev_closeness_centrality_outward = stdev(closeness_centrality_outward.values())

    graph_info(G, visit_id, max_in_degree, max_out_degree,
               connected_components, largest_cc_nodes, smallest_cc_nodes, 
                max_avg_path_length, component_with_max_path, 
                degree_centrality, max_degree_centrality, min_degree_centrality, 
                closeness_centrality, max_closeness_centrality, min_closeness_centrality,
                closeness_centrality_outward, max_closeness_centrality_outward, min_closeness_centrality_outward,
                graph_type, graph_path)


    connectivity_feature = [density, avg_clustering_coefficient, transitivity, number_of_ccs, \
                             average_size_cc, largest_cc, smallest_cc, max_avg_path_length, \
                            average_degree_centrality, median_degree_centrality, max_degree_centrality, \
                            min_degree_centrality, std_dev_degree_centrality, \
                            average_closeness_centrality, median_closeness_centrality, max_closeness_centrality, \
                            min_closeness_centrality, std_dev_closeness_centrality, \
                            average_closeness_centrality_outward, median_closeness_centrality_outward,\
                            max_closeness_centrality_outward, min_closeness_centrality_outward, std_dev_closeness_centrality_outward]       
    
    connectivity_feature_names = ['density', 'avg_clustering_coefficient', 'transitivity', 'number_of_ccs', \
                             'average_size_cc', 'largest_cc', 'smallest_cc', 'max_avg_path_length', \
                            'average_degree_centrality', 'median_degree_centrality', 'max_degree_centrality', \
                            'min_degree_centrality', 'std_dev_degree_centrality',  \
                            'average_closeness_centrality', 'median_closeness_centrality', 'max_closeness_centrality', \
                            'min_closeness_centrality', 'std_dev_closeness_centrality', \
                            'average_closeness_centrality_outward', 'median_closeness_centrality_outward',\
                            'max_closeness_centrality_outward', 'min_closeness_centrality_outward', 'std_dev_closeness_centrality_outward']       
    
    return node_feature, node_feature_name, connectivity_feature, connectivity_feature_names




