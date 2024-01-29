import json
import networkx as nx
import numpy as np    
from statistics import median
from statistics import stdev

def graph_info_to_csv(G, visit_id, max_in_degree,max_out_degree,
               connected_components, largest_cc, smallest_cc, 
               max_avg_path_length, component_with_max_path, 
                degree_centrality, max_degree_centrality, min_degree_centrality,
                closeness_centrality, max_closeness_centrality, min_closeness_centrality,
                closeness_centrality_outward, max_closeness_centrality_outward, min_closeness_centrality_outward,
                average_path_length_for_largest_cc, graph_type, graph_path):
    
    number_of_ccs = len(connected_components)
    #if graph_type == 'phase1':
    #    csv_path = graph_path.replace("graph_", "graph_analysis_phase1_")
    #else:
    #    csv_path = graph_path.replace("graph_", "graph_analysis_fullGraph_")

    # if csv_path exist, append


    graph_info = {'visit_id': visit_id}
    max_in_degree_nodes = []
    max_out_degree_nodes = []
    max_closeness_centrality_inward_nodes = []
    max_closeness_centrality_outward_nodes = []
    max_degree_centrality_nodes = []
    min_degree_centrality_nodes = []
    largest_cc_nodes = []


    for node, degree in G.in_degree():
        if degree == max_in_degree:
            node_type = G.nodes[node].get('type', 'N/A')  
            if node_type == "Document" or node_type == "Request" or node_type == "Script":
                node_attr = G.nodes[node].get('attr', 'N/A')  
                node_data = {
                'node': node,
                'type': node_type
                }
                max_in_degree_nodes.append(node_data) 
    graph_info['max_in_degree'] = max_in_degree_nodes


    for node, degree in G.out_degree():
        if degree == max_out_degree:
            node_type = G.nodes[node].get('type', 'N/A')  
            if node_type == "Document" or node_type == "Request" or node_type == "Script":
                node_attr = G.nodes[node].get('attr', 'N/A')  
                node_data = {
                'node': node,
                'type': node_type
                }
                max_out_degree_nodes.append(node_data)  
    graph_info['max_out_degree'] = max_out_degree_nodes

    for node, centrality in closeness_centrality.items():
        if centrality == max_closeness_centrality: 
            node_type = G.nodes[node].get('type', 'N/A')  
            if node_type == "Document" or node_type == "Request" or node_type == "Script":
                node_attr = G.nodes[node].get('attr', 'N/A')  
                node_data = {
                'node': node,
                'type': node_type
                }
                max_closeness_centrality_inward_nodes.append(node_data)
    graph_info['max_closeness_centrality_inward'] = max_closeness_centrality_inward_nodes
               
    for node, centrality in closeness_centrality_outward.items():
        if centrality == max_closeness_centrality_outward:
            node_type = G.nodes[node].get('type', 'N/A')  
            if node_type == "Document" or node_type == "Request" or node_type == "Script":
                node_attr = G.nodes[node].get('attr', 'N/A')  
                node_data = {
                'node': node,
                'type': node_type
                }
                max_closeness_centrality_outward_nodes.append(node_data)
    graph_info['max_closeness_centrality_outward'] = max_closeness_centrality_outward_nodes
               
    for node, centrality in degree_centrality.items():
        if centrality == max_degree_centrality:
            node_type = G.nodes[node].get('type', 'N/A')  
            if node_type == "Document" or node_type == "Request" or node_type == "Script":
                node_data = {
                'node': node,
                'type': node_type
                }
                max_degree_centrality_nodes.append(node_data)            
    
        if centrality == min_degree_centrality:
            node_type = G.nodes[node].get('type', 'N/A')  
            if node_type == "Document" or node_type == "Request" or node_type == "Script":
                node_data = {
                'node': node,
                'type': node_type
                }
                min_degree_centrality_nodes.append(node_data) 
    graph_info['max_degree_centrality'] = max_degree_centrality_nodes
    graph_info['min_degree_centrality'] = min_degree_centrality_nodes

   
    graph_info['max_avg_path_length'] = max_avg_path_length
    graph_info['number of cc'] = number_of_ccs
    graph_info['component with max_avg_path_length'] = len(component_with_max_path)

       
    for node in largest_cc:
        node_type = G.nodes[node].get('type', 'N/A')  
        if node_type == "Document" or node_type == "Request" or node_type == "Script":
           if node_type == "Document" or node_type == "Request" or node_type == "Script":
                node_data = {
                'node': node,
                'type': node_type
                }
                largest_cc_nodes.append(node_data) 
    graph_info['largest_cc'] = len(largest_cc)
    graph_info['largest_cc_nods'] = largest_cc_nodes                   

def graph_info_full_data_in_txt(G, visit_id, max_in_degree,max_out_degree,
               connected_components, largest_cc, smallest_cc, 
               max_avg_path_length, component_with_max_path, 
                degree_centrality, max_degree_centrality, min_degree_centrality,
                closeness_centrality, max_closeness_centrality, min_closeness_centrality,
                closeness_centrality_outward, max_closeness_centrality_outward, min_closeness_centrality_outward,
                average_path_length_for_largest_cc, graph_type, graph_path):
    number_of_ccs = len(connected_components)

    if graph_type == 'phase1':
        #txt_path = graph_path.replace("graph_", "graph_analysis_phase1_").replace(".csv", ".txt")
        txt_path = graph_path.replace("graph", "graph_analysis_phase1").replace(".csv", ".txt")
    else:
        #txt_path = graph_path.replace("graph_", "graph_analysis_fullGraph_").replace(".csv", ".txt")
        txt_path = graph_path.replace("graph", "graph_analysis_fullGraph").replace(".csv", ".txt")

    with open(txt_path, 'a') as file:
        file.write(f"\n=============== Graph Analysis for Visit ID: {visit_id} =====================\n")

        # Find nodes with in and out degrees
        file.write(f"\nMaximum In-Degree: {max_in_degree}\n")
        decoration_count = 0
        element_count = 0
        storage_count = 0
        for node, degree in G.in_degree():
            if degree == max_in_degree:
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    file.write(f"\t\t{node_type}: {node}\n")
                elif node_type == "Storage":
                    storage_count += 1
                elif node_type == "Element":
                    element_count += 1
                else:
                    decoration_count += 1
        if storage_count > 0:
            file.write(f"\t\tStorage: {storage_count}\n")
        if element_count > 0:
            file.write(f"\t\tElement: {element_count}\n")
        if decoration_count > 0:
            file.write(f"\t\tDecoration: {decoration_count}\n")
                

        file.write(f"\nMaximum Out-Degree: {max_out_degree}\n")
        decoration_count = 0
        element_count = 0
        storage_count = 0
        for node, degree in G.out_degree():
            if degree == max_out_degree:
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    file.write(f"\t\t{node_type}: {node}\n")
                elif node_type == "Storage":
                    storage_count += 1
                elif node_type == "Element":
                    element_count += 1
                else:
                    decoration_count += 1
        if storage_count > 0:
            file.write(f"\t\tStorage: {storage_count}\n")
        if element_count > 0:
            file.write(f"\t\tElement: {element_count}\n")
        if decoration_count > 0:
            file.write(f"\t\tDecoration: {decoration_count}\n")


        file.write(f"\nMaximum Closeness Centrality (inward): {max_closeness_centrality}\n")
        decoration_count = 0
        element_count = 0
        storage_count = 0
        for node, centrality in closeness_centrality.items():
            if centrality == max_closeness_centrality:
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    file.write(f"\t\t{node_type}: {node}\n")
                elif node_type == "Storage":
                    storage_count += 1
                elif node_type == "Element":
                    element_count += 1
                else:
                    decoration_count += 1
        if storage_count > 0:
            file.write(f"\t\tStorage: {storage_count}\n")
        if element_count > 0:
            file.write(f"\t\tElement: {element_count}\n")
        if decoration_count > 0:
            file.write(f"\t\tDecoration: {decoration_count}\n")


        file.write(f"\nMinimum Closeness Centrality (inward): {min_closeness_centrality}\n")   
        decoration_count = 0
        element_count = 0
        storage_count = 0 
        for node, centrality in closeness_centrality.items():
            if centrality == min_closeness_centrality:
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    file.write(f"\t\t{node_type}: {node}\n")
                elif node_type == "Storage":
                    storage_count += 1
                elif node_type == "Element":
                    element_count += 1
                else:
                    decoration_count += 1
        if storage_count > 0:
            file.write(f"\t\tStorage: {storage_count}\n")
        if element_count > 0:
            file.write(f"\t\tElement: {element_count}\n")
        if decoration_count > 0:
            file.write(f"\t\tDecoration: {decoration_count}\n")


        file.write(f"\nMaximum Closeness Centrality (outward): {max_closeness_centrality_outward}\n")
        decoration_count = 0
        element_count = 0
        storage_count = 0 
        for node, centrality in closeness_centrality_outward.items():
            if centrality == max_closeness_centrality_outward:
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    file.write(f"\t\t{node_type}: {node}\n")
                elif node_type == "Storage":
                    storage_count += 1
                elif node_type == "Element":
                    element_count += 1
                else:
                    decoration_count += 1
        if storage_count > 0:
            file.write(f"\t\tStorage: {storage_count}\n")
        if element_count > 0:
            file.write(f"\t\tElement: {element_count}\n")
        if decoration_count > 0:
            file.write(f"\t\tDecoration: {decoration_count}\n")


        file.write(f"\nMinimum Closeness Centrality (outward): {min_closeness_centrality_outward}\n")
        decoration_count = 0
        element_count = 0
        storage_count = 0     
        for node, centrality in closeness_centrality_outward.items():
            if centrality == min_closeness_centrality_outward:
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    file.write(f"\t\t{node_type}: {node}\n")
                elif node_type == "Storage":
                    storage_count += 1
                elif node_type == "Element":
                    element_count += 1
                else:
                    decoration_count += 1
        if storage_count > 0:
            file.write(f"\t\tStorage: {storage_count}\n")
        if element_count > 0:
            file.write(f"\t\tElement: {element_count}\n")
        if decoration_count > 0:
            file.write(f"\t\tDecoration: {decoration_count}\n")
                  

        file.write(f"\nMaximum Degree Centrality: {max_degree_centrality}\n")
        decoration_count = 0
        element_count = 0
        storage_count = 0     
        for node, centrality in degree_centrality.items():
            if centrality == max_degree_centrality:
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    file.write(f"\t\t{node_type}: {node}\n")
                elif node_type == "Storage":
                    storage_count += 1
                elif node_type == "Element":
                    element_count += 1
                else:
                    decoration_count += 1
        if storage_count > 0:
            file.write(f"\t\tStorage: {storage_count}\n")
        if element_count > 0:
            file.write(f"\t\tElement: {element_count}\n")
        if decoration_count > 0:
            file.write(f"\t\tDecoration: {decoration_count}\n")


        file.write(f"\nMinimum Degree Centrality: {min_degree_centrality}\n")    
        decoration_count = 0
        element_count = 0
        storage_count = 0  
        for node, centrality in degree_centrality.items():
            if centrality == min_degree_centrality:
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    file.write(f"\t\t{node_type}: {node}\n")
                elif node_type == "Storage":
                    storage_count += 1
                elif node_type == "Element":
                    element_count += 1
                else:
                    decoration_count += 1
        if storage_count > 0:
            file.write(f"\t\tStorage: {storage_count}\n")
        if element_count > 0:
            file.write(f"\t\tElement: {element_count}\n")
        if decoration_count > 0:
            file.write(f"\t\tDecoration: {decoration_count}\n")



        file.write(f"\nNumber of connected components: {number_of_ccs}\n")

        file.write("\nComponent with Maximum Average Shortest Path Length:\n")
        file.write(f"\t\tLength: {max_avg_path_length}, Component Size: {len(component_with_max_path)}\n")

        file.write("\nAverage Shortest Path Length for Largest Component:\n")
        file.write(f"\t\tLength: {average_path_length_for_largest_cc}, Largest Component Size: {len(largest_cc)}\n")

        file.write(f"\nLargest Connected Component (Size: {len(largest_cc)}):\n")
        decoration_count = 0
        element_count = 0
        storage_count = 0  
        for node in largest_cc:
            node_type = G.nodes[node].get('type', 'N/A')  
            if node_type == "Document" or node_type == "Request" or node_type == "Script":
                file.write(f"\t\t{node_type}: {node}\n")
            elif node_type == "Storage":
                storage_count += 1
            elif node_type == "Element":
                element_count += 1
            else:
                decoration_count += 1
        if storage_count > 0:
            file.write(f"\t\tStorage: {storage_count}\n")
        if element_count > 0:
            file.write(f"\t\tElement: {element_count}\n")
        if decoration_count > 0:
            file.write(f"\t\tDecoration: {decoration_count}\n")
             

        if number_of_ccs > 1:
            decoration_count = 0
            element_count = 0
            storage_count = 0  
            file.write(f"\nSmallest Connected Component (Size: {len(smallest_cc)}):\n")
            for node in smallest_cc:
                node_type = G.nodes[node].get('type', 'N/A')  
                if node_type == "Document" or node_type == "Request" or node_type == "Script":
                    file.write(f"\t\t{node_type}: {node}\n")
                elif node_type == "Storage":
                    storage_count += 1
                elif node_type == "Element":
                    element_count += 1
                else:
                    decoration_count += 1
            if storage_count > 0:
                file.write(f"\t\tStorage: {storage_count}\n")
            if element_count > 0:
                file.write(f"\t\tElement: {element_count}\n")
            if decoration_count > 0:
                file.write(f"\t\tDecoration: {decoration_count}\n")


            file.write("\nOther Connected Components:\n")
            for cc in connected_components:
                if cc != largest_cc and cc != smallest_cc:
                    file.write(f"\nConnected Component (Size: {len(cc)}):\n")
                    for node in cc:
                        node_type = G.nodes[node].get('type', 'N/A')  
                        if node_type == "Document" or node_type == "Request" or node_type == "Script":
                           file.write(f"\t\t{node_type}: {node}\n")
                        elif node_type == "Storage":
                            storage_count += 1
                        elif node_type == "Element":
                            element_count += 1
                        else:
                            decoration_count += 1
                    if storage_count > 0:
                        file.write(f"\t\tStorage: {storage_count}\n")
                    if element_count > 0:
                        file.write(f"\t\tElement: {element_count}\n")
                    if decoration_count > 0:
                        file.write(f"\t\tDecoration: {decoration_count}\n")
                    file.write("\n")
        file.write("\n")

def extract_graph_features(G, visit_id, graph_type, graph_path): 
    node_feature = []
    connectivity_feature = []
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    nodes_div_by_edges = num_nodes/num_edges
    edges_div_by_nodes = num_edges/num_nodes


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

    # Find the average shortest path length for the largest connected component
    largest_cc_subgraph = G.subgraph(largest_cc_nodes)
    average_path_length_for_largest_cc = nx.average_shortest_path_length(largest_cc_subgraph.to_undirected())

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

    graph_info_full_data_in_txt(G, visit_id, max_in_degree, max_out_degree,
               connected_components, largest_cc_nodes, smallest_cc_nodes, 
                max_avg_path_length, component_with_max_path, 
                degree_centrality, max_degree_centrality, min_degree_centrality, 
                closeness_centrality, max_closeness_centrality, min_closeness_centrality,
                closeness_centrality_outward, max_closeness_centrality_outward, min_closeness_centrality_outward,
                average_path_length_for_largest_cc, graph_type, graph_path)

    # graph_info_to_csv(G, visit_id, max_in_degree, max_out_degree,
    #           connected_components, largest_cc_nodes, smallest_cc_nodes, 
    #            max_avg_path_length, component_with_max_path, 
    #            degree_centrality, max_degree_centrality, min_degree_centrality, 
    #            closeness_centrality, max_closeness_centrality, min_closeness_centrality,
    #            closeness_centrality_outward, max_closeness_centrality_outward, min_closeness_centrality_outward,
    #            average_path_length_for_largest_cc, graph_type, graph_path)

    connectivity_feature = [density, avg_clustering_coefficient, transitivity, number_of_ccs, \
                             average_size_cc, largest_cc, smallest_cc, max_avg_path_length, \
                            average_degree_centrality, median_degree_centrality, max_degree_centrality, \
                            min_degree_centrality, std_dev_degree_centrality, \
                            average_closeness_centrality, median_closeness_centrality, max_closeness_centrality, \
                            min_closeness_centrality, std_dev_closeness_centrality, \
                            average_closeness_centrality_outward, median_closeness_centrality_outward,\
                            max_closeness_centrality_outward, min_closeness_centrality_outward, std_dev_closeness_centrality_outward, \
                            average_path_length_for_largest_cc]       
    
    connectivity_feature_names = ['density', 'avg_clustering_coefficient', 'transitivity', 'number_of_ccs', \
                             'average_size_cc', 'largest_cc', 'smallest_cc', 'max_avg_path_length', \
                            'average_degree_centrality', 'median_degree_centrality', 'max_degree_centrality', \
                            'min_degree_centrality', 'std_dev_degree_centrality',  \
                            'average_closeness_centrality', 'median_closeness_centrality', 'max_closeness_centrality', \
                            'min_closeness_centrality', 'std_dev_closeness_centrality', \
                            'average_closeness_centrality_outward', 'median_closeness_centrality_outward',\
                            'max_closeness_centrality_outward', 'min_closeness_centrality_outward', 'std_dev_closeness_centrality_outward', \
                            'average_path_length_for_largest_cc']       
    

    simpler_feature = [num_nodes, num_edges, max_in_degree, max_out_degree, nodes_div_by_edges, \
                            edges_div_by_nodes, density, largest_cc, number_of_ccs, transitivity, average_path_length_for_largest_cc]
    
    simpler_feature_name = ['num_nodes', 'num_edges', 'max_in_degree', 'max_out_degree', 'nodes_div_by_edges', \
                            'edges_div_by_nodes', 'density', 'largest_cc', 'number_of_ccs', 'transitivity', 'average_path_length_for_largest_cc']

    return node_feature, node_feature_name, connectivity_feature, connectivity_feature_names, simpler_feature, simpler_feature_name



