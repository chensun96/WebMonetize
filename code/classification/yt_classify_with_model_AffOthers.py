from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import pandas as pd
import os
import sys
from yaml import load, dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from treeinterpreter import treeinterpreter as ti
import json
from collections import Counter
import random
import numpy as np
import collections
import tldextract
import pickle
from tqdm.auto import tqdm

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def report_feature_importance(feature_importances, result_dir, file_name="featimpcomplete"):
    """
    Function to make classification stats report.

    Args:
    report, result_dir, avg='macro avg', stats=['mean', 'std']
    Returns:
    Nothing, writes to a file

    This functions does the following:

    1. Splits the visit IDs into sets for each fold (one set will be used as test data).
    2. Creates test and train data.
    3. Performs training/classification.
    """

    fname = os.path.join(result_dir, file_name)
    with open(fname, "a") as f:
        f.write(feature_importances.to_string())
        f.write("\n")


def report_true_pred(
    y_true,
    y_pred,
    name,
    vid,
    i,
    result_dir,
):
    """
    Function to make truth/prediction output file.

    Args:
    y_true: Truth values
    y_pred: Predicted values
    name: Classified resource URLs
    vid: Visit IDs
    i: Fold number
    result_dir: Output directory
    Returns:
    Nothing, writes to a file

    This functions does the following:

    1. Obtains the classification metrics for each fold.
    """

    fname = os.path.join(result_dir, "tp_%s" % str(i))
    with open(fname, "w") as f:
        for i in range(0, len(y_true)):
            f.write(
                "%s |$| %s |$| %s |$| %s\n" % (y_true[i], y_pred[i], name[i], vid[i])
            )

    fname = os.path.join(result_dir, "confusion_matrix")
    with open(fname, "a") as f:
        f.write(
            np.array_str(
                confusion_matrix(y_true, y_pred, labels=["others", "affiliate"])
            )
            + "\n\n"
        )

def get_domain(url):

    try:
        if (isinstance(url, list)):
            domains = []
            for u in url:
                u = tldextract.extract(u)
                domains.append(u.domain+"."+u.suffix)
            return domains
        else:
            u = tldextract.extract(url)
            return u.domain+"."+u.suffix
    except:
        #traceback.print_exc()
        return None
    

def label_party(name):
    parts = name.split("||")

    if get_domain(parts[0].strip()) == get_domain(parts[1].strip()):
        return "First"
    else:
        return "Third"



def classify_logistic_regression(df, result_dir, file_name, model_name):
    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir)
    # load the pickled model
    clf = pickle.load(open(model_name, "rb"))


    fields_to_remove = ["visit_id", "name", "landing_page_domain", "label", "top_level_url", "Unnamed: 0", 'Unnamed: 0.1', "Unnamed: 0_x" ,"Unnamed: 0_y"]

    df_features = df.drop(fields_to_remove, axis=1, errors="ignore")
    #df_features.to_csv("/home/data/chensun/affi_project/purl/output/affiliate/fullGraph/test_2.csv")

    columns = df_features.columns
    print("columns: ", columns)

    # Scaling features
    scaler = MinMaxScaler()
    df_features_scaled = scaler.fit_transform(df_features)

    # predict the labels
    y_pred = clf.predict(df_features_scaled)
    # y_pred_proba[:, 1] > 0.5, then the predicted label would be 'others' 
    y_pred_proba = clf.predict_proba(df_features_scaled)[:, 1]  # Assuming binary classification

    y_true = df["label"].tolist()
    print("y_pred: ", y_pred)
    print("\ny_true: ", y_true)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label='affiliate')  # adjust 'affiliate' if needed
    recall = recall_score(y_true, y_pred, average='binary', pos_label='affiliate')  # adjust 'affiliate' if needed

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=["affiliate", "others"])

    # Save metrics and confusion matrix to a file
    metrics_file = os.path.join(result_dir, "classification_metrics.txt")
    with open(metrics_file, "w") as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(conf_matrix, separator=", "))

    # predict the probabilities
    y_pred_proba = clf.predict_proba(df_features)
    print(clf.classes_)  # e.g., ['others' 'affiliate']

    # add the predicted labels to the dataframe
    df["clabel"] = y_pred

    # add the predicted probabilities to the dataframe
    # [:, 1] means the probability of the sample belonging to the second class
    df["clabel_prob"] = y_pred_proba[:, 1]

    # save the results
    df.to_csv(os.path.join(result_dir, file_name), index=False)

    
def classify_unseen_with_threshold(df, result_dir, file_name, model_name, threshold):

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    fields_to_remove = ["visit_id", "name", "landing_page_domain", "label", "top_level_url", "Unnamed: 0", 'Unnamed: 0.1', "Unnamed: 0_x", "Unnamed: 0_y"]
    df_features = df.drop(fields_to_remove, axis=1, errors="ignore")
    columns = df_features.columns
    df_features = df_features.to_numpy()

    # load the pickled model
    clf = pickle.load(open(model_name, "rb"))

    # predict the probabilities
    y_pred_proba = clf.predict_proba(df_features)
    # Apply the threshold to get new predictions
    y_pred_custom_boolean = y_pred_proba[:, 0] >= threshold
    y_pred_custom_labels = np.where(y_pred_custom_boolean, 'affiliate', 'others')

    # Set true labels and predicted labels
    y_true = df["label"].tolist()
    y_pred = y_pred_custom_labels  # Use the thresholded predictions

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='affiliate')
    recall = recall_score(y_true, y_pred, pos_label='affiliate')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=["affiliate", "others"])

    # Save metrics and confusion matrix to a file
    metrics_file = os.path.join(result_dir, "classification_metrics.txt")
    with open(metrics_file, "w") as file:
        file.write(f"Threshold: {threshold}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(conf_matrix, separator=", "))

    # save the results
    df["clabel"] = y_pred
    df["clabel_prob"] = y_pred_proba[:, 0]
    df.to_csv(os.path.join(result_dir, file_name), index=False)

    # feature importance (assuming report_feature_importance is defined elsewhere)
    feature_importances = pd.DataFrame(
        clf.feature_importances_, index=columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir, file_name.split(".")[0] + "_featimp")


def classify_all_others(df, result_dir, file_name, model_name, threshold):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    fields_to_remove = ["visit_id", "name", "landing_page_domain", "label", "top_level_url", "Unnamed: 0", 'Unnamed: 0.1', "Unnamed: 0_x", "Unnamed: 0_y", "clabel", "clabel_prob"]
    df_features = df.drop(fields_to_remove, axis=1, errors="ignore")
    columns = df_features.columns
    df_features = df_features.to_numpy()

    # load the pickled model
    clf = pickle.load(open(model_name, "rb"))

    # predict the probabilities
    y_pred_proba = clf.predict_proba(df_features)
    # Apply the threshold to get new predictions
    y_pred_custom_boolean = y_pred_proba[:, 0] >= threshold
    y_pred_custom_labels = np.where(y_pred_custom_boolean, 'affiliate', 'others')
    y_pred = y_pred_custom_labels
    df["clabel"] = y_pred
    df["clabel_prob"] = y_pred_proba[:, 0]
    df.to_csv(os.path.join(result_dir, file_name), index=False)
    print(os.path.join(result_dir, file_name))

    
def classify_seen_with_threshold(df, result_dir, file_name, model_name, threshold):

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    fields_to_remove = ["visit_id", "name", "landing_page_domain", "label", "top_level_url", "Unnamed: 0", 'Unnamed: 0.1', "Unnamed: 0_x", "Unnamed: 0_y", "clabel", "clabel_prob"]
    df_features = df.drop(fields_to_remove, axis=1, errors="ignore")
    columns = df_features.columns
    df_features = df_features.to_numpy()

    # load the pickled model
    clf = pickle.load(open(model_name, "rb"))

    # predict the probabilities
    y_pred_proba = clf.predict_proba(df_features)
    # Apply the threshold to get new predictions
    y_pred_custom_boolean = y_pred_proba[:, 0] >= threshold
    y_pred_custom_labels = np.where(y_pred_custom_boolean, 'affiliate', 'others')

    # Set true labels and predicted labels
    y_true = df["label"].tolist()
    y_pred = y_pred_custom_labels  # Use the thresholded predictions

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='affiliate')
    recall = recall_score(y_true, y_pred, pos_label='affiliate')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=["affiliate", "others"])

    # Save metrics and confusion matrix to a file
    metrics_file = os.path.join(result_dir, "classification_metrics.txt")
    with open(metrics_file, "w") as file:
        file.write(f"Threshold: {threshold}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(conf_matrix, separator=", "))

    # save the results
    df["clabel"] = y_pred
    df["clabel_prob"] = y_pred_proba[:, 0]
    df.to_csv(os.path.join(result_dir, file_name), index=False)

    # feature importance (assuming report_feature_importance is defined elsewhere)
    feature_importances = pd.DataFrame(
        clf.feature_importances_, index=columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir, file_name.split(".")[0] + "_featimp")



def append_csv_data(folder_path, file_name, df_accumulator):
    """Appends data from CSV files """
    for crawl_id in os.listdir(folder_path):
        each_crawl = os.path.join(folder_path, crawl_id)
        file_path = os.path.join(each_crawl, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, on_bad_lines='skip')
            df['visit_id'] = df['visit_id'].astype(str)
            if file_name == 'url_features.csv':
                df = df.drop_duplicates(subset=["visit_id"])
            
            df_accumulator = df_accumulator.append(df)
    return df_accumulator


def deduplicate_features(df_records, df_features):
    """Deduplicates features based on certain columns."""
    df_records['parent_visit_id'] = df_records['visit_id'].str.split('_', expand=True)[0]
    merged_df = pd.merge(df_records, df_features, on='visit_id', how='inner')
    deduplicated_df = merged_df.drop_duplicates(subset=["name", "num_nodes", "num_edges", "parent_domain", "parent_visit_id"])
    return df_features.merge(deduplicated_df[['visit_id']], on=["visit_id"])


def prepare_dataset_features(folder):
    # Create DataFrames
    df_others_labels, df_others_records, df_others_url_features = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_others_phaseA_features_all, df_others_phaseA_features_simple = pd.DataFrame(), pd.DataFrame()
    
    # Append data from CSV files for both 'others' and 'affiliate' folders
    df_others_phaseA_features_all = append_csv_data(others_folder, "features_phase1.csv", df_others_phaseA_features_all)
    df_others_phaseA_features_simple = append_csv_data(others_folder, "features_phase1_simple.csv", df_others_phaseA_features_simple)
    df_others_labels = append_csv_data(others_folder, "rule_based_label.csv", df_others_labels)
    df_others_records = append_csv_data(others_folder, "records.csv", df_others_records)
    df_others_url_features = append_csv_data(others_folder, "url_features.csv", df_others_url_features)  

    # Deduplicate features
    df_others_phaseA_features_all = deduplicate_features(df_others_records, df_others_phaseA_features_all)
    df_others_phaseA_features_simple = deduplicate_features(df_others_records, df_others_phaseA_features_simple)
    
    df_records = df_others_records
   

    #remove the storage relative features
    features_to_remove = ["num_get_storage" ,"num_set_storage" , "num_get_storage_js", "num_set_storage_js", "num_all_gets", "num_all_sets", "num_get_storage_in_product_node", "num_set_storage_in_product_node", "num_get_storage_js_in_product_node", "num_set_storage_js_in_product_node", "num_all_gets_in_product_node", "num_all_sets_in_product_node"]
    df_others_phaseA_features_all = df_others_phaseA_features_all.drop(features_to_remove, axis=1, errors="ignore")
    df_others_phaseA_features_simple = df_others_phaseA_features_simple.drop(features_to_remove, axis=1, errors="ignore")


    # merge all the features to include url level features and simple graph features
    df_others_graph_features = df_others_phaseA_features_simple.merge(
    df_others_phaseA_features_all, 
    on=['visit_id', 'top_level_url', 'name', 'num_nodes', 'num_edges', 'max_in_degree', 'max_out_degree', 'density', 'largest_cc', 'number_of_ccs', 'transitivity', 'average_path_length_for_largest_cc'],  # Include all common columns here
    how='inner'
    )
    df_others_all_features = df_others_graph_features.merge(df_others_url_features, on='visit_id', how='inner')


    print("Classifying the phaseA all features")
    df_features_phaseA_all = df_others_all_features
    
    # Reduce some features
    featurs_to_remove_2 = ["average_size_cc" ,"init_url_shannon_entropy" , "min_degree_centrality", "average_in_degree", "number_of_ccs", "init_url_num_query_params", "median_closeness_centrality_outward", "min_closeness_centrality", "min_closeness_centrality_outward", "median_out_degree", "init_url_path_depth"]
    df_features_phaseA_all = df_features_phaseA_all.drop(featurs_to_remove_2, axis=1, errors="ignore")

    df_others_labels = df_others_labels.drop_duplicates(subset=['visit_id', 'url'])

    df_features_phaseA_all.drop(columns=['name', 'Unnamed: 0', "Unnamed: 0_x" ,"Unnamed: 0_y", "Unnamed: 0_x_x", "Unnamed: 0_x_y", "Unnamed: 0_y_x", "Unnamed: 0_y_y"], inplace=True, errors="ignore")
    #df_features.rename(columns={"name_x": "name"}, inplace=True)

    # change the "redirect_domain_total" to "name"
    df_others_labels.rename(columns={"redirect_domain_total": "name"}, inplace=True)
    df_others_labels.rename(columns={"final_rules_based_label": "label"}, inplace=True)

    #  merge label, landing_page_domain, features based on name 
    df = df_features_phaseA_all.merge(df_others_labels[['visit_id', 'label', 'name']], on=["visit_id"])
    
    
    df = df.merge(df_records[['visit_id', 'landing_page_domain']], on=["visit_id"])

    # if a row in df, its df['landing_page_domain]  in ["facebook.com", "instagram.com", "twitter.com", "youtube.com", "linkedin.com", "pinterest.com", "weibo.com"]
    # ignore this row
    excluded_domains = ["facebook.com", "instagram.com", "twitter.com", "youtube.com", 
                    "linkedin.com", "pinterest.com", "weibo.com"]
    df = df[~df['landing_page_domain'].isin(excluded_domains)]

    #df.to_csv("/home/data/chensun/affi_project/purl/output/test_2.csv")
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    print(df.columns)
    print(len(df))
    return df

    
def apply_to_all_links(others_folder):
    features_types = ["phase1"]
    #graph_types = ["affiliate", "others"]
    num_trials = 1  # change this
    for iteration in range(num_trials):
        iteration = num_trials
 
        for feature_type in features_types:
            MODEL_NAME = f"../../output/results/04_25_yt_threshold=0.5_undersampling_5fold_CV_reducedF/{feature_type}/{iteration}/best_model.sav"
            RESULT_DIR = f"../../output/results/04_25_yt_threshold=0.5_undersampling_5fold_CV_reducedF/{feature_type}/{iteration}/all_others_data_with_model_0.5"
            
            print(f"Classifying unseen data with {feature_type} features")
            df = prepare_dataset_features(others_folder)
           
            threshold = 0.5
         
            classify_all_others(df, RESULT_DIR, "classify_all_others.csv", MODEL_NAME, threshold)
    


if __name__ == "__main__":
    others_folder = "../../output/rule_based_others_yt"
    #apply_to_all_links(others_folder)

    """ Unseen data set """
    
    #features_types = ["phase1", "phase1_simple"]
    features_types = ["phase1"]
    #graph_types = ["affiliate", "others"]
    num_trials = 10  # change this
    for iteration in range(num_trials):
 
        for feature_type in features_types:
            MODEL_NAME = f"../../output/results/05_08_yt_threshold=0.5_undersampling_5fold_CV_reducedF/{feature_type}/{iteration}/best_model.sav"
            RESULT_DIR = f"../../output/results/05_08_yt_threshold=0.5_undersampling_5fold_CV_reducedF/{feature_type}/{iteration}/with_model_0.3"
            
            print(f"Classifying unseen data with {feature_type} features")
            unseen_data_path = f"../../output/results/05_08_yt_threshold=0.5_undersampling_5fold_CV_reducedF/{feature_type}/unseen_{iteration}.csv" 
            df_labelled = pd.read_csv(unseen_data_path)

            threshold = 0.3
            # classify_logistic_regression(df_labelled, RESULT_DIR, "labelled_results.csv", MODEL_NAME)
            classify_unseen_with_threshold(df_labelled, RESULT_DIR, "labelled_results.csv", MODEL_NAME, threshold)
    

  
    """ Seen data set """
    """
    #features_types = ["phase1", "phase1_simple"]
    features_types = ["phase1"]
    #graph_types = ["affiliate", "others"]
    num_trials = 5  # change this
    for iteration in range(num_trials):
 
        for feature_type in features_types:
            MODEL_NAME = f"../../output/results/04_23_yt_threshold=0.2_undersampling_5fold_CV_importantF/{feature_type}/{iteration}/best_model.sav"
            RESULT_DIR = f"../../output/results/04_23_yt_threshold=0.2_undersampling_5fold_CV_importantF/{feature_type}/{iteration}/seen_data_with_model_0.4"
            
            print(f"Classifying seen data with {feature_type} features")
            seen_data_path = f"../../output/results/04_23_yt_threshold=0.2_undersampling_5fold_CV_importantF/{feature_type}/{iteration}/result.csv" 
            df_labelled = pd.read_csv(seen_data_path)

            threshold = 0.4
            # classify_logistic_regression(df_labelled, RESULT_DIR, "labelled_results.csv", MODEL_NAME)
            #classify_seen_with_threshold(df_labelled, RESULT_DIR, "labelled_results.csv", MODEL_NAME, threshold)
    """