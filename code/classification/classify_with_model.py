from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
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
                confusion_matrix(y_true, y_pred, labels=["ads", "affiliate"])
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

def classify(df, result_dir, file_name, model_name):
    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir)
    # load the pickled model
    clf = pickle.load(open(model_name, "rb"))


    fields_to_remove = ["visit_id", "name", "label", "top_level_url", "Unnamed: 0", 'Unnamed: 0.1']

    df_features = df.drop(fields_to_remove, axis=1, errors="ignore")
    #df_features.to_csv("/home/data/chensun/affi_project/purl/output/affiliate/fullGraph/test_2.csv")

    columns = df_features.columns
    print("columns: ", columns)

    df_features = df_features.to_numpy()

    # predict the labels
    y_pred = clf.predict(df_features)
    print("y_pred: ", y_pred)
    print("\ny_true: ", df["label"].tolist())

    str_confusion_matrix = np.array_str(confusion_matrix(df["label"].tolist(), y_pred, labels=["ads", "affiliate"]))
    print("confusion_matrix:\n", str_confusion_matrix)

    # predict the probabilities
    y_pred_proba = clf.predict_proba(df_features)
    print(clf.classes_)  # e.g., ['affiliate' 'ads']

    # add the predicted labels to the dataframe
    df["clabel"] = y_pred

    # add the predicted probabilities to the dataframe
    # [:, 1] means the probability of the sample belonging to the second class
    df["clabel_prob"] = y_pred_proba[:, 1]

    # save the results
    df.to_csv(os.path.join(result_dir, file_name), index=False)

    # feature importance
    feature_importances = pd.DataFrame(
        clf.feature_importances_, index=columns, columns=["importance"]
    ).sort_values("importance", ascending=False)

    report_feature_importance(feature_importances, result_dir, file_name.split(".")[0] + "_featimp")


if __name__ == "__main__":

    # phaseA classification
    normal_phaseA_folder = "../../output/normal/fullGraph"  # change this!
    ads_phaseA_folder = "../../output/ads/fullGraph"    # change this!
    affiliate_phaseA_folder = "../../output/affiliate/fullGraph"    # change this!
    RESULT_DIR = "../../output/results/aff_ads_graph_level_fullGraph_3/with_model_1"
    MODEL_NAME = "../../output/results/aff_ads_graph_level_fullGraph_3/model_1.sav"
    #RESULT_DIR = "../../output/results/aff_ads_mean_phaseA/with_model_2"
    #MODEL_NAME = "../../output/results/aff_ads_mean_phaseA/model_2.sav"

    print("here start the aff_ads")
    hold_out_path = "/home/data/chensun/affi_project/purl/output/results/df_labelled_holdout.csv"
    df_labelled = pd.read_csv(hold_out_path)
    # here we say unknown is normal type url
    df_labelled = df_labelled[df_labelled['label'] != "normal"]
    df_unknown = df_labelled[df_labelled["label"] == "normal"]

    classify(df_labelled, RESULT_DIR, "labelled_results.csv", MODEL_NAME)
    #classify(df_unknown, RESULT_DIR, "unknown_results.csv", MODEL_NAME)

    #classify(df, RESULT_DIR, "all_results.csv", MODEL_NAME)

    """
    # get features
    df_features = pd.DataFrame()
    for filename in os.listdir(normal_phaseA_folder):
        if filename.startswith("features_mean") and filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(normal_phaseA_folder, filename)
            df = pd.read_csv(file_path)
            print(len(df))
            df['label'] = "normal"
            df_features = df_features.append(df)
    for filename in os.listdir(ads_phaseA_folder):
        if filename.startswith("graph_level_features") and filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(ads_phaseA_folder, filename)
            df = pd.read_csv(file_path)
            print(len(df))
            df['label'] = "ads"
            df_features = df_features.append(df)

    for filename in os.listdir(affiliate_phaseA_folder):
        if filename.startswith("graph_level_features") and filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(affiliate_phaseA_folder, filename)
            df = pd.read_csv(file_path)
            print(len(df))
            df['label'] = "affiliate"
            df_features = df_features.append(df)
    print("len of features: ", len(df_features))
    #df_features.to_csv("../../output/test.csv")

   
    # get labels
    df_labels = pd.DataFrame()

    for filename in os.listdir(normal_phaseA_folder):
        if filename.startswith("labels_") and filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(normal_phaseA_folder, filename)
            df = pd.read_csv(file_path)
            print(len(df))
            df_labels = df_labels.append(df)
    for filename in os.listdir(ads_phaseA_folder):
        if filename.startswith("labels_") and filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(ads_phaseA_folder, filename)
            df = pd.read_csv(file_path)
            print(len(df))
            df_labels = df_labels.append(df)

    for filename in os.listdir(affiliate_phaseA_folder):
        if filename.startswith("labels_") and filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(affiliate_phaseA_folder, filename)
            df = pd.read_csv(file_path)
            print(len(df))
            df_labels = df_labels.append(df)
    print("len of labels: ", len(df_labels))

    df_labels = df_labels.drop_duplicates(subset=['visit_id'])
    print("df_labels: ", df_labels.head())

    # drop "top_level_url" column
    new_df_labels = df_labels.drop('top_level_url', axis=1)

    df = df_features.merge(new_df_labels[['visit_id', 'label', 'name']], on=["visit_id"])
    #df.to_csv("/home/data/chensun/affi_project/purl/output/affiliate/fullGraph/test_2.csv")

    # only need to drop label_y in phaseA?
    df.drop(columns=["label_y"], inplace=True)
    df.rename(columns={"label_x": "label"}, inplace=True)

    df.drop(columns=["name_y"], inplace=True)
    df.rename(columns={"name_x": "name"}, inplace=True)
    df = df.drop_duplicates()

    """
   