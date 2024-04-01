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

def classify(df, result_dir, file_name, model_name):
    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir)
    # load the pickled model
    clf = pickle.load(open(model_name, "rb"))

    df_labelled.drop(columns=['name_x'], inplace=True)
    df_labelled.rename(columns={"name_y": "name"}, inplace=True)

    fields_to_remove = ["visit_id", "name", "url_domain", "label", "top_level_url", "Unnamed: 0", 'Unnamed: 0.1', "Unnamed: 0_x" ,"Unnamed: 0_y"]

    df_features = df.drop(fields_to_remove, axis=1, errors="ignore")
    #df_features.to_csv("/home/data/chensun/affi_project/purl/output/affiliate/fullGraph/test_2.csv")

    columns = df_features.columns
    print("columns: ", columns)

    df_features = df_features.to_numpy()

    # predict the labels
    y_pred = clf.predict(df_features)
    y_true = df["label"].tolist()
    print("y_pred: ", y_pred)
    print("\ny_true: ", y_true)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label='affiliate')  # adjust 'affiliate' if needed
    recall = recall_score(y_true, y_pred, average='binary', pos_label='affiliate')  # adjust 'affiliate' if needed

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=["others", "affiliate"])

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

    # feature importance
    feature_importances = pd.DataFrame(
        clf.feature_importances_, index=columns, columns=["importance"]
    ).sort_values("importance", ascending=False)

    report_feature_importance(feature_importances, result_dir, file_name.split(".")[0] + "_featimp")


if __name__ == "__main__":

    # phaseA classification
    others_folder = "../../output/others"  # change this!
    ads_folder = "../../output/ads"    # change this!
    affiliate_folder = "../../output/affiliate"    # change this!
    
    #features_types = ["phase1", "phase1_simple"]
    features_types = ["phase1"]
    #graph_types = ["affiliate", "others"]
    num_trials = 100
    for iteration in range(num_trials):
 
        for feature_type in features_types:
            MODEL_NAME = f"../../output/results/02_28_uniform_Kfold_CV_AFF_OTHERS_allF/{feature_type}/{iteration}/best_model.sav"
            RESULT_DIR = f"../../output/results/02_28_uniform_Kfold_CV_AFF_OTHERS_allF/{feature_type}/{iteration}/with_model"

            
            print(f"Classifying unseen data with {feature_type} features")
            unseen_data_path = f"../../output/results/02_28_uniform_Kfold_CV_AFF_OTHERS_allF/{feature_type}/unseen_{iteration}.csv" 
            df_labelled = pd.read_csv(unseen_data_path)

            classify(df_labelled, RESULT_DIR, "labelled_results.csv", MODEL_NAME)


   



    
