from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
import functools
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    fbeta_score
)
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
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
import pickle
from tqdm.auto import tqdm
import tldextract
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.model_selection import KFold
import ast


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

def select_high_freq_domains(df, cutoff_ratio):
    # Calculate domain frequencies
    domain_counts = df['landing_page_domain'].value_counts()
    # Sort domains by frequency (value_counts already sorts them in descending order)
    sorted_domains = domain_counts.index.tolist()
    print("\tSorted_domains: ", sorted_domains)
    # Select domains for training based on cutoff ratio
    cutoff = int(len(sorted_domains) * cutoff_ratio)
    sorted_domains = set(sorted_domains[:cutoff])
    print(f"\tTop {cutoff_ratio*100}% domain: {sorted_domains}")

    df_top_domain = df[df['landing_page_domain'].isin(sorted_domains)]
    print(f"\tNumber of url in top {cutoff_ratio*100}% domain: {len(df_top_domain)}")

    df_not_in_top_domain = df[~df['landing_page_domain'].isin(sorted_domains)]
    print(f"\tNumber of url not in top {cutoff_ratio*100}% domain: {len(df_not_in_top_domain)}\n")
    return sorted_domains

# TODO: debug
def check_duplicate_domains(selected_unique_domains, label_type, unseen_domain_record_path):
    with open(unseen_domain_record_path, 'r') as f:
        records = f.read().strip().split('\n\n')
    print(records[0])

    for record in records:
        if label_type == "affiliate":
            print(record)
            print(record.split('\n')[4])
            generated_domains = ast.literal_eval(record.split('\n')[4].split(': ')[1])
        else: 
            print(record.split('\n')[5])
            generated_domains = ast.literal_eval(record.split('\n')[5].split(': ')[1])
        
        # Check if all selected domains are in the affiliate domains
        if all(domain in generated_domains for domain in selected_unique_domains):
            return True  # Skip to next iteration

    return False  # Continue with current iterat


def perpare_unseen_landing_page_domain(df, num_trials, label_type, result_dir):
    random.seed(42)
    # Randomly shuffle # of unique domain times
    # Each round will pick one domain as unseen data set

    total_unique_domains = df['landing_page_domain'].nunique()
    print("Total unique domains:", total_unique_domains)

    all_selected_domains_with_freq = []

    for _ in range(num_trials):
        selected_domains = {}
        unique_domains_list = df['landing_page_domain'].unique()

        # Calculate 20% of the total unique domains
        num_domains_to_select = int(total_unique_domains * 0.2)

        selected_unique_domains = random.sample(list(unique_domains_list), num_domains_to_select)

        # TODO
        unseen_domain_record_path = os.path.join(result_dir, "unseen_domain_record")

        # check if the domains generated before. If generated, continue
        #if check_duplicate_domains(selected_unique_domains, label_type, unseen_domain_record_path):
        #    continue
        
        for domain in selected_unique_domains:
            domain_count = df[df['landing_page_domain'] == domain]['landing_page_domain'].count()
            selected_domains[domain] = domain_count
        
        all_selected_domains_with_freq.append(selected_domains)

    print(f"Total number of selected unique domain sets after {num_trials} trials:", len(all_selected_domains_with_freq))

    for i in range(len(all_selected_domains_with_freq)):
        print(all_selected_domains_with_freq[i])
    
    return all_selected_domains_with_freq


"""
# select 5 domain as unseen data
def perpare_unseen_landing_page_domain(df, num_trials, label_type, result_dir):

    random.seed(42)
    # Randomly shuffle # of unique domain times
    # Each round will pick one domain as unseen data set

    total_unique_domains = df['landing_page_domain'].nunique()
    print("Total unique domains:", total_unique_domains)

    all_selected_domains_with_freq = []

    for _ in range(num_trials):
        selected_domains = {}
        unique_domains_list = df['landing_page_domain'].unique()
        # selected_unique_domains = random.sample(list(unique_domains_list), int(total_unique_domains * cutoff_ratio))
        selected_unique_domains = random.sample(list(unique_domains_list), 5)

        # TODO
        unseen_domain_record_path = os.path.join(result_dir, "unseen_domain_record")

        # check if the domains generated before. If generated, continue
        #if check_duplicate_domains(selected_unique_domains, label_type, unseen_domain_record_path):
        #    continue
        
        for domain in selected_unique_domains:
            domain_count = df[df['landing_page_domain'] == domain]['landing_page_domain'].count()
            selected_domains[domain] = domain_count
        
        all_selected_domains_with_freq.append(selected_domains)

    print(f"Total number of selected unique domain sets after {num_trials} trials:", len(all_selected_domains_with_freq))

    for i in range(len(all_selected_domains_with_freq)):
        print(all_selected_domains_with_freq[i])
    
    return all_selected_domains_with_freq
"""



def reduce_data(df_records, df_affiliate_phaseA_features_all, df_affiliate_phaseA_features_simple, keyword, fraction_to_select):
    df_contain_keyword = df_records[df_records['landing_page_domain'] == keyword]
    print(f"\nFor {keyword}")
    print("\tnumber of url has key words: ", len(df_contain_keyword))
    # Randomly select fraction_to_select of data

    random_sample_df = df_contain_keyword.sample(frac=fraction_to_select, random_state=42)
    print(f"\tselect to keep :{len(random_sample_df)}")
    df_non_keyword = df_records[df_records['landing_page_domain'] != keyword]
    print("\tlen of df_non_keyword: ", len(df_non_keyword))
 
    reduced_df = pd.concat([df_non_keyword, random_sample_df])

    #print("visit_ids: ", len(visit_ids))
    #print("Original size:", len(df_features))
    #print("Reduced size:", len(reduced_df_features))
    print("\tlen of new reduced df: ", len(reduced_df))

    visit_ids = reduced_df['visit_id']

    # remove the same visit id from other dataframe
    filtered_df_phaseA_features_all = df_affiliate_phaseA_features_all[df_affiliate_phaseA_features_all['visit_id'].isin(visit_ids)]
    print(f"\tlength of filtered_df_phaseA_features_all: {len(filtered_df_phaseA_features_all)}")
    filtered_df_phaseA_features_simple = df_affiliate_phaseA_features_simple[df_affiliate_phaseA_features_simple['visit_id'].isin(visit_ids)]
    print(f"\tlength of filtered_df_phaseA_features_simple {len(filtered_df_phaseA_features_simple)}")
    return reduced_df, filtered_df_phaseA_features_all, filtered_df_phaseA_features_simple
   

def print_stats(report, result_dir, avg="macro avg", stats=["mean", "std"]):
    """
    Function to make classification stats report.

    Args:
    report,
    result_dir,
    avg='macro avg',
    stats=['mean', 'std']
    Returns:
    Nothing, writes to a file

    This functions does the following:

    1. Splits the visit IDs into sets for each fold (one set will be used as test data).
    2. Creates test and train data.
    3. Performs training/classification.
    """

    by_label = report.groupby("label").describe()
    fname = os.path.join(result_dir, "scores")
    with open(fname, "w") as f:
        for stat in stats:
            print(by_label.loc[avg].xs(stat, level=1))
            x = by_label.loc[avg].xs(stat, level=1)
            f.write(by_label.loc[avg].xs(stat, level=1).to_string())
            f.write("\n")


def report_feature_importance(feature_importances, result_dir):
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

    fname = os.path.join(result_dir, "featimp")
    with open(fname, "a") as f:
        f.write(feature_importances.to_string())
        f.write("\n")


def report_true_pred(y_true, y_pred, name, vid, i, result_dir):
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
                confusion_matrix(y_true, y_pred, labels=["affiliate", "others"])
            )
            + "\n\n"
        )


def describe_classif_reports(results, result_dir):
    """
    Function to make classification stats report.

    Args:
    results: Results of classification
    result_dir: Output directory
    Returns:
    all_folds: DataFrame of results

    This functions does the following:

    1. Obtains the classification metrics for each fold.
    """

    true_vectors, pred_vectors, name_vectors, vid_vectors = (
        [r[0] for r in results],
        [r[1] for r in results],
        [r[2] for r in results],
        [r[3] for r in results],
    )
    fname = os.path.join(result_dir, "scores")

    all_folds = pd.DataFrame(
        columns=["label", "fold", "precision", "recall", "f1-score", "support"]
    )
    for i, (y_true, y_pred, name, vid) in enumerate(
        zip(true_vectors, pred_vectors, name_vectors, vid_vectors)
    ):
        report_true_pred(y_true, y_pred, name, vid, i, result_dir)
        output = classification_report(y_true, y_pred)
        with open(fname, "a") as f:
            f.write(output)
            f.write("\n\n")
        # df = pd.DataFrame(output).transpose().reset_index().rename(columns={'index': 'label'})
        # df['fold'] = i
        # all_folds = all_folds.append(df)
    return all_folds


def log_prediction_probability(
    fitted_model, df_feature_test, cols, test_mani, y_pred, result_dir, tag
):
    y_pred_prob = fitted_model.predict_proba(df_feature_test)

    fname = os.path.join(result_dir, "predict_prob_" + str(tag))
    with open(fname, "w") as f:
        class_names = [str(x) for x in fitted_model.classes_]
        class_names = " |$| ".join(class_names)
        f.write("Truth |$| Pred |$| " + class_names + " |$| Name |$| VID" + "\n")

        truth_labels = [str(x) for x in list(test_mani.label)]
        pred_labels = [str(x) for x in list(y_pred)]
        truth_names = [str(x) for x in list(test_mani.name)]
        truth_vids = [str(x) for x in list(test_mani.visit_id)]

        for i in range(0, len(y_pred_prob)):
            preds = [str(x) for x in y_pred_prob[i]]
            preds = " |$| ".join(preds)
            f.write(
                "%s |$| %s |$| %s |$| %s |$| %s\n"
                % (
                    truth_labels[i],
                    pred_labels[i],
                    preds,
                    truth_names[i],
                    truth_vids[i],
                )
            )

    preds, bias, contributions = ti.predict(fitted_model, df_feature_test)
    fname = os.path.join(result_dir, "interpretation_" + str(tag))
    with open(fname, "w") as f:
        data_dict = {}
        for i in range(len(df_feature_test)):
            name = test_mani.iloc[i]["name"]
            vid = str(test_mani.iloc[i]["visit_id"])
            key = str(name) + "_" + str(vid)
            data_dict[key] = {}
            data_dict[key]["name"] = name
            data_dict[key]["vid"] = vid
            c = list(contributions[i, :, 0])
            c = [round(float(x), 2) for x in c]
            fn = list(cols)
            fn = [str(x) for x in fn]
            feature_contribution = list(zip(c, fn))
            # feature_contribution = list(zip(contributions[i,:,0], df_feature_test.columns))
            data_dict[key]["contributions"] = feature_contribution
        f.write(json.dumps(data_dict, indent=4))



def get_perc(num, den):
    return str(round(num / den * 100, 2)) + "%"

def label_party(name):
    parts = name.split("||")

    if get_domain(parts[0].strip()) == get_domain(parts[1].strip()):
        return "First"
    else:
        return "Third"

def gird_search_LeaveOneGroupOut(df_labelled, df_labelled_holdout, result_dir, iteration, log_pred_probability):
    result_dir = result_dir + "/" + str(iteration)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    # Extract 'landing_page_domain' for LeaveOneGroupOut before preprocessing
    groups = df_labelled['landing_page_domain'].values

    df_labelled.drop(columns=['name_x'], inplace=True)
    df_labelled.rename(columns={"name_y": "name"}, inplace=True)
    df_labelled_holdout.drop(columns=['name_x'], inplace=True)
    df_labelled_holdout.rename(columns={"name_y": "name"}, inplace=True)


    train_mani = df_labelled.copy()
    holdout_mani = df_labelled_holdout.copy()

    fields_to_remove = ["visit_id", "landing_page_domain", "name", "label", "party", "Unnamed: 0", 'top_level_url', 'Unnamed: 0_x', 'Unnamed: 0_y', 'Unnamed: 0_x_x', 'Unnamed: 0_x_y', 'Unnamed: 0_y_x', 'Unnamed: 0_y_y']
    
    # Store the columns you want to retain
    train_retained = train_mani[["visit_id", "name", "top_level_url"]]
    holdout_retained = holdout_mani[["visit_id", "name", "top_level_url"]]

    df_feature_train = train_mani.drop(fields_to_remove, axis=1, errors="ignore")
    train_labels = train_mani.label
    col_train = df_feature_train.columns
    df_feature_holdout = df_labelled_holdout.drop(fields_to_remove, axis=1, errors="ignore")

    # Align the order of features in df_feature_test with df_feature_train
    df_feature_holdout = df_feature_holdout[col_train]
    holdout_labels = holdout_mani.label
    col_holdout = df_feature_holdout.columns

    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_labels.to_numpy()

    #result_df = df_feature_holdout.copy()
    df_feature_holdout = df_feature_holdout.to_numpy()
    holdout_labels = holdout_labels.to_numpy()

    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50,100,150], # number of trees in the forest
        'max_features': ['sqrt'],   # consider every features /square root of features
        'max_depth': [2, 5, 10],
        'min_samples_split': [5, 10, 15],  # minimum number of samples that are required to split an internal node.
        'min_samples_leaf': [2, 5, 10],
        'bootstrap': [False]
    }
    # Initialize the classifier
    rf = RandomForestClassifier()

    # Initialize LeaveOneGroupOut
    logo = LeaveOneGroupOut()

    # Initialize Grid Search with 10-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=logo, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(df_feature_train, train_labels, groups=groups)

    # Get the best model
    best_model = grid_search.best_estimator_

    best_params = best_model.get_params()
    params_filename = os.path.join(result_dir, "best_model_parameters.txt")
    with open(params_filename, 'w') as file:
        for param, value in best_params.items():
            print(f"{param}: {value}")
            file.write(f"{param}: {value}\n")   

    # Save the model to disk
    filename = os.path.join(result_dir, "best_model.sav")
    pickle.dump(best_model, open(filename, "wb"))

    
    # Get feature importances
    feature_importances = pd.DataFrame(
        best_model.feature_importances_, 
        index=col_train, 
        columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir)


    # Make predictions on the hold-out set
    y_pred = best_model.predict(df_feature_holdout)
    y_pred_proba = best_model.predict_proba(df_feature_holdout)
    print(best_model.classes_)  # e.g., ['others' 'affiliate']


    result_df = pd.DataFrame(df_feature_holdout, columns=col_train)
    result_df["clabel"] = y_pred
    result_df["clabel_prob"] = y_pred_proba[:, 1]  # assuming binary classification
    result_df['label'] = holdout_labels

    # Concatenate the retained columns with result_df
    result_df = pd.concat([holdout_retained.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

    # Save to CSV
    result_df.to_csv(os.path.join(result_dir, "result.csv"), index=False)


    acc = accuracy_score(holdout_labels, y_pred)
    prec_binary = precision_score(holdout_labels, y_pred, pos_label="affiliate")
    rec_binary = recall_score(holdout_labels, y_pred, pos_label="affiliate")
    prec_micro = precision_score(holdout_labels, y_pred, average="micro")
    rec_micro = recall_score(holdout_labels, y_pred, average="micro")
    prec_macro = precision_score(holdout_labels, y_pred, average="macro")
    rec_macro = recall_score(holdout_labels, y_pred, average="macro")

    # Write accuracy score
    fname = os.path.join(result_dir, "accuracy")
    with open(fname, "a") as f:
        f.write("\nAccuracy score: " + str(round(acc * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: binary " + str(round(prec_binary * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: binary " + str(round(rec_binary * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: micro " + str(round(prec_micro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: micro " + str(round(rec_micro * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: macro " + str(round(prec_macro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: macro " + str(round(rec_macro * 100, 3)) + "%" + "\n")

    print("Accuracy Score:", acc)

    if log_pred_probability:
        log_prediction_probability(
            best_model, df_feature_holdout, col_holdout, df_labelled_holdout, y_pred, result_dir, tag='0'
        )
    


def gird_search_LeavePGroupOut(df_labelled, df_labelled_holdout, result_dir, iteration, log_pred_probability):
    result_dir = result_dir + "/" + str(iteration)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    # Extract 'landing_page_domain' for LeaveOneGroupOut before preprocessing
    groups = df_labelled['landing_page_domain'].values

    df_labelled.drop(columns=['name_x'], inplace=True)
    df_labelled.rename(columns={"name_y": "name"}, inplace=True)
    df_labelled_holdout.drop(columns=['name_x'], inplace=True)
    df_labelled_holdout.rename(columns={"name_y": "name"}, inplace=True)


    train_mani = df_labelled.copy()
    holdout_mani = df_labelled_holdout.copy()

    fields_to_remove = ["visit_id", "landing_page_domain", "name", "label", "party", "Unnamed: 0", 'top_level_url', 'Unnamed: 0_x', 'Unnamed: 0_y']
    
    # Store the columns you want to retain
    train_retained = train_mani[["visit_id", "name", "top_level_url"]]
    holdout_retained = holdout_mani[["visit_id", "name", "top_level_url"]]

    df_feature_train = train_mani.drop(fields_to_remove, axis=1, errors="ignore")
    train_labels = train_mani.label
    col_train = df_feature_train.columns
    df_feature_holdout = df_labelled_holdout.drop(fields_to_remove, axis=1, errors="ignore")

    # Align the order of features in df_feature_test with df_feature_train
    df_feature_holdout = df_feature_holdout[col_train]
    holdout_labels = holdout_mani.label
    col_holdout = df_feature_holdout.columns

    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_labels.to_numpy()

    #result_df = df_feature_holdout.copy()
    df_feature_holdout = df_feature_holdout.to_numpy()
    holdout_labels = holdout_labels.to_numpy()

    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100,150,200], # number of trees in the forest
        'max_features': [None,'sqrt'],   # consider every features /square root of features
        'max_depth': [5, 10, 20],
        'min_samples_split': [5, 10, 15],  # minimum number of samples that are required to split an internal node.
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [False]
    }
    # Initialize the classifier
    rf = RandomForestClassifier()

    # Initialize LeaveOneGroupOut
    lkgo = LeavePGroupsOut(n_groups=5)  # Change n_groups as needed

    # Initialize Grid Search with 10-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=lkgo, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(df_feature_train, train_labels, groups=groups)

    # Get the best model
    best_model = grid_search.best_estimator_

    best_params = best_model.get_params()
    params_filename = os.path.join(result_dir, "best_model_parameters.txt")
    with open(params_filename, 'w') as file:
        for param, value in best_params.items():
            print(f"{param}: {value}")
            file.write(f"{param}: {value}\n")   

    # Save the model to disk
    filename = os.path.join(result_dir, "best_model.sav")
    pickle.dump(best_model, open(filename, "wb"))

    
    # Get feature importances
    feature_importances = pd.DataFrame(
        best_model.feature_importances_, 
        index=col_train, 
        columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir)


    # Make predictions on the hold-out set
    y_pred = best_model.predict(df_feature_holdout)
    y_pred_proba = best_model.predict_proba(df_feature_holdout)
    print(best_model.classes_)  # e.g., ['others' 'affiliate']


    result_df = pd.DataFrame(df_feature_holdout, columns=col_train)
    result_df["clabel"] = y_pred
    result_df["clabel_prob"] = y_pred_proba[:, 1]  # assuming binary classification
    result_df['label'] = holdout_labels

    # Concatenate the retained columns with result_df
    result_df = pd.concat([holdout_retained.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

    # Save to CSV
    result_df.to_csv(os.path.join(result_dir, "result.csv"), index=False)


    acc = accuracy_score(holdout_labels, y_pred)
    prec_binary = precision_score(holdout_labels, y_pred, pos_label="affiliate")
    rec_binary = recall_score(holdout_labels, y_pred, pos_label="affiliate")
    prec_micro = precision_score(holdout_labels, y_pred, average="micro")
    rec_micro = recall_score(holdout_labels, y_pred, average="micro")
    prec_macro = precision_score(holdout_labels, y_pred, average="macro")
    rec_macro = recall_score(holdout_labels, y_pred, average="macro")

    # Write accuracy score
    fname = os.path.join(result_dir, "accuracy")
    with open(fname, "a") as f:
        f.write("\nAccuracy score: " + str(round(acc * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: binary " + str(round(prec_binary * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: binary " + str(round(rec_binary * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: micro " + str(round(prec_micro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: micro " + str(round(rec_micro * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: macro " + str(round(prec_macro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: macro " + str(round(rec_macro * 100, 3)) + "%" + "\n")

    print("Accuracy Score:", acc)

    if log_pred_probability:
        log_prediction_probability(
            best_model, df_feature_holdout, col_holdout, df_labelled_holdout, y_pred, result_dir, tag='0'
        )


# logistic regression
def gird_search_Kfold_CV_Logistic_Regression(df_labelled, df_labelled_holdout, result_dir, iteration, log_pred_probability):
    result_dir = result_dir + "/" + str(iteration)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    # Extract 'landing_page_domain' for LeaveOneGroupOut before preprocessing
    groups = df_labelled['landing_page_domain'].values

    df_labelled.drop(columns=['name_x'], inplace=True)
    df_labelled.rename(columns={"name_y": "name"}, inplace=True)
    df_labelled_holdout.drop(columns=['name_x'], inplace=True)
    df_labelled_holdout.rename(columns={"name_y": "name"}, inplace=True)


    train_mani = df_labelled.copy()
    holdout_mani = df_labelled_holdout.copy()

    fields_to_remove = ["visit_id", "landing_page_domain", "name", "label", "party", "Unnamed: 0", 'top_level_url', 'Unnamed: 0_x', 'Unnamed: 0_y']
    
    # Store the columns you want to retain
    train_retained = train_mani[["visit_id", "name", "top_level_url"]]
    holdout_retained = holdout_mani[["visit_id", "name", "top_level_url"]]

    df_feature_train = train_mani.drop(fields_to_remove, axis=1, errors="ignore")
    train_labels = train_mani.label
    col_train = df_feature_train.columns
    df_feature_holdout = df_labelled_holdout.drop(fields_to_remove, axis=1, errors="ignore")

    # Align the order of features in df_feature_test with df_feature_train
    df_feature_holdout = df_feature_holdout[col_train]
    holdout_labels = holdout_mani.label
    col_holdout = df_feature_holdout.columns

    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_labels.to_numpy()

    #result_df = df_feature_holdout.copy()
    df_feature_holdout = df_feature_holdout.to_numpy()
    holdout_labels = holdout_labels.to_numpy()

    scaler = MinMaxScaler()
    df_feature_train_scaled = scaler.fit_transform(df_feature_train)  # Fit and transform the scaler on the training data
    df_feature_holdout_scaled = scaler.transform(df_feature_holdout)  # Transform the holdout data using the same scaler


    # Define the parameter grid for logistic regression
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l1', 'l2'],  # Type of regularization
        'solver': ['liblinear']  # Solver that supports L1 penalty
    }

    # Initialize and fit the logistic regression model using scaled data
    lr = LogisticRegression()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=kfold, n_jobs=-1, verbose=2)
    grid_search.fit(df_feature_train_scaled, train_labels)

    # Get the best model
    best_model = grid_search.best_estimator_

    best_params = best_model.get_params()
    params_filename = os.path.join(result_dir, "best_model_parameters.txt")
    with open(params_filename, 'w') as file:
        for param, value in best_params.items():
            print(f"{param}: {value}")
            file.write(f"{param}: {value}\n")   

    # Save the model to disk
    filename = os.path.join(result_dir, "best_model.sav")
    pickle.dump(best_model, open(filename, "wb"))

    """
    # Get feature importances
    feature_importances = pd.DataFrame(
        best_model.feature_importances_, 
        index=col_train, 
        columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir)
    """

    
    y_pred = best_model.predict(df_feature_holdout_scaled)  # Use scaled features here
    y_pred_proba = best_model.predict_proba(df_feature_holdout_scaled)  # And here
    print(best_model.classes_)  # e.g., ['others', 'affiliate']


    result_df = pd.DataFrame(df_feature_holdout, columns=col_train)
    result_df["clabel"] = y_pred
    result_df["clabel_prob"] = y_pred_proba[:, 1]  # assuming binary classification
    result_df['label'] = holdout_labels

    # Concatenate the retained columns with result_df
    result_df = pd.concat([holdout_retained.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

    # Save to CSV
    result_df.to_csv(os.path.join(result_dir, "result.csv"), index=False)


    acc = accuracy_score(holdout_labels, y_pred)
    prec_binary = precision_score(holdout_labels, y_pred, pos_label="affiliate")
    rec_binary = recall_score(holdout_labels, y_pred, pos_label="affiliate")
    prec_micro = precision_score(holdout_labels, y_pred, average="micro")
    rec_micro = recall_score(holdout_labels, y_pred, average="micro")
    prec_macro = precision_score(holdout_labels, y_pred, average="macro")
    rec_macro = recall_score(holdout_labels, y_pred, average="macro")

    # Write accuracy score
    fname = os.path.join(result_dir, "accuracy")
    with open(fname, "a") as f:
        f.write("\nAccuracy score: " + str(round(acc * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: binary " + str(round(prec_binary * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: binary " + str(round(rec_binary * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: micro " + str(round(prec_micro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: micro " + str(round(rec_micro * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: macro " + str(round(prec_macro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: macro " + str(round(rec_macro * 100, 3)) + "%" + "\n")

    print("Accuracy Score:", acc)

    """
    if log_pred_probability:
        log_prediction_probability(
            best_model, df_feature_holdout, col_holdout, df_labelled_holdout, y_pred, result_dir, tag='0'
        )
    """


# random forest
def gird_search_Kfold_CV(df_labelled, df_labelled_holdout, result_dir, iteration, log_pred_probability):
    result_dir = result_dir + "/" + str(iteration)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    # Extract 'landing_page_domain' for LeaveOneGroupOut before preprocessing
    groups = df_labelled['landing_page_domain'].values

    df_labelled.drop(columns=['name_x'], inplace=True)
    df_labelled.rename(columns={"name_y": "name"}, inplace=True)
    df_labelled_holdout.drop(columns=['name_x'], inplace=True)
    df_labelled_holdout.rename(columns={"name_y": "name"}, inplace=True)


    train_mani = df_labelled.copy()
    holdout_mani = df_labelled_holdout.copy()

    fields_to_remove = ["visit_id", "landing_page_domain", "name", "label", "party", "Unnamed: 0", 'top_level_url', 'Unnamed: 0_x', 'Unnamed: 0_y']
    
    # Store the columns you want to retain
    train_retained = train_mani[["visit_id", "name", "top_level_url"]]
    holdout_retained = holdout_mani[["visit_id", "name", "top_level_url"]]

    df_feature_train = train_mani.drop(fields_to_remove, axis=1, errors="ignore")
    train_labels = train_mani.label
    col_train = df_feature_train.columns
    df_feature_holdout = df_labelled_holdout.drop(fields_to_remove, axis=1, errors="ignore")

    # Align the order of features in df_feature_test with df_feature_train
    df_feature_holdout = df_feature_holdout[col_train]
    holdout_labels = holdout_mani.label
    col_holdout = df_feature_holdout.columns

    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_labels.to_numpy()

    #result_df = df_feature_holdout.copy()
    df_feature_holdout = df_feature_holdout.to_numpy()
    holdout_labels = holdout_labels.to_numpy()


    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100,150,200], # number of trees in the forest
        'max_features': [None,'sqrt'],   # consider every features /square root of features
        'max_depth': [2, 5, 10, 20],
        'min_samples_split': [5, 10, 15],  # minimum number of samples that are required to split an internal node.
        'min_samples_leaf': [1, 2, 5, 10],
        'bootstrap': [True, False]
    }

    scoring = {
        'precision': make_scorer(precision_score, pos_label='affiliate'),
        'recall': make_scorer(recall_score, pos_label='affiliate'),
        'f1': make_scorer(f1_score, pos_label='affiliate')
    }

    # Initialize the classifier
    rf = RandomForestClassifier()

    # Initialize KFold cross-validation with 10 folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scoring, refit='recall', cv=kfold, n_jobs=0, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(df_feature_train, train_labels, groups=groups)

    # Get the best model
    best_model = grid_search.best_estimator_

    best_params = best_model.get_params()
    params_filename = os.path.join(result_dir, "best_model_parameters.txt")
    with open(params_filename, 'w') as file:
        for param, value in best_params.items():
            print(f"{param}: {value}")
            file.write(f"{param}: {value}\n")   

    # Save the model to disk
    filename = os.path.join(result_dir, "best_model.sav")
    pickle.dump(best_model, open(filename, "wb"))

    # Convert results to DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Save to CSV
    gridsearch_result_path = os.path.join(result_dir, 'grid_search_results.csv') 
    results_df.to_csv(gridsearch_result_path, index=False)

    
    # Get feature importances
    feature_importances = pd.DataFrame(
        best_model.feature_importances_, 
        index=col_train, 
        columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir)


    # Make predictions on the hold-out set
    y_pred = best_model.predict(df_feature_holdout)
    y_pred_proba = best_model.predict_proba(df_feature_holdout)
    print("best_model.classes_: ", best_model.classes_)  # e.g., ['affiliate', 'others']


    result_df = pd.DataFrame(df_feature_holdout, columns=col_train)
    result_df["clabel"] = y_pred
    result_df["clabel_prob"] = y_pred_proba[:, 0]  # assuming binary classification
    result_df['label'] = holdout_labels

    # Concatenate the retained columns with result_df
    result_df = pd.concat([holdout_retained.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

    # Save to CSV
    result_df.to_csv(os.path.join(result_dir, "result.csv"), index=False)


    acc = accuracy_score(holdout_labels, y_pred)
    prec_binary = precision_score(holdout_labels, y_pred, pos_label="affiliate")
    rec_binary = recall_score(holdout_labels, y_pred, pos_label="affiliate")
    prec_micro = precision_score(holdout_labels, y_pred, average="micro")
    rec_micro = recall_score(holdout_labels, y_pred, average="micro")
    prec_macro = precision_score(holdout_labels, y_pred, average="macro")
    rec_macro = recall_score(holdout_labels, y_pred, average="macro")

    # Write accuracy score
    fname = os.path.join(result_dir, "accuracy")
    with open(fname, "a") as f:
        f.write("\nAccuracy score: " + str(round(acc * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: binary " + str(round(prec_binary * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: binary " + str(round(rec_binary * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: micro " + str(round(prec_micro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: micro " + str(round(rec_micro * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: macro " + str(round(prec_macro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: macro " + str(round(rec_macro * 100, 3)) + "%" + "\n")

    print("Accuracy Score:", acc)

    if log_pred_probability:
        log_prediction_probability(
            best_model, df_feature_holdout, col_holdout, df_labelled_holdout, y_pred, result_dir, tag='0'
        )


# random forest
def gird_search_Kfold_CV_custom_scorer(df_labelled, df_labelled_holdout, result_dir, iteration, log_pred_probability, threshold):
    result_dir = result_dir + "/" + str(iteration)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    # Extract 'landing_page_domain' for LeaveOneGroupOut before preprocessing
    groups = df_labelled['landing_page_domain'].values

    train_mani = df_labelled.copy()
    holdout_mani = df_labelled_holdout.copy()

    fields_to_remove = ["visit_id", "landing_page_domain", "name", "label", "party", "Unnamed: 0", 'top_level_url', 'Unnamed: 0_x', 'Unnamed: 0_y']
    
    # Store the columns you want to retain
    train_retained = train_mani[["visit_id", "name", "top_level_url"]]
    holdout_retained = holdout_mani[["visit_id", "name", "top_level_url"]]

    df_feature_train = train_mani.drop(fields_to_remove, axis=1, errors="ignore")
    train_labels = train_mani.label
    print("1 train_labels: ", train_labels)
    col_train = df_feature_train.columns
    df_feature_holdout = df_labelled_holdout.drop(fields_to_remove, axis=1, errors="ignore")

    # Align the order of features in df_feature_test with df_feature_train
    df_feature_holdout = df_feature_holdout[col_train]
    holdout_labels = holdout_mani.label
    col_holdout = df_feature_holdout.columns

    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_labels.to_numpy()


    #result_df = df_feature_holdout.copy()
    df_feature_holdout = df_feature_holdout.to_numpy()
    holdout_labels = holdout_labels.to_numpy()
    
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100,150,200], # number of trees in the forest
        'max_features': [None,'sqrt'],   # consider every features /square root of features
        'max_depth': [2, 5, 10, 20],
        'min_samples_split': [5, 10, 15],  # minimum number of samples that are required to split an internal node.
        'min_samples_leaf': [1, 2, 5, 10],
        'bootstrap': [True, False]
    }
    
    '''
    def f2_score_at_thresh(y_true, y_prob, threshold=threshold):
       
        y_pred = (y_prob > threshold).astype(int)   # > 0.8 is others
        y_pred = np.where(y_pred == 1, 'others', 'affiliate')
   
        return fbeta_score(y_true, y_pred, beta=2, pos_label='affiliate')
    

    # Create a partially filled version of f2_score_at_thresh that always uses a specific threshold
    custom_f2_scorer = functools.partial(f2_score_at_thresh, threshold=threshold)
    my_scorer = make_scorer(custom_f2_scorer, response_method='predict_proba')
    '''

    def custom_threshold_accuracy(y_true, y_prob, threshold):
  
        y_pred = (y_prob > threshold).astype(int)   # > 0.75 is others
        #print("\n y_pred: ", y_pred)
        y_pred = np.where(y_pred == 1, 'others', 'affiliate')
        return recall_score(y_true, y_pred, pos_label='affiliate')

    my_scorer = make_scorer(custom_threshold_accuracy, needs_proba=True, threshold=threshold)
    
    # Initialize the classifier
    rf = RandomForestClassifier()

    # Initialize KFold cross-validation with 10 folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=my_scorer, cv=kfold, n_jobs=50, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(df_feature_train, train_labels, groups=groups)

    # Get the best model
    best_model = grid_search.best_estimator_

    best_params = best_model.get_params()
    params_filename = os.path.join(result_dir, "best_model_parameters.txt")
    with open(params_filename, 'w') as file:
        for param, value in best_params.items():
            print(f"{param}: {value}")
            file.write(f"{param}: {value}\n")   

    # Save the model to disk
    filename = os.path.join(result_dir, "best_model.sav")
    pickle.dump(best_model, open(filename, "wb"))

    # Convert results to DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Save to CSV
    gridsearch_result_path = os.path.join(result_dir, 'grid_search_results.csv') 
    results_df.to_csv(gridsearch_result_path, index=False)

    
    # Get feature importances
    feature_importances = pd.DataFrame(
        best_model.feature_importances_, 
        index=col_train, 
        columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir)


    # Make predictions on the hold-out set

    y_pred = best_model.predict(df_feature_holdout)
    y_pred_proba = best_model.predict_proba(df_feature_holdout)
    print("best_model.classes_: ", best_model.classes_)  # e.g., ['affiliate', 'others']

    print("holdout_labels: ", holdout_labels)
    
    

    result_df = pd.DataFrame(df_feature_holdout, columns=col_train)
    result_df["clabel"] = y_pred
    result_df["clabel_prob"] = y_pred_proba[:, 0]  # assuming binary classification
    result_df['label'] = holdout_labels

    # Concatenate the retained columns with result_df
    result_df = pd.concat([holdout_retained.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

    # Save to CSV
    result_df.to_csv(os.path.join(result_dir, "result.csv"), index=False)

    acc = accuracy_score(holdout_labels, y_pred)
    prec_binary = precision_score(holdout_labels, y_pred, pos_label="affiliate")
    rec_binary = recall_score(holdout_labels, y_pred, pos_label="affiliate")
    prec_micro = precision_score(holdout_labels, y_pred, average="micro")
    rec_micro = recall_score(holdout_labels, y_pred, average="micro")
    prec_macro = precision_score(holdout_labels, y_pred, average="macro")
    rec_macro = recall_score(holdout_labels, y_pred, average="macro")

    # Write accuracy score
    fname = os.path.join(result_dir, "accuracy")
    with open(fname, "a") as f:
        f.write("\nAccuracy score: " + str(round(acc * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: binary " + str(round(prec_binary * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: binary " + str(round(rec_binary * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: micro " + str(round(prec_micro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: micro " + str(round(rec_micro * 100, 3)) + "%" + "\n")
        f.write(
            "Precision score: macro " + str(round(prec_macro * 100, 3)) + "%" + "\n"
        )
        f.write("Recall score: macro " + str(round(rec_macro * 100, 3)) + "%" + "\n")

    print("Accuracy Score:", acc)

    if log_pred_probability:
        log_prediction_probability(
            best_model, df_feature_holdout, col_holdout, df_labelled_holdout, y_pred, result_dir, tag='0'
        )  


def pipeline(df_features, df_labels, df_records, result_dir, threshold):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    df_labels = df_labels.drop_duplicates(subset=['visit_id', 'url'])
    print("df_labels: ", len(df_labels))
    #df_records.to_csv("/home/data/chensun/affi_project/purl/output/test_3.csv")

    # Limit to 30 URLs if there are more than 30
    unique_domains = df_records['landing_page_domain'].unique()
    df_selected_urls = pd.DataFrame(columns=df_records.columns)
    for domain in unique_domains:
        # Select URLs for the current domain
        domain_records = df_records[df_records['landing_page_domain'] == domain]

        if len(domain_records) > 30:
            selected_records = domain_records.sample(n=30, random_state=42)  
        else:
            selected_records = domain_records.copy()

        # Append selected URLs to the DataFrame
        df_selected_urls = df_selected_urls.append(selected_records)


    print("Uniform distribution: ", len(df_selected_urls))
    print(df_selected_urls.columns)

    df_records = df_selected_urls
    #df_records.to_csv("/home/data/chensun/affi_project/purl/output/test_2.csv")

    df_features.drop(columns=['name', 'Unnamed: 0', "Unnamed: 0_x" ,"Unnamed: 0_y", "Unnamed: 0_x_x", "Unnamed: 0_x_y", "Unnamed: 0_y_x", "Unnamed: 0_y_y"], inplace=True, errors="ignore")
    #df_features.rename(columns={"name_x": "name"}, inplace=True)

    # change the "redirect_domain_total" to "name"
    df_labels.rename(columns={"redirect_domain_total": "name"}, inplace=True)
    df_labels.rename(columns={"final_rules_based_label": "label"}, inplace=True)

    #  merge label, landing_page_domain, features based on name 
    df = df_features.merge(df_labels[['visit_id', 'label', 'name']], on=["visit_id"])
    
    
    df = df.merge(df_records[['visit_id', 'landing_page_domain']], on=["visit_id"])

    #df.to_csv("/home/data/chensun/affi_project/purl/output/test_2.csv")
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df_labelled = df
    df_positive = df[df["label"] == "affiliate"]
    df_negative = df[df["label"] == "others"]
    #df_negative.to_csv("/home/data/chensun/affi_project/purl/output/test_2.csv")
    #print("len df_positive: ", len(df_positive))
    #print("len df_negative: ", len(df_negative))
    df_unknown = df[df["label"] == "unknown"]
    # find nan values
    #print("Nan values")
    #print(df.isnull().values.any())
    #print("df_positive: ", df_positive.head(n=5))
    #  remove nan
    df_labelled = df_labelled.dropna()
    df_unknown = df_unknown.dropna()
    df_positive = df_positive.dropna()
    df_negative = df_negative.dropna()


    
    ### handle data imbalance ###
    # Step 1: Identify 120 unique domains randomly from df_negative
    unique_domains = df_negative['landing_page_domain'].unique()
    selected_domains = pd.Series(unique_domains).sample(n=200, random_state=42).tolist()

    # Step 2: Filter df_negative to only include rows from the selected domains
    df_negative = df_negative[df_negative['landing_page_domain'].isin(selected_domains)]
    
  
    fname = os.path.join(result_dir, "composition")
    with open(fname, "a") as f:
        f.write("Number of samples: " + str(len(df)) + "\n")
        f.write(
            "Labelled samples: "
            + str(len(df_labelled))
            + " "
            + get_perc(len(df_labelled), len(df))
            + "\n"
        )
        f.write(
            "Positive samples (affiliate): "
            + str(len(df_positive))
            + " "
            + get_perc(len(df_positive), len(df))
            + "\n"
        )
        f.write(
            "Negative samples (others): "
            + str(len(df_negative))
            + " "
            + get_perc(len(df_negative), len(df))
            + "\n"
        )
        f.write("Selected Domains for Undersampling (Negative samples):\n")
        for domain in selected_domains:
            f.write(domain + "\n")
        f.write("\n")
    
        
   
    print("\nPerpare unseen data for affiliate ...")
    num_trials = 10
    aff_type ='affiliate'
    all_positive_domains = perpare_unseen_landing_page_domain(df_positive, num_trials, aff_type, result_dir)
    
    others_type ='others'
    print("\nPerpare unseen data for others ...")
    all_negative_domains = perpare_unseen_landing_page_domain(df_negative, num_trials, others_type, result_dir)

    print(f"number of unqiue domain in affiliate {len(all_positive_domains)}")
    print(f"number of unqiue domain in others {len(all_negative_domains)}")
    
    
    for iteration in range(num_trials):
         
        print(f'Iteration: {iteration}')

        positive_domains_set = set(all_positive_domains[iteration].keys())
        negative_domains_set = set(all_negative_domains[iteration].keys())

        # Union of the two sets  ==> use for unseen data
        unseen_domains_set = positive_domains_set.union(negative_domains_set)
        print(f"\n\nDomain for unseen data: {unseen_domains_set}")

        # Check if a domain appears in both sets
        common_unseen_domains = positive_domains_set.intersection(negative_domains_set)
        print("\n Common domains: ", common_unseen_domains)

        fname = os.path.join(result_dir, "unseen_domain_record")
        with open(fname, "a") as f:
            f.write("Iteration: " + str(iteration) + "\n")
            f.write(
                "Domain for unseen data: "
                + str(unseen_domains_set)
                + "\n"
            )
            f.write("\n")
            f.write(
                "Common domains: "
                + str(common_unseen_domains)
                + "\n"
            )
            f.write("\n")
            f.write(
                "Affiliate domains: "
                + str(all_positive_domains[iteration])
                + "\n"
            )
            f.write("\n")
            f.write(
                "Others domains: "
                + str(all_negative_domains[iteration])
                + "\n"
            )
            f.write("\n\n")

        # exclude the unknown data
        df_labelled = pd.concat([df_positive,df_negative])
        
        # Perpare unseen data set
        # Ensure every domain in high_freq_affiliate_domains is represented in both training and testing set    
        df_holdout_2 = pd.DataFrame()
        df_holdout_2 = df_labelled[df_labelled['landing_page_domain'].isin(unseen_domains_set)]
        df_holdout_2.to_csv(os.path.join(result_dir, f"unseen_{iteration}.csv"), index=False)
        
        df_others_domain = df_labelled[~df_labelled['landing_page_domain'].isin(unseen_domains_set)]
        other_unique_domains = df_others_domain['landing_page_domain'].unique().tolist()

        df_labelled_train = pd.DataFrame()
        df_labelled_test = pd.DataFrame()

        for domain in other_unique_domains:
            print(f"Domain: {domain}")
            df_subset = df_labelled[df_labelled['landing_page_domain'] == domain]
            if len(df_subset) == 1:
                print("\tOnly one sample. Appending to training data.")
                df_labelled_train = df_labelled_train.append(df_subset)
                continue
            X_train, X_test = train_test_split(df_subset, test_size=0.2, random_state=42)
            print(f"\tNumber of url in Training: {len(X_train)} || Testing: {len(X_test)}")
            df_labelled_train = df_labelled_train.append(X_train)
            df_labelled_test = df_labelled_test.append(X_test)

        print(f"\nNumber of training data: {len(df_labelled_train)}")
        print(f"Number of testing data: {len(df_labelled_test)}")

        log_pred_probability=True

        # gird_search_Kfold_CV_Logistic_Regression(df_labelled_train, df_labelled_test, result_dir, iteration, log_pred_probability=True)
        # gird_search_Kfold_CV(df_labelled_train, df_labelled_test, result_dir, iteration, log_pred_probability=True)
        gird_search_Kfold_CV_custom_scorer(df_labelled_train, df_labelled_test, result_dir, iteration, log_pred_probability, threshold)
   

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


if __name__ == "__main__":

    others_folder = "../../output/rule_based_others_yt"
    affiliate_folder = "../../output/rule_based_aff_yt"
    RESULT_DIR = "../../output/results/05_08_yt_threshold=0.5_undersampling_5fold_CV_reducedF/"
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    threshold = 0.5
    RESULT_DIR_phaseA_all = os.path.join(RESULT_DIR, "phase1")
    RESULT_DIR_phaseA_simple = os.path.join(RESULT_DIR, "phase1_simple")

   

    # Create DataFrames
    df_others_labels, df_others_records, df_others_url_features = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_others_phaseA_features_all, df_others_phaseA_features_simple = pd.DataFrame(), pd.DataFrame()
    
    df_affiliate_labels, df_affiliate_records, df_affiliate_url_features = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_affiliate_phaseA_features_all, df_affiliate_phaseA_features_simple = pd.DataFrame(), pd.DataFrame()

    # Append data from CSV files for both 'others' and 'affiliate' folders
    df_others_phaseA_features_all = append_csv_data(others_folder, "features_phase1.csv", df_others_phaseA_features_all)
    df_others_phaseA_features_simple = append_csv_data(others_folder, "features_phase1_simple.csv", df_others_phaseA_features_simple)
    df_others_labels = append_csv_data(others_folder, "rule_based_label.csv", df_others_labels)
    df_others_records = append_csv_data(others_folder, "records.csv", df_others_records)
    df_others_url_features = append_csv_data(others_folder, "url_features.csv", df_others_url_features)

    df_affiliate_phaseA_features_all = append_csv_data(affiliate_folder, "features_phase1.csv", df_affiliate_phaseA_features_all)
    df_affiliate_phaseA_features_simple = append_csv_data(affiliate_folder, "features_phase1_simple.csv", df_affiliate_phaseA_features_simple)
    df_affiliate_labels = append_csv_data(affiliate_folder, "rule_based_label.csv", df_affiliate_labels)
    df_affiliate_records = append_csv_data(affiliate_folder, "records.csv", df_affiliate_records)
    df_affiliate_url_features = append_csv_data(affiliate_folder, "url_features.csv", df_affiliate_url_features)
   

    # Deduplicate features
    df_others_phaseA_features_all = deduplicate_features(df_others_records, df_others_phaseA_features_all)
    df_others_phaseA_features_simple = deduplicate_features(df_others_records, df_others_phaseA_features_simple)
    df_affiliate_phaseA_features_all = deduplicate_features(df_affiliate_records, df_affiliate_phaseA_features_all)
    df_affiliate_phaseA_features_simple = deduplicate_features(df_affiliate_records, df_affiliate_phaseA_features_simple)

    df_labels = pd.DataFrame()
    df_labels = pd.concat([df_others_labels,df_affiliate_labels])

    df_records = pd.DataFrame()
    df_records = pd.concat([df_others_records,df_affiliate_records])

    #remove the storage relative features
    features_to_remove = ["num_get_storage" ,"num_set_storage" , "num_get_storage_js", "num_set_storage_js", "num_all_gets", "num_all_sets", "num_get_storage_in_product_node", "num_set_storage_in_product_node", "num_get_storage_js_in_product_node", "num_set_storage_js_in_product_node", "num_all_gets_in_product_node", "num_all_sets_in_product_node"]
    df_affiliate_phaseA_features_all = df_affiliate_phaseA_features_all.drop(features_to_remove, axis=1, errors="ignore")
    df_others_phaseA_features_all = df_others_phaseA_features_all.drop(features_to_remove, axis=1, errors="ignore")
    df_affiliate_phaseA_features_simple = df_affiliate_phaseA_features_simple.drop(features_to_remove, axis=1, errors="ignore")
    df_others_phaseA_features_simple = df_others_phaseA_features_simple.drop(features_to_remove, axis=1, errors="ignore")
    print("2: ", len(df_affiliate_phaseA_features_all))


    # merge all the features to include url level features and simple graph features
    df_affiliate_graph_features = df_affiliate_phaseA_features_simple.merge(
    df_affiliate_phaseA_features_all, 
    on=['visit_id', 'top_level_url', 'name', 'num_nodes', 'num_edges', 'max_in_degree', 'max_out_degree', 'density', 'largest_cc', 'number_of_ccs', 'transitivity', 'average_path_length_for_largest_cc'],  # Include all common columns here
    how='inner'
    )

    # merge url features
    df_affiliate_all_features = df_affiliate_graph_features.merge(df_affiliate_url_features, on='visit_id', how='inner')

    df_others_graph_features = df_others_phaseA_features_simple.merge(
    df_others_phaseA_features_all, 
    on=['visit_id', 'top_level_url', 'name', 'num_nodes', 'num_edges', 'max_in_degree', 'max_out_degree', 'density', 'largest_cc', 'number_of_ccs', 'transitivity', 'average_path_length_for_largest_cc'],  # Include all common columns here
    how='inner'
    )
    df_others_all_features = df_others_graph_features.merge(df_others_url_features, on='visit_id', how='inner')


    print("Classifying the phaseA all features")
    df_features_phaseA_all = pd.DataFrame()
    df_features_phaseA_all = pd.concat([df_affiliate_all_features,df_others_all_features])

    # Reduce some features
    featurs_to_remove_2 = ["average_size_cc" ,"init_url_shannon_entropy" , "min_degree_centrality", "average_in_degree", "number_of_ccs", "init_url_num_query_params", "median_closeness_centrality_outward", "min_closeness_centrality", "min_closeness_centrality_outward", "median_out_degree", "init_url_path_depth"]
    df_features_phaseA_all = df_features_phaseA_all.drop(featurs_to_remove_2, axis=1, errors="ignore")
    
    # Or, only keep important features
    #features_to_keep = ['visit_id', 'top_level_url', 'name', 'total_num_query_params', 'num_of_redirect', 'average_shannon_entropy', 'average_num_query_params', 'total_num_path_depth', 'average_num_path_depth', 'std_dev_degree_centrality', 'std_dev_closeness_centrality', 'num_edges', 'num_nodes', 'density', 'max_closeness_centrality', 'max_degree_centrality', 'largest_cc', 'max_avg_path_length'] 
    #df_features_phaseA_all = df_features_phaseA_all[features_to_keep]
    pipeline(df_features_phaseA_all, df_labels, df_records, RESULT_DIR_phaseA_all, threshold)



    # no url features
    #df_affiliate_phaseA_features_simple = df_affiliate_phaseA_features_simple.merge(df_affiliate_url_features, on='visit_id', how='inner')
    # df_others_phaseA_features_simple = df_others_phaseA_features_simple.merge(df_others_url_features, on='visit_id', how='inner')
    # print("df_others_phaseA_features_simple: ", len(df_others_phaseA_features_simple))
    # print("df_affiliate_phaseA_features_simple: ", len(df_affiliate_phaseA_features_simple))

    # print("\n\nClassifying the phaseA simpler features")
    # df_features_phaseA_simple = pd.DataFrame()
    # df_features_phaseA_simple = pd.concat([df_others_phaseA_features_simple,df_affiliate_phaseA_features_simple])
    # pipeline(df_features_phaseA_simple, df_labels, df_records, RESULT_DIR_phaseA_simple)
    
   