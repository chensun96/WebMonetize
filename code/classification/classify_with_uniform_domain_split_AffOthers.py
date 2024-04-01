from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
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
    domain_counts = df['url_domain'].value_counts()
    # Sort domains by frequency (value_counts already sorts them in descending order)
    sorted_domains = domain_counts.index.tolist()
    print("\tSorted_domains: ", sorted_domains)
    # Select domains for training based on cutoff ratio
    cutoff = int(len(sorted_domains) * cutoff_ratio)
    sorted_domains = set(sorted_domains[:cutoff])
    print(f"\tTop {cutoff_ratio*100}% domain: {sorted_domains}")

    df_top_domain = df[df['url_domain'].isin(sorted_domains)]
    print(f"\tNumber of url in top {cutoff_ratio*100}% domain: {len(df_top_domain)}")

    df_not_in_top_domain = df[~df['url_domain'].isin(sorted_domains)]
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

def perpare_unseen_url_domain(df, num_trials, label_type, result_dir):

    random.seed(42)
    # Randomly shuffle # of unique domain times
    # Each round will pick one domain as unseen data set

    total_unique_domains = df['url_domain'].nunique()
    print("Total unique domains:", total_unique_domains)

    all_selected_domains_with_freq = []

    for _ in range(num_trials):
        selected_domains = {}
        unique_domains_list = df['url_domain'].unique()
        # selected_unique_domains = random.sample(list(unique_domains_list), int(total_unique_domains * cutoff_ratio))
        selected_unique_domains = random.sample(list(unique_domains_list), 5)

        # TODO
        unseen_domain_record_path = os.path.join(result_dir, "unseen_domain_record")

        # check if the domains generated before. If generated, continue
        #if check_duplicate_domains(selected_unique_domains, label_type, unseen_domain_record_path):
        #    continue
        
        for domain in selected_unique_domains:
            domain_count = df[df['url_domain'] == domain]['url_domain'].count()
            selected_domains[domain] = domain_count
        
        all_selected_domains_with_freq.append(selected_domains)

    print(f"Total number of selected unique domain sets after {num_trials} trials:", len(all_selected_domains_with_freq))

    for i in range(len(all_selected_domains_with_freq)):
        print(all_selected_domains_with_freq[i])
    
    return all_selected_domains_with_freq




def reduce_data(df_records, df_affiliate_phaseA_features_all, df_affiliate_phaseA_features_simple, keyword, fraction_to_select):
    df_contain_keyword = df_records[df_records['url_domain'] == keyword]
    print(f"\nFor {keyword}")
    print("\tnumber of url has key words: ", len(df_contain_keyword))
    # Randomly select fraction_to_select of data

    random_sample_df = df_contain_keyword.sample(frac=fraction_to_select, random_state=42)
    print(f"\tselect to keep :{len(random_sample_df)}")
    df_non_keyword = df_records[df_records['url_domain'] != keyword]
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


    # Extract 'url_domain' for LeaveOneGroupOut before preprocessing
    groups = df_labelled['url_domain'].values

    df_labelled.drop(columns=['name_x'], inplace=True)
    df_labelled.rename(columns={"name_y": "name"}, inplace=True)
    df_labelled_holdout.drop(columns=['name_x'], inplace=True)
    df_labelled_holdout.rename(columns={"name_y": "name"}, inplace=True)


    train_mani = df_labelled.copy()
    holdout_mani = df_labelled_holdout.copy()

    fields_to_remove = ["visit_id", "url_domain", "name", "label", "party", "Unnamed: 0", 'top_level_url', 'Unnamed: 0_x', 'Unnamed: 0_y']
    
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


    # Extract 'url_domain' for LeaveOneGroupOut before preprocessing
    groups = df_labelled['url_domain'].values

    df_labelled.drop(columns=['name_x'], inplace=True)
    df_labelled.rename(columns={"name_y": "name"}, inplace=True)
    df_labelled_holdout.drop(columns=['name_x'], inplace=True)
    df_labelled_holdout.rename(columns={"name_y": "name"}, inplace=True)


    train_mani = df_labelled.copy()
    holdout_mani = df_labelled_holdout.copy()

    fields_to_remove = ["visit_id", "url_domain", "name", "label", "party", "Unnamed: 0", 'top_level_url', 'Unnamed: 0_x', 'Unnamed: 0_y']
    
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
    

def gird_search_Kfold_CV(df_labelled, df_labelled_holdout, result_dir, iteration, log_pred_probability):
    result_dir = result_dir + "/" + str(iteration)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    # Extract 'url_domain' for LeaveOneGroupOut before preprocessing
    groups = df_labelled['url_domain'].values

    df_labelled.drop(columns=['name_x'], inplace=True)
    df_labelled.rename(columns={"name_y": "name"}, inplace=True)
    df_labelled_holdout.drop(columns=['name_x'], inplace=True)
    df_labelled_holdout.rename(columns={"name_y": "name"}, inplace=True)


    train_mani = df_labelled.copy()
    holdout_mani = df_labelled_holdout.copy()

    fields_to_remove = ["visit_id", "url_domain", "name", "label", "party", "Unnamed: 0", 'top_level_url', 'Unnamed: 0_x', 'Unnamed: 0_y']
    
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

    # Initialize KFold cross-validation with 10 folds
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize Grid Search with 10-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, n_jobs=-1, verbose=2)

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
    


def pipeline(df_features, df_labels, df_records, result_dir):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


    df_labels = df_labels.drop_duplicates(subset=['visit_id', 'name'])
    print("df_labels: ", len(df_labels))
    #df_records.to_csv("/home/data/chensun/affi_project/purl/output/test_3.csv")

    # Limit to 30 URLs if there are more than 30
    unique_domains = df_records['url_domain'].unique()
    df_selected_urls = pd.DataFrame(columns=df_records.columns)
    for domain in unique_domains:
        # Select URLs for the current domain
        domain_records = df_records[df_records['url_domain'] == domain]

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


    # drop "top_level_url" column
    new_df_labels = df_labels.drop('top_level_url', axis=1)

    df_features.drop(columns=['Unnamed: 0', "Unnamed: 0_x" ,"Unnamed: 0_y"], inplace=True)
    #df_features.rename(columns={"name_x": "name"}, inplace=True)


    #  merge label, url_domain, features based on name 
    df = df_features.merge(new_df_labels[['visit_id', 'label', 'name']], on=["visit_id"])
    print("1: ", len(df))
    
    
    df = df.merge(df_records[['visit_id', 'url_domain']], on=["visit_id"])

    #df.to_csv("/home/data/chensun/affi_project/purl/output/test_2.csv")
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df_labelled = df
    df_positive = df[df["label"] == "affiliate"]
    df_negative = df[df["label"] == "others"]
    #df_negative.to_csv("/home/data/chensun/affi_project/purl/output/test_2.csv")
    #print("len df_positive: ", len(df_positive))
    #print("len df_negative: ", len(df_negative))
    df_unknown = df[df["label"] == "normal"]
    # find nan values
    #print("Nan values")
    #print(df.isnull().values.any())
    #print("df_positive: ", df_positive.head(n=5))
    #  remove nan
    df_labelled = df_labelled.dropna()
    df_unknown = df_unknown.dropna()
    df_positive = df_positive.dropna()
    df_negative = df_negative.dropna()
  
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
        f.write("\n")
        
   
    print("\nPerpare unseen data for affiliate ...")
    num_trials = 100
    aff_type ='affiliate'
    all_positive_domains = perpare_unseen_url_domain(df_positive, num_trials, aff_type, result_dir)
    
    others_type ='others'
    print("\nPerpare unseen data for others ...")
    all_negative_domains = perpare_unseen_url_domain(df_negative, num_trials, others_type, result_dir)

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
        
        # Perpare unseen data set
        # Ensure every domain in high_freq_affiliate_domains is represented in both training and testing set    
        df_holdout_2 = pd.DataFrame()
        df_holdout_2 = df_labelled[df_labelled['url_domain'].isin(unseen_domains_set)]
        df_holdout_2.to_csv(os.path.join(result_dir, f"unseen_{iteration}.csv"), index=False)
        
        df_others_domain = df_labelled[~df_labelled['url_domain'].isin(unseen_domains_set)]
        other_unique_domains = df_others_domain['url_domain'].unique().tolist()

        df_labelled_train = pd.DataFrame()
        df_labelled_test = pd.DataFrame()

        for domain in other_unique_domains:
            print(f"Domain: {domain}")
            df_subset = df_labelled[df_labelled['url_domain'] == domain]
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

        gird_search_Kfold_CV(df_labelled_train, df_labelled_test, result_dir, iteration, log_pred_probability=True)

   

if __name__ == "__main__":
    
    # fullGraph classification
    others_folder = "../../output/others"
    ads_folder = "../../output/ads"
    affiliate_folder = "../../output/affiliate"
 
    RESULT_DIR = "../../output/results/02_28_uniform_Kfold_CV_AFF_OTHERS/"
    if os.path.exists(RESULT_DIR) == False:
        os.makedirs(RESULT_DIR)
    
    RESULT_DIR_phaseA_all = os.path.join(RESULT_DIR, "phase1")
    RESULT_DIR_phaseA_simple = os.path.join(RESULT_DIR, "phase1_simple")
    
    
    # get features
    df_others_phaseA_features_all = pd.DataFrame()
    df_others_phaseA_features_simple = pd.DataFrame()
    df_others_labels = pd.DataFrame()
    df_others_records = pd.DataFrame()
    
    for crawl_id in os.listdir(others_folder):
        visited = ["crawl_aff_normal_10_2", "crawl_aff_normal_260", "crawl_aff_normal_120"]
        if crawl_id not in visited:
            continue
        each_crawl =  os.path.join(others_folder, crawl_id)
        for filename in os.listdir(each_crawl):
            
            # phase A 
            if filename == "features_phase1.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_others_phaseA_features_all = df_others_phaseA_features_all.append(df)

            # phase A simple
            elif filename == "features_phase1_simple.csv": 
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_others_phaseA_features_simple = df_others_phaseA_features_simple.append(df)
            elif filename == "label.csv": 
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_others_labels = df_others_labels.append(df)
            elif filename == "records.csv" or filename == "record.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_others_records = df_others_records.append(df)
            else:
                continue
        
    # remove duplicate clicked others

    print("Before dedepulicate: ", len(df_others_phaseA_features_all))
    df_others_records['parent_visit_id'] = df_others_records['visit_id'].str.split('_', expand=True)[0]
    merged_df = pd.merge(df_others_records, df_others_phaseA_features_all, on='visit_id', how='inner') 
    df_others_phaseA_features_deduplicate =merged_df.drop_duplicates(subset=["name", "num_nodes", "num_edges", "parent_domain", "parent_visit_id"])
    df_others_phaseA_features_all = df_others_phaseA_features_all.merge(df_others_phaseA_features_deduplicate[['visit_id']], on=["visit_id"])
    print("After dedepulicate: ", len(df_others_phaseA_features_deduplicate))
  



    print("Before dedepulicate: ", len(df_others_phaseA_features_simple))
    merged_others_phaseA_simple_df = pd.merge(df_others_records, df_others_phaseA_features_simple, on='visit_id', how='inner') 
    df_others_phaseA_simple_deduplicate =merged_others_phaseA_simple_df.drop_duplicates(subset=["name", "num_nodes", "num_edges", "parent_domain", "parent_visit_id"])
    df_others_phaseA_features_simple = df_others_phaseA_features_simple.merge(df_others_phaseA_simple_deduplicate[['visit_id']], on=["visit_id"])
    print("After dedepulicate: ", len(df_others_phaseA_features_simple))

   
    print("len of others phaseA all features: ", len(df_others_phaseA_features_all))
    print("len of others phaseA simple features: ", len(df_others_phaseA_features_simple))
    print("len of others label ", len(df_others_labels))
    print("len of others records ", len(df_others_records))

    df_affiliate_fullGraph_features_all = pd.DataFrame()
    df_affiliate_fullGraph_features_simple = pd.DataFrame()
    df_affiliate_phaseA_features_all = pd.DataFrame()
    df_affiliate_phaseA_features_simple = pd.DataFrame()
    df_affiliate_labels = pd.DataFrame()
    df_affiliate_records = pd.DataFrame()

    for crawl_id in os.listdir(affiliate_folder):
        if "unseen" in crawl_id:
            print("\tIgnore this folder, since it is for testing")
            continue
        #if crawl_id == "crawl_1" or crawl_id == "crawl_5" or crawl_id == "crawl_6" or crawl_id == "crawl_10":
        #    print("\tIgnore this folder, since it has to much Amazon link")
        #    continue
        each_crawl =  os.path.join(affiliate_folder, crawl_id)
        for filename in os.listdir(each_crawl):
            # phase A 
            if filename == "features_phase1.csv": 
                file_path = os.path.join(each_crawl, filename)
                #print(file_path)
                
                df = pd.read_csv(file_path, on_bad_lines='skip')
                #print("df visit_id: ", df['visit_id'].dtype)
                df['visit_id'] = df['visit_id'].astype(str)
                df_affiliate_phaseA_features_all = df_affiliate_phaseA_features_all.append(df)

            # phase A simple
            elif filename == "features_phase1_simple.csv": 
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df['visit_id'] = df['visit_id'].astype(str)
                df_affiliate_phaseA_features_simple = df_affiliate_phaseA_features_simple.append(df)
            
            elif filename == "label.csv": 
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df['visit_id'] = df['visit_id'].astype(str)
                df_affiliate_labels = df_affiliate_labels.append(df)
            
            elif filename == "records.csv" or filename == "record.csv":
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df['visit_id'] = df['visit_id'].astype(str)
                df_affiliate_records = df_affiliate_records.append(df)
            else:
                continue


    # remove duplicate clicked affiliate
            
    print("Before dedepulicate: ", len(df_affiliate_phaseA_features_all))
    # Split the DataFrame into two based on whether visit_id contains "_"
    df_aff_with_underscore = df_affiliate_records[df_affiliate_records['visit_id'].str.contains('_')]
    df_aff_without_underscore = df_affiliate_records[~df_affiliate_records['visit_id'].str.contains('_')]
    df_aff_with_underscore['parent_visit_id'] = df_aff_with_underscore['visit_id'].str.split('_', expand=True)[0]
    
    merged_df = pd.merge(df_aff_with_underscore, df_affiliate_phaseA_features_all, on='visit_id', how='inner') 
    df_aff_with_underscore_phaseA_features_dedup =merged_df.drop_duplicates(subset=["name", "num_nodes", "num_edges", "parent_domain", "parent_visit_id"])
    df_aff_with_underscore_phaseA_features_dedup = df_affiliate_phaseA_features_all.merge(df_aff_with_underscore_phaseA_features_dedup[['visit_id']], on=["visit_id"])
    df_aff_without_underscore_phaseA_features = df_affiliate_phaseA_features_all.merge(df_aff_without_underscore[['visit_id']], on=["visit_id"])
    df_affiliate_phaseA_features_all = pd.concat([df_aff_with_underscore_phaseA_features_dedup, df_aff_without_underscore_phaseA_features])
    print("After dedepulicate: ", len(df_affiliate_phaseA_features_all))
  


    print("Before dedepulicate: ", len(df_affiliate_phaseA_features_simple))
    merged_aff_phaseA_simple_df = pd.merge(df_aff_with_underscore, df_affiliate_phaseA_features_simple, on='visit_id', how='inner') 
    df_aff_with_underscore_phaseA_simple_dedup =merged_aff_phaseA_simple_df.drop_duplicates(subset=["name", "num_nodes", "num_edges", "parent_domain", "parent_visit_id"])
    df_aff_with_underscore_phaseA_simple_dedup = df_affiliate_phaseA_features_simple.merge(df_aff_with_underscore_phaseA_simple_dedup[['visit_id']], on=["visit_id"])
    df_aff_without_underscore_phaseA_simple = df_affiliate_phaseA_features_simple.merge(df_aff_without_underscore[['visit_id']], on=["visit_id"])
    df_affiliate_phaseA_features_simple = pd.concat([df_aff_with_underscore_phaseA_simple_dedup, df_aff_without_underscore_phaseA_simple])
    
    print("After dedepulicate: ", len(df_affiliate_phaseA_features_simple))

    #print("len of affiliate fullGraph all features: ", len(df_affiliate_fullGraph_features_all))
    #print("len of affiliate fullGraph simple features: ", len(df_affiliate_fullGraph_features_simple))    
    print("len of affiliate phaseA all features: ", len(df_affiliate_phaseA_features_all))
    print("len of affiliate phaseA simple features: ", len(df_affiliate_phaseA_features_simple))
    print("len of affiliate label ", len(df_affiliate_labels))

 
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



    print("Classifying the phaseA all features")
    df_features_phaseA_all = pd.DataFrame()
    df_features_phaseA_all = pd.concat([df_others_phaseA_features_all,df_affiliate_phaseA_features_all])
    pipeline(df_features_phaseA_all, df_labels, df_records, RESULT_DIR_phaseA_all)

    print("\n\nClassifying the phaseA simpler features")
    df_features_phaseA_simple = pd.DataFrame()
    df_features_phaseA_simple = pd.concat([df_others_phaseA_features_simple,df_affiliate_phaseA_features_simple])
    #pipeline(df_features_phaseA_simple, df_labels, df_records, RESULT_DIR_phaseA_simple)
    

