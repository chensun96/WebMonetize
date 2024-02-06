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

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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
                confusion_matrix(y_true, y_pred, labels=["affiliate", "ads"])
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


def classify(train, test, result_dir, tag, sample, log_pred_probability):
    train_mani = train.copy()
    test_mani = test.copy()
    clf = RandomForestClassifier(n_estimators=100)
    # clf = AdaBoostClassifier(n_estimators=100)
    fields_to_remove = ["visit_id", "name", "label", "top_level_url", "Unnamed: 0", "Unnamed: 0_x"]
    df_feature_train = train_mani.drop(fields_to_remove, axis=1, errors="ignore")
    df_feature_test = test_mani.drop(fields_to_remove, axis=1, errors="ignore")
    #df_feature_train.to_csv("/home/data/chensun/affi_project/purl/output/affiliate/fullGraph/df_feature_train.csv")

    columns = df_feature_train.columns
    print("columns: ", columns)
    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_mani.label.to_numpy()

    if sample:
        oversample = RandomOverSampler(sampling_strategy=0.5)
        df_feature_train, train_labels = oversample.fit_resample(
            df_feature_train, train_labels
        )
        undersample = RandomUnderSampler(sampling_strategy=0.5)
        df_feature_train, train_labels = undersample.fit_resample(
            df_feature_train, train_labels
        )

        fname = os.path.join(result_dir, "composition")
        with open(fname, "a") as f:
            counts = collections.Counter(train_labels)
            f.write(
                "\nAfter sampling, new composition: "
                + str(counts["Positive"])
                + " "
                + get_perc(counts["Positive"], len(train_labels))
                + "\n"
            )

    # Perform training
    clf.fit(df_feature_train, train_labels)

    # save the model to disk
    filename = os.path.join(result_dir, "model_" + str(tag) + ".sav")
    pickle.dump(clf, open(filename, "wb"))

    # Obtain feature importances
    feature_importances = pd.DataFrame(
        clf.feature_importances_, index=columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir)

    # Perform classification and get predictions
    cols = df_feature_test.columns
    df_feature_test = df_feature_test.to_numpy()
    y_pred = clf.predict(df_feature_test)

    acc = accuracy_score(test_mani.label, y_pred)
    prec_binary = precision_score(test_mani.label, y_pred, pos_label="affiliate")
    rec_binary = recall_score(test_mani.label, y_pred, pos_label="affiliate")
    prec_micro = precision_score(test_mani.label, y_pred, average="micro")
    rec_micro = recall_score(test_mani.label, y_pred, average="micro")
    prec_macro = precision_score(test_mani.label, y_pred, average="macro")
    rec_macro = recall_score(test_mani.label, y_pred, average="macro")

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
            clf, df_feature_test, cols, test_mani, y_pred, result_dir, tag
        )

    return (
        list(test_mani.label),
        list(y_pred),
        list(test_mani.name),
        list(test_mani.visit_id),
    )


def classify_unknown(df_train, df_test, result_dir):
    train_mani = df_train.copy()
    test_mani = df_test.copy()
    # test_mani = test_mani[test_mani['single'] != "NegBinary"]
    # print(test_mani['single'].value_counts())
    clf = RandomForestClassifier(n_estimators=100)
    # clf = AdaBoostClassifier(n_estimators=100)
    fields_to_remove = ["visit_id", "name", "label", "party", "Unnamed: 0"]
    #'ascendant_script_length', 'ascendant_script_has_fp_keyword',
    #'ascendant_has_ad_keyword',
    #'ascendant_script_has_eval_or_function']
    # 'num_exfil', 'num_infil',
    #                   'num_url_exfil', 'num_header_exfil', 'num_body_exfil',
    #                   'num_ls_exfil', 'num_ls_infil',
    #                   'num_ls_url_exfil', 'num_ls_header_exfil', 'num_ls_body_exfil', 'num_cookieheader_exfil',
    #                   'indirect_in_degree', 'indirect_out_degree',
    #                   'indirect_ancestors', 'indirect_descendants',
    #                   'indirect_closeness_centrality', 'indirect_average_degree_connectivity',
    #                   'indirect_eccentricity', 'indirect_all_in_degree',
    #                   'indirect_all_out_degree', 'indirect_all_ancestors',
    #                   'indirect_all_descendants', 'indirect_all_closeness_centrality',
    #                   'indirect_all_average_degree_connectivity', 'indirect_all_eccentricity'
    # ]
    #'num_nodes', 'num_edges',
    #'nodes_div_by_edges', 'edges_div_by_nodes']
    df_feature_train = train_mani.drop(fields_to_remove, axis=1, errors="ignore")
    df_feature_test = test_mani.drop(fields_to_remove, axis=1, errors="ignore")

    columns = df_feature_train.columns
    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_mani.label.to_numpy()

    # Perform training
    clf.fit(df_feature_train, train_labels)

    # Obtain feature importances
    feature_importances = pd.DataFrame(
        clf.feature_importances_, index=columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    report_feature_importance(feature_importances, result_dir)

    df_feature_test = df_feature_test.to_numpy()
    y_pred = clf.predict(df_feature_test)
    y_pred = list(y_pred)
    name = list(test_mani.name)
    vid = list(test_mani.visit_id)

    fname = os.path.join(result_dir, "predictions")
    with open(fname, "w") as f:
        for i in range(0, len(y_pred)):
            f.write("%s |$| %s |$| %s\n" % (y_pred[i], name[i], vid[i]))

    preds, bias, contributions = ti.predict(clf, df_feature_test)
    fname = os.path.join(result_dir, "interpretations")
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
            fn = list(columns)
            fn = [str(x) for x in fn]
            feature_contribution = list(zip(c, fn))
            # feature_contribution = list(zip(contributions[i,:,0], df_feature_test.columns))
            data_dict[key]["contributions"] = feature_contribution
        f.write(json.dumps(data_dict, indent=4))


def classify_validation(
    df_train, df_validation, result_dir, sample=False, log_pred_probability=False
):
    i = 0
    result = classify(
        df_train, df_validation, result_dir, i, sample, log_pred_probability
    )
    results = [result]

    return results


def classify_crossval(
    df_labelled, result_dir, sample=False, log_pred_probability=False
):
    vid_list = df_labelled["visit_id"].unique()
    num_iter = 10
    num_test_vid = int(len(vid_list) / num_iter)
    print("VIDs", len(vid_list))
    print("To use!", num_test_vid)
    used_test_ids = []
    results = []

    for i in range(0, num_iter):
        print("Fold", i)
        vid_list_iter = list(set(vid_list) - set(used_test_ids))
        chosen_test_vid = random.sample(vid_list_iter, num_test_vid)
        used_test_ids += chosen_test_vid

        df_train = df_labelled[~df_labelled["visit_id"].isin(chosen_test_vid)]
        df_test = df_labelled[df_labelled["visit_id"].isin(chosen_test_vid)]

        fname = os.path.join(result_dir, "composition")
        train_pos = len(df_train[df_train["label"] == "affiliate"])
        test_pos = len(df_test[df_test["label"] == "affiliate"])

        with open(fname, "a") as f:
            f.write("\nFold " + str(i) + "\n")
            f.write(
                "Train: "
                + str(train_pos)
                + " "
                + get_perc(train_pos, len(df_train))
                + "\n"
            )
            f.write(
                "Test: " + str(test_pos) + " " + get_perc(test_pos, len(df_test)) + "\n"
            )
            f.write("\n")

        result = classify(
            df_train, df_test, result_dir, i, sample, log_pred_probability
        )
        results.append(result)

    return results


def get_perc(num, den):
    return str(round(num / den * 100, 2)) + "%"

def label_party(name):
    parts = name.split("||")

    if get_domain(parts[0].strip()) == get_domain(parts[1].strip()):
        return "First"
    else:
        return "Third"

def gird_search(df_labelled, df_labelled_holdout, result_dir, log_pred_probability):
    train_mani = df_labelled.copy()
    holdout_mani = df_labelled_holdout.copy()

    fields_to_remove = ["visit_id", "name", "label", "party", "Unnamed: 0", 'top_level_url', 'Unnamed: 0_x', 'Unnamed: 0_y']
    
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
        'n_estimators': [100,150,200,250,300], # number of trees in the forest
        'max_features': [None,'sqrt'],   # consider every features /square root of features
        'max_depth': [5, 10, 20],
        'min_samples_split': [5, 10, 15],  # minimum number of samples that are required to split an internal node.
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    # Initialize the classifier
    rf = RandomForestClassifier()

    # Initialize Grid Search with 10-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(df_feature_train, train_labels)

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
    print(best_model.classes_)  # e.g., ['ads' 'affiliate']


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



def pipeline(df, df_labels, result_dir):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # how to merge label and features based on name 
    df_labels = df_labels.drop_duplicates(subset=['visit_id'])
    print("df_labels: ", len(df_labels))

    # drop "top_level_url" column
    new_df_labels = df_labels.drop('top_level_url', axis=1)

    df = df.merge(new_df_labels[['visit_id', 'label', 'name']], on=["visit_id"])
    #df.to_csv("/home/data/chensun/affi_project/purl/output/test_1.csv")
    

    # only need to drop label_y in phaseA?
    #df.drop(columns=["label_y"], inplace=True)
    #df.rename(columns={"label_x": "label"}, inplace=True)

    df.drop(columns=["name_y", "Unnamed: 0_x" ,"Unnamed: 0_y"], inplace=True)
    df.rename(columns={"name_x": "name"}, inplace=True)
    df.to_csv("/home/data/chensun/affi_project/purl/output/test_2.csv")

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    print(df['label'].value_counts())
    print(len(df))
    df_labelled = df
    df_positive = df[df["label"] == "affiliate"]
    df_negative = df[df["label"] == "ads"]
    df_unknown = df[df["label"] == "normal"]
    # find nan values
    print("Nan values")
    print(df.isnull().values.any())
   
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
            "Negative samples (ads): "
            + str(len(df_negative))
            + " "
            + get_perc(len(df_negative), len(df))
            + "\n"
        )
        f.write("\n")
        
    # sample negative labels to match positive labels
    # df_negative = df_negative.sample(n=len(df_positive), random_state=1)
    
    # sample positive labels to match negative labels
    df_positive = df_positive.sample(n=len(df_negative), random_state=1)
    df_labelled = pd.concat([df_positive, df_negative])
    vid_list = df_labelled["visit_id"].unique()
    print("vid_list: ", len(vid_list))

    
    
    # [Added] prepare for holdout data set
    
    num_test_vid_holdout = int(int(len(vid_list))*0.8)
    chosen_test_vid_holdout = random.sample(list(vid_list), num_test_vid_holdout)
    #vid_list_holdout = chosen_test_vid_holdout["visit_id"].unique()
    print("vid_list (without holdout): ", len(chosen_test_vid_holdout))
    df_labelled_crossval = df_labelled[df_labelled["visit_id"].isin(chosen_test_vid_holdout)]
    df_labelled_holdout = df_labelled[~df_labelled["visit_id"].isin(chosen_test_vid_holdout)]
    
    gird_search(df_labelled, df_labelled_holdout, result_dir, log_pred_probability=True)
    
    #results = classify_crossval(
    #    df_labelled_crossval, result_dir, sample=False, log_pred_probability=True
    #)
    #report = describe_classif_reports(results, result_dir)
    # print(report)
    # print_stats(report, result_dir)

    #valid_result_dir = os.path.join(result_dir, "validation")
    #os.mkdir(valid_result_dir)
    #results = classify_validation(df_crossval, df_validation, valid_result_dir, sample=False, log_pred_probability=True)
    #report = describe_classif_reports(results, valid_result_dir)

    # Unknown labels
    # unknown_result_dir = os.path.join(result_dir, "unlabelled")
    # os.mkdir(unknown_result_dir)
    # classify_unknown(df_labelled, df_unknown, unknown_result_dir)
   

if __name__ == "__main__":
    
    # fullGraph classification
    normal_folder = "../../output/normal"
    ads_folder = "../../output/ads"
    affiliate_folder = "../../output/affiliate"

    #RESULT_DIR_fullGraph_all = "../../output/results/01_31/fullGraph_all"
    #RESULT_DIR_fullGraph_simple = "../../output/results/01_31/fullGraph_simple" 
    RESULT_DIR_phaseA_all = "../../output/results/01_31/phaseA_all"
    RESULT_DIR_phaseA_simple = "../../output/results/01_31/phaseA_simple"
    
    # get features
    df_ads_fullGraph_features_all = pd.DataFrame()
    df_ads_fullGraph_features_simple = pd.DataFrame()
    df_ads_phaseA_features_all = pd.DataFrame()
    df_ads_phaseA_features_simple = pd.DataFrame()
    df_ads_labels = pd.DataFrame()
    
    for crawl_id in os.listdir(ads_folder):
        if "unseen" in crawl_id:
                print("\tIgnore this folder, since it is for testing")
                continue
        each_crawl =  os.path.join(ads_folder, crawl_id)
        for filename in os.listdir(each_crawl):
            # full graph
            if filename == "features_fullGraph.csv": 
                continue
                # Construct the full file path
                #file_path = os.path.join(each_crawl, filename)
                #df = pd.read_csv(file_path, on_bad_lines='skip')
                #df_ads_fullGraph_features_all = df_ads_fullGraph_features_all.append(df)

            # full graph simple
            elif filename == "features_fullGraph_simple.csv": 
                continue
                # Construct the full file path
                #file_path = os.path.join(each_crawl, filename)
                #df = pd.read_csv(file_path, on_bad_lines='skip')
                #df_ads_fullGraph_features_simple = df_ads_fullGraph_features_simple.append(df)

            # phase A 
            elif filename == "features_phase1.csv": 
                # Construct the full file path
                file_path = os.path.join(each_crawl, filename)
                print(file_path)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_ads_phaseA_features_all = df_ads_phaseA_features_all.append(df)

            # phase A simple
            elif filename == "features_phase1_simple.csv": 
                # Construct the full file path
                
                file_path = os.path.join(each_crawl, filename)
                print(file_path)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_ads_phaseA_features_simple = df_ads_phaseA_features_simple.append(df)
            elif filename == "label.csv": 
                # Construct the full file path
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_ads_labels = df_ads_labels.append(df)
            else:
                continue
        

    print("len of ads fullGraph all features: ", len(df_ads_fullGraph_features_all))
    print("len of ads fullGraph simple features: ", len(df_ads_fullGraph_features_simple))    
    print("len of ads phaseA all features: ", len(df_ads_phaseA_features_all))
    print("len of ads phaseA simple features: ", len(df_ads_phaseA_features_simple))
    print("len of ads label ", len(df_ads_labels))


    df_affiliate_fullGraph_features_all = pd.DataFrame()
    df_affiliate_fullGraph_features_simple = pd.DataFrame()
    df_affiliate_phaseA_features_all = pd.DataFrame()
    df_affiliate_phaseA_features_simple = pd.DataFrame()
    df_affiliate_labels = pd.DataFrame()

    for crawl_id in os.listdir(affiliate_folder):
        if "unseen" in crawl_id:
            print("\tIgnore this folder, since it is for testing")
            continue
        if crawl_id == "crawl_1" or crawl_id == "crawl_5" or crawl_id == "crawl_6" or crawl_id == "crawl_10":
            print("\tIgnore this folder, since it has to much Amazon link")
            continue
        each_crawl =  os.path.join(affiliate_folder, crawl_id)
        for filename in os.listdir(each_crawl):
            # full graph
            if filename == "features_fullGraph.csv": 
                # Construct the full file path
                continue
                #file_path = os.path.join(each_crawl, filename)
                #df = pd.read_csv(file_path, on_bad_lines='skip')
                #df_affiliate_fullGraph_features_all = df_affiliate_fullGraph_features_all.append(df)

            # full graph simple
            elif filename == "features_fullGraph_simple.csv": 
                continue
                # Construct the full file path
                #file_path = os.path.join(each_crawl, filename)
                #df = pd.read_csv(file_path, on_bad_lines='skip')
                #df_affiliate_fullGraph_features_simple = df_affiliate_fullGraph_features_simple.append(df)

            # phase A 
            elif filename == "features_phase1.csv": 
                # Construct the full file path
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_affiliate_phaseA_features_all = df_affiliate_phaseA_features_all.append(df)

            # phase A simple
            elif filename == "features_phase1_simple.csv": 
                # Construct the full file path
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_affiliate_phaseA_features_simple = df_affiliate_phaseA_features_simple.append(df)
            elif filename == "label.csv": 
                # Construct the full file path
                file_path = os.path.join(each_crawl, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df_affiliate_labels = df_affiliate_labels.append(df)
            else:
                continue

    print("len of affiliate fullGraph all features: ", len(df_affiliate_fullGraph_features_all))
    print("len of affiliate fullGraph simple features: ", len(df_affiliate_fullGraph_features_simple))    
    print("len of affiliate phaseA all features: ", len(df_affiliate_phaseA_features_all))
    print("len of affiliate phaseA simple features: ", len(df_affiliate_phaseA_features_simple))
    print("len of affiliate label ", len(df_affiliate_labels))

    
    df_labels_fullGraph = pd.DataFrame()
    df_labels_fullGraph = pd.concat([df_ads_labels,df_affiliate_labels])

    
    #print("Classifying the fullGraph all features")
    #df_features_fullGraph_all = pd.DataFrame()
    #df_features_fullGraph_all = pd.concat([df_ads_fullGraph_features_all,df_affiliate_fullGraph_features_all])
    #pipeline(df_features_fullGraph_all, df_labels_fullGraph, RESULT_DIR_fullGraph_all)


    #print("Classifying the fullGraph simpler features")
    #df_features_fullGraph_simple = pd.DataFrame()
    #df_features_fullGraph_simple = pd.concat([df_ads_fullGraph_features_simple,df_affiliate_fullGraph_features_simple])
    #pipeline(df_features_fullGraph_simple, df_labels_fullGraph, RESULT_DIR_fullGraph_simple)


    print("Classifying the phaseA all features")
    df_features_phaseA_all = pd.DataFrame()
    df_features_phaseA_all = pd.concat([df_ads_phaseA_features_all,df_affiliate_phaseA_features_all])
    pipeline(df_features_phaseA_all, df_labels_fullGraph, RESULT_DIR_phaseA_all)

    print("Classifying the phaseA simpler features")
    df_features_phaseA_simple = pd.DataFrame()
    df_features_phaseA_simple = pd.concat([df_ads_phaseA_features_simple,df_affiliate_phaseA_features_simple])
    pipeline(df_features_phaseA_simple, df_labels_fullGraph, RESULT_DIR_phaseA_simple)
    


    



    """
    dfs = []
    for filename in os.listdir(affiliate_fullGraph_folder):
        if filename.startswith("features_fullGraph") and filename.endswith(".csv"): # change this
            if "test" in filename:
                print("ignore the test file")
                continue
            file_path = os.path.join(affiliate_fullGraph_folder, filename)
            df = pd.read_csv(file_path, on_bad_lines='skip')
            dfs.append(df)

    # Concatenate all DataFrames and reset the index
    df_affiliate_features = pd.concat(dfs, ignore_index=True)

    # Step 2: Reduce Amazon affiliate links
    
    is_amazon = df_affiliate_features['name'].str.contains('amazon', case=False, na=False)
    amazon_count = is_amazon.sum()
    # Randomly select 1/10th of amazon link
    fraction_to_select = 0.1
    selected_amazon_indices = np.random.choice(
        df_affiliate_features[is_amazon].index,
        size=int(amazon_count * fraction_to_select),
        replace=False
    )
    non_amazon_df = df_affiliate_features[~is_amazon]
    selected_amazon_df = df_affiliate_features.loc[selected_amazon_indices]
    reduced_df_aff_features = pd.concat([non_amazon_df, selected_amazon_df])
    visit_ids = reduced_df_aff_features['visit_id']
    #print("visit_ids: ", len(visit_ids))
    #print("Original size:", len(df_features))
    #print("Reduced size:", len(reduced_df_features))
    print("len of aff features: ", len(reduced_df_aff_features))
    df_features = pd.DataFrame()
    df_features = pd.concat([df_ads_features,reduced_df_aff_features])
    print("len of all features: ", len(df_features))
    
    # get labels
    df_ads_labels = pd.DataFrame()
    
    for filename in os.listdir(ads_fullGraph_folder):
        if filename.startswith("label") and filename.endswith(".csv"):
            # Construct the full file path
            if "test" in filename:
                print("ignore the test file")
                continue
            file_path = os.path.join(ads_fullGraph_folder, filename)
            df = pd.read_csv(file_path, on_bad_lines='skip')
            print(len(df))
            df_ads_labels = df_ads_labels.append(df)
    print("len of ads label: ", len(df_ads_labels))


    df_aff_labels = pd.DataFrame()  # Make sure df_labels is initialized
    for filename in os.listdir(affiliate_fullGraph_folder):
        if filename.startswith("label") and filename.endswith(".csv"):
            if "test" in filename:
                print("ignore the test file")
                continue
            file_path = os.path.join(affiliate_fullGraph_folder, filename)
            df = pd.read_csv(file_path, on_bad_lines='skip')
            #print("File:", filename, "Length:", len(df))
            df_aff_labels = df_aff_labels.append(df)

    filtered_df_labels = df_aff_labels[df_aff_labels['visit_id'].isin(visit_ids)]
    print("len of aff labels: ", len(filtered_df_labels))

    df_labels = pd.DataFrame()
    df_labels = pd.concat([df_ads_labels,filtered_df_labels])
    print("len of all labels: ", len(df_labels))
    """


   