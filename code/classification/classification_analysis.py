import pandas as pd
from tqdm.auto import tqdm
import os 
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
# plot the data
import seaborn as sns
import scienceplots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

def count_decoration(df_labeled):
    # plt.figure(figsize=(8, 6))
    count, bins_negative = np.histogram(df_labeled[df_labeled.counts_decoration < 120][df_labeled.label == 'normal'].counts_decoration, 24)
    pdf_negative = count / sum(count)
    cdf_negative = np.cumsum(pdf_negative)
    plt.plot(bins_negative[1:], cdf_negative, label="normal", color='green')

    count, bins_positive = np.histogram(df_labeled[df_labeled.counts_decoration < 120][df_labeled.label == 'affiliate'].counts_decoration, 24)
    pdf_positive = count / sum(count)
    cdf_positive = np.cumsum(pdf_positive)
    plt.plot(bins_positive[1:], cdf_positive, label="affiliate", color='red')
    plt.legend()
    plt.xlabel("Number of decoration")
    plt.ylabel("CDF")
    plt.savefig("/home/data/chensun/affi_project/purl/output/results/aff_normal_mean_phaseA/num_decoration_cdf.pdf", dpi=300, bbox_inches='tight')

    df_labeled[df_labeled.label == 'Positive'].num_exfil.describe()




if __name__ == "__main__":

    # Accuracy when apply classification with model
    RESULT_DIR = "../../output/results/aff_ads_graph_level_fullGraph_3/with_model_1/labelled_results.csv"
    # RESULT_DIR = "../../output/results/aff_normal_mean_phaseA/labelled_results.csv"
    df = pd.read_csv(RESULT_DIR)

    # Assuming df is your DataFrame with the 'label' as the true labels
    # and 'clabel' as the predicted labels
    y_true = df['label']
    y_pred = df['clabel']

    # Calculate the accuracy
    accuracy = accuracy_score(y_true, y_pred)

    #acc = accuracy_score(test_mani.label, y_pred)
    #prec_binary = precision_score(y_true.label, y_pred, pos_label="affiliate")
    #rec_binary = recall_score(y_true.label, y_pred, pos_label="affiliate")
    #prec_micro = precision_score(y_true.label, y_pred, average="micro")
    #rec_micro = recall_score(y_true.label, y_pred, average="micro")
    #prec_macro = precision_score(y_true.label, y_pred, average="macro")
    #rec_macro = recall_score(y_true.label, y_pred, average="macro")

    print(f'Accuracy: {accuracy:.2f}')  
    

    