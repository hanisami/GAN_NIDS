import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import os

for dirname, _, filenames in os.walk('CIC-IDS-2018'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")


def fix_data_type(df):
    df = df[df['Dst Port'] != 'Dst Port']

    df['Dst Port'] = df['Dst Port'].astype(int)
    df['Protocol'] = df['Protocol'].astype(int)
    df['Flow Duration'] = df['Flow Duration'].astype(int)
    df['Tot Fwd Pkts'] = df['Tot Fwd Pkts'].astype(int)
    df['Tot Bwd Pkts'] = df['Tot Bwd Pkts'].astype(int)
    df['TotLen Fwd Pkts'] = df['TotLen Fwd Pkts'].astype(int)
    df['TotLen Bwd Pkts'] = df['TotLen Bwd Pkts'].astype(int)
    df['Fwd Pkt Len Max'] = df['Fwd Pkt Len Max'].astype(int)
    df['Fwd Pkt Len Min'] = df['Fwd Pkt Len Min'].astype(int)
    df['Fwd Pkt Len Mean'] = df['Fwd Pkt Len Mean'].astype(float)
    df['Fwd Pkt Len Std'] = df['Fwd Pkt Len Std'].astype(float)
    df['Bwd Pkt Len Max'] = df['Bwd Pkt Len Max'].astype(int)
    df['Bwd Pkt Len Min'] = df['Bwd Pkt Len Min'].astype(int)
    df['Bwd Pkt Len Mean'] = df['Bwd Pkt Len Mean'].astype(float)
    df['Bwd Pkt Len Std'] = df['Bwd Pkt Len Std'].astype(float)
    df['Flow Byts/s'] = df['Flow Byts/s'].astype(float)
    df['Flow Pkts/s'] = df['Flow Pkts/s'].astype(float)
    df['Flow IAT Mean'] = df['Flow IAT Mean'].astype(float)
    df['Flow IAT Std'] = df['Flow IAT Std'].astype(float)
    df['Flow IAT Max'] = df['Flow IAT Max'].astype(int)
    df['Flow IAT Min'] = df['Flow IAT Min'].astype(int)
    df['Fwd IAT Tot'] = df['Fwd IAT Tot'].astype(int)
    df['Fwd IAT Mean'] = df['Fwd IAT Mean'].astype(float)
    df['Fwd IAT Std'] = df['Fwd IAT Std'].astype(float)
    df['Fwd IAT Max'] = df['Fwd IAT Max'].astype(int)
    df['Fwd IAT Min'] = df['Fwd IAT Min'].astype(int)
    df['Bwd IAT Tot'] = df['Bwd IAT Tot'].astype(int)
    df['Bwd IAT Mean'] = df['Bwd IAT Mean'].astype(float)
    df['Bwd IAT Std'] = df['Bwd IAT Std'].astype(float)
    df['Bwd IAT Max'] = df['Bwd IAT Max'].astype(int)
    df['Bwd IAT Min'] = df['Bwd IAT Min'].astype(int)
    df['Fwd PSH Flags'] = df['Fwd PSH Flags'].astype(int)
    df['Bwd PSH Flags'] = df['Bwd PSH Flags'].astype(int)
    df['Fwd URG Flags'] = df['Fwd URG Flags'].astype(int)
    df['Bwd URG Flags'] = df['Bwd URG Flags'].astype(int)
    df['Fwd Header Len'] = df['Fwd Header Len'].astype(int)
    df['Bwd Header Len'] = df['Bwd Header Len'].astype(int)
    df['Fwd Pkts/s'] = df['Fwd Pkts/s'].astype(float)
    df['Bwd Pkts/s'] = df['Bwd Pkts/s'].astype(float)
    df['Pkt Len Min'] = df['Pkt Len Min'].astype(int)
    df['Pkt Len Max'] = df['Pkt Len Max'].astype(int)
    df['Pkt Len Mean'] = df['Pkt Len Mean'].astype(float)
    df['Pkt Len Std'] = df['Pkt Len Std'].astype(float)
    df['Pkt Len Var'] = df['Pkt Len Var'].astype(float)
    df['FIN Flag Cnt'] = df['FIN Flag Cnt'].astype(int)
    df['SYN Flag Cnt'] = df['SYN Flag Cnt'].astype(int)
    df['RST Flag Cnt'] = df['RST Flag Cnt'].astype(int)
    df['PSH Flag Cnt'] = df['PSH Flag Cnt'].astype(int)
    df['ACK Flag Cnt'] = df['ACK Flag Cnt'].astype(int)
    df['URG Flag Cnt'] = df['URG Flag Cnt'].astype(int)
    df['CWE Flag Count'] = df['CWE Flag Count'].astype(int)
    df['ECE Flag Cnt'] = df['ECE Flag Cnt'].astype(int)
    df['Down/Up Ratio'] = df['Down/Up Ratio'].astype(int)
    df['Pkt Size Avg'] = df['Pkt Size Avg'].astype(float)
    df['Fwd Seg Size Avg'] = df['Fwd Seg Size Avg'].astype(float)
    df['Bwd Seg Size Avg'] = df['Bwd Seg Size Avg'].astype(float)
    df['Fwd Byts/b Avg'] = df['Fwd Byts/b Avg'].astype(int)
    df['Fwd Pkts/b Avg'] = df['Fwd Pkts/b Avg'].astype(int)
    df['Fwd Blk Rate Avg'] = df['Fwd Blk Rate Avg'].astype(int)
    df['Bwd Byts/b Avg'] = df['Bwd Byts/b Avg'].astype(int)
    df['Bwd Pkts/b Avg'] = df['Bwd Pkts/b Avg'].astype(int)
    df['Bwd Blk Rate Avg'] = df['Bwd Blk Rate Avg'].astype(int)
    df['Subflow Fwd Pkts'] = df['Subflow Fwd Pkts'].astype(int)
    df['Subflow Fwd Byts'] = df['Subflow Fwd Byts'].astype(int)
    df['Subflow Bwd Pkts'] = df['Subflow Bwd Pkts'].astype(int)
    df['Subflow Bwd Byts'] = df['Subflow Bwd Byts'].astype(int)
    df['Init Fwd Win Byts'] = df['Init Fwd Win Byts'].astype(int)
    df['Init Bwd Win Byts'] = df['Init Bwd Win Byts'].astype(int)
    df['Fwd Act Data Pkts'] = df['Fwd Act Data Pkts'].astype(int)
    df['Fwd Seg Size Min'] = df['Fwd Seg Size Min'].astype(int)
    df['Active Mean'] = df['Active Mean'].astype(float)
    df['Active Std'] = df['Active Std'].astype(float)
    df['Active Max'] = df['Active Max'].astype(int)
    df['Active Min'] = df['Active Min'].astype(int)
    df['Idle Mean'] = df['Idle Mean'].astype(float)
    df['Idle Std'] = df['Idle Std'].astype(float)
    df['Idle Max'] = df['Idle Max'].astype(int)
    df['Idle Min'] = df['Idle Min'].astype(int)

    return df


def drop_infinate_null(df):
    print(df.shape)

    # replace infinity value as null value
    df = df.replace(["Infinity", "infinity"], np.inf)
    df = df.replace([np.inf, -np.inf], np.nan)

    # drop all null values
    df.dropna(inplace=True)

    # print(df.shape)

    return df


def drop_unnecessary_column(df):
    df.drop(columns="Timestamp", inplace=True)
    # print (df.shape)
    return df


def generate_binary_label(df):
    # encode the target feature
    df['Threat'] = df['Label'].apply(lambda x: "Benign" if x == 'Benign' else "Malicious")
    # print(df['Threat'].unique())
    # print(df['Threat'].value_counts())
    return df


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []
    for col in props.columns:
        if props[col].dtype != object:

            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            else:
                props[col] = props[col].astype(np.float32)

    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    # print("Memory usage is: ", mem_usg, " MB")
    # print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist


def transform_multi_label(df):
    mapping = {'SSH-Bruteforce': 'Brute-force',
               'FTP-BruteForce': 'Brute-force',
               ################ Brute-force

               'Brute Force -XSS': 'Web attack',
               'Brute Force -Web': 'Web attack',
               'SQL Injection': 'Web attack',
               ################ Web attack

               'DoS attacks-Hulk': 'DoS attack',
               'DoS attacks-SlowHTTPTest': 'DoS attack',
               'DoS attacks-Slowloris': 'DoS attack',
               'DoS attacks-GoldenEye': 'DoS attack',
               ################ DoS attack

               'DDOS attack-HOIC': 'DDoS attack',
               'DDOS attack-LOIC-UDP': 'DDoS attack',
               'DDoS attacks-LOIC-HTTP': 'DDoS attack',
               ################ DDoS attack

               'Bot': 'Botnet',
               ################ Botnet

               'Infilteration': 'Infilteration',
               ################ Infilteration

               'Benign': 'Benign',
               'Label': 'Benign',
               ################ Infilteration
               }
    print(df['Label'])
    df['Label'] = df['Label'].map(mapping)
    return df


def balance_data(df):
    X = df.drop(["Label"], axis=1)
    y = df["Label"]

    rus = RandomUnderSampler()
    X_balanced, y_balanced = rus.fit_resample(X, y)

    df = pd.concat([X_balanced, y_balanced], axis=1)
    del X, y, X_balanced, y_balanced
    print(df.shape)
    print(df['Label'].value_counts())

    return df


def dataframe_preprocessing(df):
    df = fix_data_type(df)
    df = drop_infinate_null(df)
    df = drop_unnecessary_column(df)
    df = generate_binary_label(df)
    df, _ = reduce_mem_usage(df)
    df = transform_multi_label(df)
    df = balance_data(df)
    return df


def main():
    # df_d1 = pd.read_csv("CIC-IDS-2018/02-14-2018.csv", low_memory=False)
    # df_d2 = pd.read_csv("CIC-IDS-2018/02-15-2018.csv", low_memory=False)
    # df_d3 = pd.read_csv("CIC-IDS-2018/02-16-2018.csv", low_memory=False)
    df_d4 = pd.read_csv("CIC-IDS-2018/02-20-2018.csv", low_memory=False)
    # df_d5 = pd.read_csv("CIC-IDS-2018/02-21-2018.csv", low_memory=False)
    # df_d6 = pd.read_csv("CIC-IDS-2018/02-22-2018.csv", low_memory=False)
    # df_d7 = pd.read_csv("CIC-IDS-2018/02-23-2018.csv", low_memory=False)
    # df_d8 = pd.read_csv("CIC-IDS-2018/02-28-2018.csv", low_memory=False)
    # df_d9 = pd.read_csv("CIC-IDS-2018/03-01-2018.csv", low_memory=False)
    # df_d10 = pd.read_csv("CIC-IDS-2018/03-02-2018.csv", low_memory=False)

    df_d4.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP'], axis=1, inplace=True)

    df_all = dataframe_preprocessing(df_d4)
    variances = df_all.var(numeric_only=True)
    constant_columns = variances[variances == 0].index
    df_all = df_all.drop(constant_columns, axis=1)

    duplicates = set()
    for i in range(0, len(df_all.columns)):
        col1 = df_all.columns[i]
        for j in range(i + 1, len(df_all.columns)):
            col2 = df_all.columns[j]
            if (df_all[col1].equals(df_all[col2])):
                duplicates.add(col2)

    df_all.drop(duplicates, axis=1, inplace=True)

    corr = df_all.corr(numeric_only=True)

    correlated_col = set()
    is_correlated = [True] * len(corr.columns)
    threshold = 0.90
    for i in range(len(corr.columns)):
        if (is_correlated[i]):
            for j in range(i):
                if (np.abs(corr.iloc[i, j]) >= threshold) and (is_correlated[j]):
                    colname = corr.columns[j]
                    is_correlated[j] = False
                    correlated_col.add(colname)

    df_all.drop(correlated_col, axis=1, inplace=True)

    label_col = "Label"
    feature_cols = list(df_all.columns)
    feature_cols.remove("Threat")
    feature_cols.remove(label_col)

    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=2, shuffle=True,
                                         stratify=df_all[label_col])

    del df_all

    minmax_scaler = MinMaxScaler()
    train_df[feature_cols] = minmax_scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = minmax_scaler.transform(test_df[feature_cols])

    order_label_list = list(np.unique(train_df[label_col]))

    label_dict = {v: v for v in order_label_list}

    with open("preprocessed/label_dict.json", "wb") as outfile:
        json.dump(label_dict, outfile)

    train_df[feature_cols + [label_col]].to_csv("preprocessed/train_df.csv", index=False)
    test_df[feature_cols + [label_col]].to_csv("preprocessed/test_df.csv", index=False)

    del train_df, test_df


main()
