import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")


from sklearn.preprocessing import LabelEncoder
import json

def get_data(encoding='Label', data_folder="./"):
    """
    Retrive Train and Test data
    """
    label_col = 'Label'

    with open('label_dict.json') as json_file:
        label_dict = json.load(json_file)

    train = pd.read_csv(data_folder + "preprocessed/train_df.csv")
    test = pd.read_csv(data_folder + "preprocessed/test_df.csv")
    attack_types = list(label_dict.keys())
    map_type = pd.Series(index=attack_types, data=np.arange(len(attack_types))).to_dict()
    labels = train[label_col].map(label_dict).map(map_type).values
    train['label'] = labels

    labels = test[label_col].map(label_dict).map(map_type).values
    test['label'] = labels

    le = LabelEncoder()
    le.fit(train.label)
    label_mapping = {l: i for i, l in enumerate(le.classes_)}

    return train, test, label_mapping
