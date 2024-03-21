import numpy as np
import seaborn as sns
import Utils as utils
from DataLoader import get_data
from tensorflow.keras.models import load_model

sns.set_style("darkgrid")

def main():
    train, test, label_mapping = get_data(encoding="Label")

    train.drop(columns=['Label'], inplace=True)
    test.drop(columns=['Label'], inplace=True)

    data_cols = list(train.columns[train.columns != 'label'])

    y_train = train.label.values
    y_test = test.label.values

    att_ind = np.where(train.label != label_mapping[0])[0]
    for_test = np.where(test.label != label_mapping[0])[0]

    del label_mapping[0]

    x = train[data_cols].values[att_ind]
    y = y_train[att_ind]

    x_test = test[data_cols].values[att_ind]
    y_test = y_test[att_ind]

    # Load pretrained ml classifiers
    ml_classifiers = utils.load_pretrained_classifiers()

    # Load trained GAN generator model
    model = load_model('./trained_generators/gen.h5')
    print("pretrained generator model load : [DONE]")

    # Genetare new data samples, fit ML models compare perfomance with ML models before data balancing
    utils.compare_classifiers(x, y, x_test[data_cols].values[for_test], y_test[for_test], model, label_mapping,
                              ml_classifiers, cv=5)


if __name__ == '__main__':
    main()
