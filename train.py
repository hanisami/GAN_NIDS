from DataLoader import get_data
import numpy as np
import seaborn as sns
import Classifiers as clfrs
import Utils as utils
import CGAN as cgan

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

    gan_params = [32, 3, 100, 256, 1, 1, 'spocu', 'sgd', 0.0006, 5]

    # train Ml classifiers
    print("Training classifiers : [Started]")
    svm = clfrs.svm(x, y, test[data_cols].values[for_test], y_test[for_test], label_mapping, True)
    randf = clfrs.random_forest(x, y, test[data_cols].values[for_test], y_test[for_test], label_mapping)
    nn = clfrs.neural_network(x, y, test[data_cols].values[for_test], y_test[for_test], label_mapping, True)
    deci = clfrs.decision_tree(x, y, test[data_cols].values[for_test], y_test[for_test], label_mapping)
    print("Training classifiers : [Finished]")

    utils.save_classifiers([randf, nn, deci, svm])
    print("Classifiers save to disk : [SUCCESSFUL]")

    # Define, Train & Save GAN
    print("GAN Training Starting ....")
    model = cgan.CGAN(gan_params, x, y.reshape(-1, 1))
    model.train()
    model.dump_to_file()
    print("GAN Training Finised!")

    # Plot GAN training logs
    gan_path = f"./logs/CGAN_{model.gan_name}.pickle"
    utils.plot_training_summary(gan_path, './imgs')


if __name__ == '__main__':
    main()
