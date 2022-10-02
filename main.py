import pandas as pd


def get_data_train():
    data_train = pd.read_csv('./dataset/train.csv', encoding='gbk')
    print('training dataset shape: ', data_train.shape)
    return data_train


def get_data_test():
    data_test = pd.read_csv('./dataset/test.csv', encoding='gbk')
    print('testing dataset shape: ', data_test.shape)
    return data_test


if __name__ == '__main__':
    get_data_train()
    get_data_test()
