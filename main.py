import pandas as pd


def get_data_train():
    return pd.read_csv('./dataset/train.csv', encoding='gbk')


def get_data_test():
    return pd.read_csv('./dataset/test.csv', encoding='gbk')


if __name__ == '__main__':
    #训练集
    train_data = get_data_train()
    #训练集行数
    train_data_row_num = train_data.shape[0]
    #训练集列数
    train_data_column_num = train_data.shape[1]
    print("训练集行数: " + str(train_data_row_num) + ", 训练集列数: " + str(train_data_column_num))

    #测试集
    test_data = get_data_test()
    #测试集行数
    test_data_row_num = test_data.shape[0]
    #测试集列数
    test_data_column_num = test_data.shape[1]
    print("测试集行数: " + str(test_data_row_num) + ", 测试集列数: " + str(test_data_column_num))

    #训练集每列数据类型
    train_data_columns = train_data.columns
    for column in train_data_columns:
        print("列: " + str(column) + ", 数据类型: " + str(train_data[column].dtype))

