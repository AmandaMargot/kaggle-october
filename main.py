import pandas as pd
import matplotlib.pyplot as plt


def get_data_train():
    return pd.read_csv('./dataset/train.csv', encoding='gbk')


def get_data_test():
    return pd.read_csv('./dataset/test.csv', encoding='gbk')


def get_column_and_row_num(data):
    # 行数
    row_num = data.shape[0]
    # 列数
    column_num = data.shape[1]
    print("行数: " + str(row_num) + ", 列数: " + str(column_num))


def get_data_type(data):
    data_columns = data.columns
    for column in data_columns:
        print("列: " + str(column) + ", 数据类型: " + str(data[column].dtype))


def get_label_analysis(data):
    # 训练集中标签分布
    print("===获取训练集标签分布===")
    plt.figure(figsize=(15, 10))
    data['Transported'].value_counts().head(2).plot(kind='bar')
    plt.title('Transported data distribution')
    plt.show()  # 平均分布

    # 训练集中的标签与与哪一个特征最相关
    print("===获取训练集标签与特征相关系数===")
    print(data.corr()['Transported']) # 和 RoomService最相关（负相关）


def get_missing_data_analysis(data):
    feature_list = []
    num_list = []
    rate_list = []
    for i in list(data.columns):
        num = data[i].isnull().sum()
        rate = num/len(data)
        feature_list.append(i)
        num_list.append(num)
        rate_list.append(rate)
    missing_data_analysis = pd.DataFrame({'训练集': feature_list, '缺失值个数': num_list, '缺失率': rate_list})
    print(missing_data_analysis)


if __name__ == '__main__':
    # 训练集
    train_data = get_data_train()
    # 训练集行列数
    print("===获取训练集行列数===")
    get_column_and_row_num(train_data)

    # 测试集
    test_data = get_data_test()
    print("===获取测试集行列数===")
    # 测试集行列数
    get_column_and_row_num(test_data)

    # 训练集每列数据类型
    print("===获取训练集数据类型===")
    get_data_type(train_data)

    # 训练集标签分布、相关性分析
    get_label_analysis(train_data)

    # 训练集缺失值分布
    print("===获取测试集缺失值分布===")
    get_missing_data_analysis(train_data)


