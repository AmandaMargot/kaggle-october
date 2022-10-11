import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


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
    print(data.corr()['Transported'])  # 和 RoomService最相关（负相关）


def get_missing_data_analysis(data):
    feature_list = []
    num_list = []
    rate_list = []
    for i in list(data.columns):
        num = data[i].isnull().sum()
        rate = num / len(data)
        feature_list.append(i)
        num_list.append(num)
        rate_list.append(rate)
    missing_data_analysis = pd.DataFrame({'训练集': feature_list, '缺失值个数': num_list, '缺失率': rate_list})
    print(missing_data_analysis)


def data_visualization(data):
    # HomePlanet 与 Transported 的分布关系
    cat_plot_bar(data, "HomePlanet", "Transported",
                 "Distribution between HomePlanet and Transported")  # Europa > Mars > Earth
    # CryoSleep 与 Transported 的分布关系
    cat_plot_bar(data, "CryoSleep", "Transported", "Distribution between CryoSleep and Transported")  # True > False
    # # Cabin 与 Transported 的分布关系
    # Todo
    # Destination 与 Transported 的分布关系
    cat_plot_bar(data, "Destination", "Transported",
                 "Distribution between Destination and Transported")  # 55 > PSO > TRAPPIST
    # Age 与 Transported 的分布关系
    # Todo
    # VIP 与 Transported 的分布关系
    cat_plot_bar(data, "VIP", "Transported",
                 "Distribution between VIP and Transported") # False > True

    # RoomService、FoodCourt、ShoppingMall、Spa、VRDeck 与 Transported 的分布关系
    # Todo
    # Name 与 Transported 的分布关系
    # Todo


def cat_plot_bar(data, x, y, title):
    plot = sns.catplot(kind="bar", x=x, y=y, data=data, height=8, aspect=.8)
    plot.fig.suptitle(title)
    plt.show()


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

    # 可视化
    data_visualization(train_data)
