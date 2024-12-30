import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import classification_report

if __name__ == '__main__':
    file_dir = './Data_students_new.csv'
    # 数据一共10列，名称分别为：'used_time', 'height', 'weight', 'consumption', 'sleeping_time','character', 'gift_language', 'gift_sport', 'gift_music', 'gift_logic', 'gender'
    # X_valid_list = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    # X_valid_list = [1, 2]  # 使用特征 'height' 和 'weight'
    X_valid_list = list(range(10))  # 数据集中所有特征的索引
    Y_valid_index = 10
    with open(file_dir, encoding='UTF8') as f:
        reader = csv.reader(f)
        name = next(reader)
        X = list()
        Y = list()
        i = 0
        for line in reader:
            X.append(list())
            Y.append(float(line[Y_valid_index]))
            for item_index in X_valid_list:
                X[i].append(float(line[item_index]))
            i = i + 1
        X = np.array(X)
        Y = np.array(Y)

    # 按比例拆分数据，50%用作训练
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i in range(len(X)):
        if i % 2 == 0:
            test_data.append(X[i])
            test_label.append(Y[i])
        else:
            train_data.append(X[i])
            train_label.append(Y[i])

    # 定义逻辑回归模型
    clf = linear_model.LogisticRegression(penalty='l1', dual=False, tol=1e-6, C=1000, class_weight=None, random_state=None,
                                          solver='liblinear', max_iter=10000, verbose=0, warm_start=False, n_jobs=1)
    clf.fit(train_data, train_label)

    # 对测试集进行预测并输出报告
    predict_test = clf.predict(test_data)
    print("Test Set Results:")
    print(classification_report(test_label, predict_test))

    # 对训练集进行预测并输出报告
    predict_train = clf.predict(train_data)
    print("Training Set Results:")
    print(classification_report(train_label, predict_train))
