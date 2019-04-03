# nhập thư viện cần dùng
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
# đọc file
f = open("Dataset.data")
data = np.array([line.split() for line in f.readlines()])
f.close()
# tách features và nhãn => đầu vào
X = data[:, :3]
Y = data[:, 3]

# one hot encode các features và nhãn để công bằng trong việc so sánh các giá
# trị của các features
X_ohe = OneHotEncoder(handle_unknown="ignore")
X_ohe.fit(X)
X = X_ohe.transform(X)

Y_ohe = LabelBinarizer()
Y_ohe.fit(Y)
Y = Y_ohe.transform(Y)

# tách training set và testing set từ data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1, stratify = Y)

# xây dựng model dự đoán cây quyết định =>output
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred = Y_ohe.inverse_transform(dt.predict(x_test))
print('Prediction result for first 10 test examples:')
print(y_pred[:10])
# điều chỉnh model theo độ cao cây tối đa và so sánh độ chính xác bằng 5-fold cv
depths = [i for i in range(1, 5)]
trainingScore5fcv = []
testingScore5fcv = []

for i in depths:
    dtc = DecisionTreeClassifier(max_depth=i).fit(x_train, y_train.ravel())
    score = np.mean(cross_val_score(dtc, x_train, y_train.ravel(), cv = 5))
    trainingScore5fcv.append(score)
    score = np.mean(cross_val_score(dtc, x_test, y_test.ravel(), cv = 5))
    testingScore5fcv.append(score)

# vẽ biểu đồ, so sánh độ chính xác giữa những số neighbors và giữa trên training set vs test set
plt.plot(depths, trainingScore5fcv, 'b', label = '5-fold CV score on training set')
plt.plot(depths, testingScore5fcv, 'r', label = '5-fold CV score on testing set')
plt.legend()
plt.ylabel('score')
plt.xlabel('tree depth')
plt.savefig('decisiontree.png')
plt.show()
