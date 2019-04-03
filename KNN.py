# nhập các thư viện
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib import pyplot as plt

# đọc file
f = open("Dataset.data")
data = np.array([line.split() for line in f.readlines()])
f.close()
# tách features và nhãn => đầu vào
X = data[:, :3]
Y = data[:, 3]

before  = X[:3]

# one hot encode các features và nhãn để công bằng trong việc so sánh các giá
# trị của các features
X_ohe = OneHotEncoder(handle_unknown="ignore")
X_ohe.fit(X)
X = X_ohe.transform(X)
after = X[:3]
Y_ohe = LabelBinarizer()
Y_ohe.fit(Y)
Y = Y_ohe.transform(Y)

# tách training set và testing set từ data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1, stratify = Y)

# model để phân lớp 1-NN => đầu ra
OneNNClassifier = KNeighborsClassifier(n_neighbors=2, weights = 'distance').fit(x_train, y_train.ravel())
y_pred_1NN = Y_ohe.inverse_transform(OneNNClassifier.predict(x_test))
print('Prediction result for first 10 test examples:')
print(y_pred_1NN[:15])

# xây dựng model KNN với số các neighbors khác nhau để tìm được số tốt nhất từ 3->30
numberOfNeighbors = [i for i in range(3,30)]
trainingScore5fcv = []
testingScore5fcv = []

for i in numberOfNeighbors:
    knnClassifier = KNeighborsClassifier(algorithm= 'brute',n_neighbors=i, weights = 'distance').fit(x_train, y_train.ravel())
    score = np.mean(cross_val_score(knnClassifier, x_train, y_train.ravel(), cv = 5))
    trainingScore5fcv.append(score)
    score = np.mean(cross_val_score(knnClassifier, x_test, y_test.ravel(), cv = 5))
    testingScore5fcv.append(score)

# vẽ biểu đồ, so sánh độ chính xác giữa những số neighbors và giữa trên training set vs test set
plt.plot(numberOfNeighbors, trainingScore5fcv, 'b', label = '5-fold CV score on training set')
plt.plot(numberOfNeighbors, testingScore5fcv, 'r', label = '5-fold CV score on testing set')
plt.legend()
plt.ylim(0,1)
plt.ylabel('score')
plt.xlabel('n-neighbor')
plt.savefig('knn.png')
plt.show()