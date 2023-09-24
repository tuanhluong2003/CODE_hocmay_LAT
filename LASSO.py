import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import metrics

# MAE => Mean Absolute Error (MAE) đo độ lớn trung bình của các lỗi trong một tập hợp các dự đoán mà không cần xem xét hướng của chúng
# NSE => Nash-Sutcliff-Efficiency là một độ đo được sử dụng để đánh giá độ chính xác của một mô hình dự đoán
# RMSE => Root Mean Squared Error là một độ đo được sử dụng để đánh giá độ chính xác của một mô hình dự đoán
# NSE => nash-cliffe efficiency
def NSE(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse_value = 1 - (numerator / denominator)
    return nse_value

#đọc dữ liệu từ file
data = pd.read_csv("traintaxippff.csv")

# Tách dữ liệu trainning = 70% test = 30%
dttrain, dttest = train_test_split(data, test_size = 0.3, shuffle = False)
X_train = dttrain.iloc[:,:-1]
Y_train = dttrain.iloc[:, 7]
X_test = dttest.iloc[:, :-1]
Y_test = dttest.iloc[:, 7]


# Lasso
print("\nLasso")
las = Lasso()
las.fit(X_train,Y_train)
y_preLs = las.predict(X_test)
print("Chenh lech %.10f" % r2_score(Y_test,y_preLs))
print('MAE:', metrics.mean_absolute_error(Y_test, y_preLs))
print('NSE:', NSE(Y_test, y_preLs))
print('RMSE:', metrics.mean_squared_error(Y_test, y_preLs, squared=False))

y = np.array(Y_test)
print("Thuc te - du doan - chech lech")
for i in range(0,len(y)):
    print(y[i],"-",y_preLs[i],"=",abs(y[i]-y_preLs[i]))

