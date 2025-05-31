import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from feature_engineering import add_features
from preprocessing import preprocess_data
import os
# Đọc dữ liệu từ file CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'processed_credit_data.csv')

data = pd.read_csv(data_path)
#data = preprocess_data(data)
data = add_features(data)

# Chuẩn bị dữ liệu
features = ["LIMIT_BAL", "Credit_Utilization", "Repayment_Trend"]
targets = ["PD", "LGD", "EAD", "EL"]

X = data[features]
y = data[targets]

# Chia dữ liệu huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình RandomForest với đa đầu ra
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Lưu mô hình
model_file = "multioutput_model.pkl"
joblib.dump(model, model_file)
print(f"Mô hình đã được lưu tại {model_file}")
