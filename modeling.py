
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import calculate_risk_metrics
from imblearn.over_sampling import SMOTE
# Hàm định dạng báo cáo phân loại thành DataFrame
def format_classification_report(report_str):
    report_data = []
    lines = report_str.split("\n")
    for line in lines[2:len(lines)-3]:  # Bỏ dòng đầu và cuối
        row_data = line.split()
        if len(row_data) != 5:  # Kiểm tra xem dòng có đủ 5 giá trị không
            continue
        try:
            class_label = row_data[0]
            precision = float(row_data[1])
            recall = float(row_data[2])
            f1_score = float(row_data[3])
            support = int(row_data[4])
            report_data.append({
                "Class": class_label,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score,
                "Support": support
            })
        except ValueError as e:  # Xử lý lỗi khi chuyển đổi không thành công
            print(f"Không thể xử lý dòng: {line}. Lỗi: {e}")
            continue
    return pd.DataFrame.from_records(report_data)


# Function to evaluate new customers
def evaluate_new_customers(model, new_data, processed_data):
    # Tính toán PD, LGD, EAD, EL
    new_data = calculate_risk_metrics(new_data)
    features = ["PD", "LGD", "EAD", "EL"]

    # Dự đoán rủi ro cho khách hàng mới
    predictions = model.predict(new_data[features])
    probabilities = model.predict_proba(new_data[features])[:, 1]

    new_data["Predicted_Default"] = predictions
    new_data["Default_Probability"] = probabilities

    return new_data

def train_model(data):
    # Tách dữ liệu
    X = data.drop(columns=["default"])
    y = data["default"]
    
    # Áp dụng SMOTE để xử lý mất cân bằng dữ liệu
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    
    # Huấn luyện mô hình
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Dự đoán và tính toán các chỉ số
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    classification_report_str = classification_report(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    
    return model, classification_report_str, auc_roc

def forecast_payments(data):
    X = data[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5"]]
    y = data["PAY_AMT6"]
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    data["Predicted_Payment"] = reg_model.predict(X)
    return data, reg_model