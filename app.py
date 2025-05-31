import streamlit as st
import pandas as pd
import os
import joblib
from preprocessing import preprocess_data
from feature_engineering import add_features, segment_customers,calculate_risk_metrics
from modeling import forecast_payments, train_model,format_classification_report
from visualization import plot_customer_segments, plot_payment_forecast, generate_summary_report,generate_risk_summary,plot_risk_heatmap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== Streamlit App ====================
# Cấu hình giao diện
st.set_page_config(page_title="Quản lý rủi ro tín dụng", layout="wide")
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'UCI_Credit_Card.csv')
# Tiêu đề chính
st.title("Quản lý rủi ro tín dụng")
results = []

# Đường dẫn đến file đã xử lý
PROCESSED_DATA_PATH = os.path.join(BASE_DIR,"processed_credit_data.csv")
# Kiểm tra nếu file đã tồn tại
if os.path.exists(PROCESSED_DATA_PATH):
    processed_data = pd.read_csv(PROCESSED_DATA_PATH)
else:
    st.error("File dữ liệu đã xử lý không tồn tại. Hãy chạy script xử lý dữ liệu trước.")

# Đọc dữ liệu
def load_default_data():
    data = pd.read_csv(DEFAULT_DATA_PATH)
    processed_data = preprocess_data(data)
    processed_data = add_features(processed_data)
    processed_data = segment_customers(processed_data)
    processed_data = calculate_risk_metrics(processed_data)
    return processed_data
processed_data = load_default_data()
st.sidebar.header("Tải lên dữ liệu khách hàng mới")
uploaded_file = st.sidebar.file_uploader("Chọn file khách hàng mới (CSV)", type=["csv"])

# Thay thế đoạn này bằng logic mới dùng mô hình
# Tải mô hình đã lưu
model_file = "risk_management\modelfile\multioutput_model.pkl"
model = joblib.load(model_file)

# Giao diện tải dữ liệu khách hàng mới
if uploaded_file:
    st.sidebar.success("Dữ liệu khách hàng mới đã được tải lên!")
    # Đọc file khách hàng mới
    new_customer_data = pd.read_csv(uploaded_file)
    st.write("### Dữ liệu khách hàng mới:")
    st.dataframe(new_customer_data)

    # Tính các chỉ số rủi ro
    features = ["LIMIT_BAL", "Credit_Utilization", "Repayment_Trend"]
    new_customer_data = preprocess_data(new_customer_data)
    new_customer_data = add_features(new_customer_data)
    new_customer_data = segment_customers(new_customer_data)
    #new_customer_data = calculate_risk_metrics(new_customer_data)
    input_data = new_customer_data[features]

    # Dự đoán các chỉ số
    predictions = model.predict(input_data)
    new_customer_data[["PD", "LGD", "EAD", "EL"]] = predictions

    # Hiển thị kết quả
    st.write("### Kết quả dự đoán:")
    st.dataframe(new_customer_data[["LIMIT_BAL", "PD", "LGD", "EAD", "EL"]])
else:
    st.info("Hãy tải lên file CSV chứa dữ liệu khách hàng mới để đánh giá.")
if results:
    st.write("### Kết quả đánh giá rủi ro (mặc định)")
    st.dataframe(pd.DataFrame(results))
    # Tính toán các chỉ số rủi ro tín dụng
processed_data = calculate_risk_metrics(processed_data)
st.write("### Báo cáo rủi ro tín dụng")
risk_summary = generate_risk_summary(processed_data)
for key, value in risk_summary.items():
    st.write(f"{key}: {value:.2f}")
# Tab chức năng
tab1, tab2, tab3 = st.tabs(["📊 Tổng quan", "📈 Phân tích & Dự báo", "🤖 Huấn luyện mô hình"])
# Tab 1: Tổng quan
with tab1:
    st.header("Báo cáo tổng hợp")
    st.write("### Danh sách khách hàng vỡ nợ")
    generate_summary_report(processed_data)
    
    # Hiển thị bảng dữ liệu mẫu
    st.dataframe(processed_data)
# Tab 2: Phân tích & Dự báo
with tab2:
    st.header("Phân tích phân khúc khách hàng")
    st.pyplot(plot_customer_segments(processed_data))
    st.header("Dự báo dòng tiền thanh toán")
    processed_data, reg_model = forecast_payments(processed_data)
    st.pyplot(plot_payment_forecast(processed_data))
    from visualization import plot_risk_heatmap
    st.header("Tương quan giữa các chỉ số rủi ro")
    heatmap_fig = plot_risk_heatmap(processed_data)
    st.pyplot(heatmap_fig)
    # Tab 3: Huấn luyện mô hình
with tab3:
    st.header("Kết quả huấn luyện mô hình")
    # Huấn luyện mô hình và lấy báo cáo
    with st.spinner("Đang huấn luyện mô hình..."):
        model_with_smote, classification_report_str, auc_roc = train_model(processed_data)
    # Định dạng và hiển thị báo cáo phân loại
    st.subheader("Báo cáo phân loại:")
    report_df = format_classification_report(classification_report_str)
    st.dataframe(report_df)  # Hiển thị dưới dạng bảng
    # Hiển thị chỉ số AUC-ROC
    st.subheader("Chỉ số AUC-ROC:")
    st.metric(label="AUC-ROC", value=f"{auc_roc:.4f}")
    st.success("Đã huấn luyện và đánh giá mô hình với kỹ thuật SMOTE để xử lý mất cân bằng dữ liệu.")
    st.write("### Kết quả đánh giá rủi ro")
    st.dataframe(pd.DataFrame(results))



