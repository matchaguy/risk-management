import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_risk_summary(data):
    summary = {
        "Tổng EL (Expected Loss)": data["EL"].sum(),
        "PD Trung bình": data["PD"].mean(),
        "LGD Trung bình": data["LGD"].mean(),
        "EAD Trung bình": data["EAD"].mean(),
    }
    return summary

def plot_default_by_category(data, category):
    sns.barplot(x=category, y="default", data=data)
    plt.title(f"Tỷ lệ vỡ nợ theo {category}")
    plt.xlabel(category)
    plt.ylabel("Tỷ lệ vỡ nợ")
    plt.show()

def plot_customer_segments(data):
    fig, ax = plt.subplots()
    scatter = ax.scatter(data["LIMIT_BAL"], data["Credit_Utilization"], c=data["Customer_Segment"], cmap="viridis")
    ax.set_xlabel("LIMIT_BAL")
    ax.set_ylabel("Credit Utilization")
    ax.set_title("Phân khúc khách hàng")
    fig.colorbar(scatter, label="Customer Segment")
    return fig

def plot_payment_forecast(data):
    fig, ax = plt.subplots()
    ax.plot(data["Predicted_Payment"], label="Dự báo thanh toán", color="blue")
    ax.plot(data["PAY_AMT6"], label="Thanh toán thực tế", color="orange")
    ax.legend()
    ax.set_title("Dự báo dòng tiền thanh toán")
    ax.set_xlabel("Khách hàng")
    ax.set_ylabel("Số tiền thanh toán (chuẩn hóa)")
    return fig

def generate_summary_report(data):
    print("Báo cáo tổng hợp:")
    print(f"Tổng số khách hàng: {len(data)}")
    print(f"Số lượng khách hàng vỡ nợ: {data['default'].sum()}")
    print(f"Tỷ lệ khách hàng vỡ nợ: {data['default'].mean() * 100:.2f}%")
    print(f"Phân khúc khách hàng:")
    print(data["Customer_Segment"].value_counts())
    print("-" * 30)
    
def plot_risk_heatmap(data):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = data[["PD", "LGD", "EAD", "EL"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Tương quan giữa các chỉ số rủi ro")
    return fig
