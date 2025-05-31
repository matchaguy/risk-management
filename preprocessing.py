import pandas as pd

def preprocess_data(data):
    data.rename(columns={"default.payment.next.month": "default"}, inplace=True)
    ## Chuyển đổi các biến danh mục
    #data["SEX"] = data["SEX"].astype("category")
    #data["EDUCATION"] = data["EDUCATION"].astype("category")
    #data["MARRIAGE"] = data["MARRIAGE"].astype("category")
    #
    ## Chuẩn hóa các biến liên tục
    #continuous_features = [
    #    "LIMIT_BAL", "AGE", 
    #    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    #    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    #]
    #data[continuous_features] = (data[continuous_features] - data[continuous_features].mean()) / data[continuous_features].std()
    #
    # Loại bỏ cột ID
    data.drop(columns=["ID"], inplace=True)
    return data