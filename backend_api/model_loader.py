import joblib

def load_model_1():
    return joblib.load("Models/Model_1/first_model.pkl")

def load_model_2():
    return joblib.load("Models/Model_2/second_model.pkl")