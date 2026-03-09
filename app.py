import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Page setup
st.set_page_config(page_title="Student Marks Predictor", page_icon="📊")

# Login credentials
USERNAME = "admin"
PASSWORD = "1234"

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login page
if not st.session_state.logged_in:

    st.title("🔐 Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

# Main app
if st.session_state.logged_in:

    st.title("📊 Student Marks Prediction AI App")

    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("student_data.csv")

    st.subheader("Dataset Preview")
    st.write(data)

    X = data[["StudyHours","Attendance"]]
    y = data["Marks"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = LinearRegression()
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    accuracy = r2_score(y_test,pred)

    st.subheader("Model Accuracy")
    st.success(f"Accuracy: {accuracy:.2f}")

    st.sidebar.header("Student Details")

    hours = st.sidebar.slider("Study Hours",0,12,4)
    attendance = st.sidebar.slider("Attendance",50,100,80)

    if st.sidebar.button("Predict"):

        prediction = model.predict([[hours,attendance]])

        st.subheader("Prediction Result")
        st.success(f"Predicted Marks: {prediction[0]:.2f}")

        st.progress(int(prediction[0]))

    st.subheader("Study Hours vs Marks")

    fig,ax = plt.subplots()
    ax.scatter(data["StudyHours"],data["Marks"])
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Marks")

    st.pyplot(fig)

    st.subheader("Attendance vs Marks")

    fig2,ax2 = plt.subplots()
    ax2.scatter(data["Attendance"],data["Marks"])
    ax2.set_xlabel("Attendance")
    ax2.set_ylabel("Marks")

    st.pyplot(fig2)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()