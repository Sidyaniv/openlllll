import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title="ПСБ Bank", layout="wide")
st.title("ПСБ Bank - Customer Product Recommender System")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'segments' not in st.session_state:
    st.session_state.segments = None

# Function to preprocess data
def preprocess_data(df, encoders=None, fit_encoders=True):
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in df.select_dtypes(include='object').columns:
        if col == 'Customer_ID':
            continue
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            if col in encoders:
                le = encoders[col]
                df[col] = le.transform(df[col].astype(str))
    return df, encoders

# Function to train model
def train_model(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Function to create customer segments
def segment_customers(df_encoded, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    segments = kmeans.fit_predict(df_encoded)
    return segments

# Page Navigation
page = st.sidebar.radio("📄 Navigate", [
    "1️⃣ Upload Data",
    "2️⃣ Predict Products",
    "3️⃣ Model Evaluation",
    "4️⃣ Customer Segmentation"
])

# Page 1 - Upload Data
if page == "1️⃣ Upload Data":
    st.subheader("📤 Upload Train Data for training the model")

    train_file = st.file_uploader("📌 Upload Training Data (.xlsx)", type="xlsx", key="train")

    if train_file:
        train_df = pd.read_excel(train_file)

        if 'Preferred_Product' not in train_df.columns:
            st.error("❌ 'Preferred_Product' column is missing in the training data.")
        else:
            st.session_state.train_df = train_df
            X_train = train_df.drop(columns=['Customer_ID', 'Predicted_Preferred_Product', 'Preferred_Product'], errors='ignore')
            y_train = train_df['Preferred_Product']

            X_train_encoded, encoders = preprocess_data(X_train)
            model = train_model(X_train_encoded, y_train)

            st.session_state.model = model
            st.session_state.label_encoders = encoders

            segments = segment_customers(X_train_encoded, n_clusters=4)
            st.session_state.segments = segments

            st.success("✅ Training file uploaded successfully!")
            st.success("✅ Model trained successfully!")

            st.write("📄 Training Data Preview:")
            st.dataframe(train_df.head())

    if st.session_state.model is not None:
        st.markdown("---")
        test_file = st.file_uploader("📌 Now upload your Test Data (.xlsx)", type="xlsx", key="test")

        if test_file:
            test_df = pd.read_excel(test_file)
            st.session_state.test_df = test_df
            st.success("✅ Test data uploaded successfully!")

            st.write("📄 Test Data Preview:")
            st.dataframe(test_df.head())

# Page 2 - Predict Products
elif page == "2️⃣ Predict Products":
    st.subheader("🔮 Predict Preferred Product for a Customer")

    if st.session_state.model is not None and st.session_state.test_df is not None:
        test_df = st.session_state.test_df
        model = st.session_state.model
        encoders = st.session_state.label_encoders

        test_features = test_df.drop(columns=['Customer_ID', 'Predicted_Preferred_Product'], errors='ignore')
        test_encoded, _ = preprocess_data(test_features, encoders=encoders, fit_encoders=False)
        predictions = model.predict(test_encoded)

        test_df['Predicted_Preferred_Product'] = predictions
        st.session_state.predictions = test_df

        customer_ids = test_df['Customer_ID'].unique()
        selected_id = st.selectbox("🔍 Select Customer ID", customer_ids)

        if st.button("🎯 Predict"):
            customer_data = test_df[test_df['Customer_ID'] == selected_id].drop(columns=['Predicted_Preferred_Product'])
            st.write("### 📌 Customer Info:")
            for col in customer_data.columns:
                st.markdown(f"**{col}**: {customer_data[col].values[0]}")
            st.success(f"🎉 Predicted Preferred Product: **{test_df.loc[test_df['Customer_ID'] == selected_id, 'Predicted_Preferred_Product'].values[0]}**")

        st.markdown("---")
        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction CSV", data=csv, file_name="predicted_test_data.csv", mime="text/csv")

    else:
        st.warning("⚠️ Please upload and train the model in Page 1 first.")

# Page 3 - Model Evaluation
elif page == "3️⃣ Model Evaluation":

    if st.session_state.train_df is not None and st.session_state.model is not None:
        train_df = st.session_state.train_df
        X_train = train_df.drop(columns=['Customer_ID', 'Predicted_Preferred_Product', 'Preferred_Product'], errors='ignore')
        y_train = train_df['Preferred_Product']
        X_train_encoded, _ = preprocess_data(X_train, encoders=st.session_state.label_encoders, fit_encoders=False)

        st.write("### 🔥 Feature Importances:")
        importances = st.session_state.model.feature_importances_
        feature_names = X_train.columns
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        fig = px.bar(imp_df, x='Feature', y='Importance', color='Importance', color_continuous_scale='bluered')
        st.plotly_chart(fig, use_container_width=True)

        st.write("### 📊 Train Data Variable - Products Distribution:")
        target_dist = train_df['Preferred_Product'].value_counts().reset_index()
        target_dist.columns = ['Preferred_Product', 'Count']
        dist_fig = px.bar(target_dist, x='Preferred_Product', y='Count', color='Preferred_Product', title='Target Variable Distribution')
        st.plotly_chart(dist_fig, use_container_width=True)

        st.write("### 🧪 Test Data - Predicted Products Distribution:")
        if 'Predicted_Preferred_Product' in st.session_state.test_df.columns:
            pred_dist = st.session_state.test_df['Predicted_Preferred_Product'].value_counts().reset_index()
            pred_dist.columns = ['Predicted_Preferred_Product', 'Count']
            pred_fig = px.bar(pred_dist,x='Predicted_Preferred_Product',y='Count',color='Predicted_Preferred_Product',
                                title='Predicted Preferred Product Distribution (Test Data)')
            st.plotly_chart(pred_fig, use_container_width=True)
        else:
            st.info("ℹ️ Predictions not yet available for test data. Please go to Page 2 and run predictions.")

    else:
        st.warning("⚠️ Please upload and train the model in Page 1 first.")

# Page 4 - Customer Segmentation
elif page == "4️⃣ Customer Segmentation":
    st.subheader("🧩 Customer Segmentation Based on Clusters")

    if st.session_state.train_df is not None and st.session_state.segments is not None:
        train_df = st.session_state.train_df.copy()
        train_df['Segment'] = st.session_state.segments

        segment_names = {
            0: "🟡 Budget-Conscious",
            1: "🔵 Premium Loyalists",
            2: "🟢 Balanced Seekers",
            3: "🟠 New Explorers"
        }

        train_df['Segment_Label'] = train_df['Segment'].map(segment_names)

        selected_id = st.selectbox("🔍 Select Customer ID", train_df['Customer_ID'].unique())
        selected_row = train_df[train_df['Customer_ID'] == selected_id]

        if not selected_row.empty:
            segment = selected_row['Segment_Label'].values[0]
            st.info(f"🧠 Customer **{selected_id}** belongs to segment: **{segment}**")
    st.write("### 🧩 Customer Segmentation Distribution (Pie Chart)")
    if 'Segment' in train_df.columns:
        seg_dist = train_df['Segment'].value_counts().reset_index()
        seg_dist.columns = ['Segment', 'Count']
        seg_dist['Segment'] = seg_dist['Segment'].map(segment_names)
        pie_fig = px.pie(
            seg_dist,
            names='Segment',
            values='Count',
            title='Distribution of Customer Segments',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.warning("⚠️ Segment information not available.")

else:
        st.warning("⚠️ Please upload and train the model in Page 1 first.")