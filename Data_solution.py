import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from ydata_profiling import ProfileReport
import tempfile

# -----------------------------
# Page Configuration & CSS
# -----------------------------
st.set_page_config(
    page_title="Data Solution",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
    :root { --primary: #38bdf8; --background: #0f172a; --card: rgba(30,41,59,0.6); --text: #e2e8f0; --secondary-text: #cbd5e1; }
    * { font-family: 'Poppins', sans-serif; }
    html, body, [class*="css"] { color: var(--text); background-color: var(--background); }
    .stApp { background: linear-gradient(135deg, rgba(15,23,42,1) 0%, rgba(23,37,64,1) 100%); }
    .stSidebar { background-color: rgba(30,41,59,0.8) !important; backdrop-filter: blur(10px); border-right: 1px solid rgba(255,255,255,0.1);}
    .stButton>button { border: 2px solid var(--primary); color: var(--primary); background: transparent; border-radius: 8px; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: rgba(56,189,248,0.1); }
    .stSelectbox, .stMultiselect, .stTextInput, .stNumberInput, .stTextArea, .stDateInput, .stTimeInput { background-color: rgba(30,41,59,0.6) !important; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); }
    .stDataFrame { background-color: rgba(30,41,59,0.6) !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <h1 style="margin: 0; color: #38bdf8; font-size: 2.5rem;">DataSolution</h1>
    <span style="margin-left: auto; font-size: 1rem; color: #cbd5e1;">Turn Raw Data into Powerful Insights</span>
</div>
""", unsafe_allow_html=True)

st.title("Data Solution")
st.markdown("This app performs automated data analysis, visualization, and predictive modeling. Upload your dataset and follow the steps!")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Step:", 
                          ["Upload Data", "Data Cleaning", "EDA", 
                           "Visualization", "Prediction", "Insights"])

# -----------------------------
# Initialize Session State
# -----------------------------
for key in ['df', 'cleaned_df', 'target', 'model_type', 'model', 'report']:
    if key not in st.session_state:
        st.session_state[key] = None

# -----------------------------
# Step 1: Upload Data
# -----------------------------
if options == "Upload Data":
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            # Convert datetime columns to string to avoid dtype errors in ML
            for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns:
                df[col] = df[col].astype(str)
            st.session_state.df = df
            st.success("Data uploaded successfully!")
            st.subheader("Data Preview")
            st.write(df.head())
            st.subheader("Basic Information")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.subheader("Data Types")
            st.write(pd.DataFrame(df.dtypes, columns=['Data Type']))
            st.subheader("Missing Values")
            missing_df = pd.DataFrame(df.isna().sum(), columns=['Missing Values'])
            missing_df['Percentage'] = (missing_df['Missing Values'] / len(df)) * 100
            st.write(missing_df)
            st.subheader("Duplicate Rows")
            st.write(f"Number of duplicate rows: {df.duplicated().sum()}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# -----------------------------
# Step 2: Data Cleaning
# -----------------------------
elif options == "Data Cleaning" and st.session_state.df is not None:
    st.header("Data Cleaning")
    df = st.session_state.df.copy()
    
    # Cleaning Options
    cleaning_options = st.multiselect(
        "Select cleaning operations to perform:",
        [
            "Remove duplicate rows",
            "Fill missing values (numeric)",
            "Fill missing values (categorical)",
            "Remove rows with missing values",
            "Remove columns with high missing values (>30%)",
            "Convert text to numeric where possible",
            "Remove outliers (numeric columns)",
            "Standardize column names"
        ]
    )
    
    if st.button("Clean Data"):
        cleaned_df = df.copy()
        if "Remove duplicate rows" in cleaning_options:
            cleaned_df = cleaned_df.drop_duplicates()
        if "Fill missing values (numeric)" in cleaning_options:
            num_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
            cleaned_df[num_cols] = SimpleImputer(strategy='mean').fit_transform(cleaned_df[num_cols])
        if "Fill missing values (categorical)" in cleaning_options:
            for col in cleaned_df.select_dtypes(include=['object']).columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
        if "Remove rows with missing values" in cleaning_options:
            cleaned_df = cleaned_df.dropna()
        if "Remove columns with high missing values (>30%)" in cleaning_options:
            threshold = len(cleaned_df) * 0.3
            cleaned_df = cleaned_df.dropna(axis=1, thresh=threshold)
        if "Convert text to numeric where possible" in cleaning_options:
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    try:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                    except: pass
        if "Remove outliers (numeric columns)" in cleaning_options:
            num_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
            for col in num_cols:
                mean,std = cleaned_df[col].mean(), cleaned_df[col].std()
                cleaned_df = cleaned_df[(cleaned_df[col] <= mean+3*std) & (cleaned_df[col] >= mean-3*std)]
        if "Standardize column names" in cleaning_options:
            cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ','_')
        
        st.session_state.cleaned_df = cleaned_df
        st.success("Data cleaning completed!")
        st.write(cleaned_df.head())

# -----------------------------
# Step 3: EDA
# -----------------------------
elif options == "EDA" and st.session_state.cleaned_df is not None:
    st.header("Exploratory Data Analysis")
    df = st.session_state.cleaned_df.copy()
    
    # Convert all object/categorical columns to string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    
    if st.button("Generate EDA Report"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            try:
                profile = ProfileReport(df, title="Profiling Report", minimal=True)
                profile.to_file(tmpfile.name)
                st.session_state.report = tmpfile.name
                with open(tmpfile.name,'r') as f:
                    st.components.v1.html(f.read(), height=1000, scrolling=True)
                with open(tmpfile.name,'rb') as f:
                    st.download_button("Download EDA Report", f, "eda_report.html", "text/html")
            except Exception as e:
                st.error(f"Error generating EDA report: {e}")

# -----------------------------
# Step 4: Visualization
# -----------------------------
elif options == "Visualization" and st.session_state.cleaned_df is not None:
    st.header("Data Visualization")
    df = st.session_state.cleaned_df.copy()
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    viz_type = st.selectbox("Select Visualization Type", ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Pie", "Heatmap"])
    
    filtered_df = df.copy() # filters could be added here
    if viz_type == "Scatter" and len(numeric_cols)>=2:
        x_axis = st.selectbox("X-axis", numeric_cols)
        y_axis = st.selectbox("Y-axis", numeric_cols)
        hue = st.selectbox("Hue", [None]+list(cat_cols))
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=hue)
        st.plotly_chart(fig)
    elif viz_type == "Histogram" and len(numeric_cols)>=1:
        col = st.selectbox("Column", numeric_cols)
        fig = px.histogram(filtered_df, x=col)
        st.plotly_chart(fig)
    elif viz_type == "Bar" and len(cat_cols)>=1 and len(numeric_cols)>=1:
        x_axis = st.selectbox("X-axis", cat_cols)
        y_axis = st.selectbox("Y-axis", numeric_cols)
        fig = px.bar(filtered_df, x=x_axis, y=y_axis)
        st.plotly_chart(fig)
    elif viz_type == "Box" and len(numeric_cols)>=1:
        y_axis = st.selectbox("Y-axis", numeric_cols)
        x_axis = st.selectbox("X-axis (optional)", [None]+list(cat_cols))
        fig = px.box(filtered_df, x=x_axis, y=y_axis)
        st.plotly_chart(fig)

# -----------------------------
# Step 5: Prediction
# -----------------------------
elif options == "Prediction" and st.session_state.cleaned_df is not None:
    st.header("Predictive Modeling")
    df = st.session_state.cleaned_df.copy()
    target = st.selectbox("Select Target Variable", df.columns)
    st.session_state.target = target
    
    # Determine problem type
    if df[target].dtype in ['int64','float64'] and df[target].nunique() >= 10:
        problem_type = "regression"
    else:
        problem_type = "classification"
    st.session_state.model_type = problem_type
    st.write(f"Problem type detected: **{problem_type}**")
    
    if st.button("Train Model"):
        X = df.drop(columns=[target])
        y = df[target]
        # Label encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        # Fill missing values
        X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
        # Split & scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Train
        if problem_type=="regression":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.success(f"R²: {model.score(X_test,y_test):.2f}, MSE: {mean_squared_error(y_test,y_pred):.2f}")
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.success(f"Accuracy: {accuracy_score(y_test,y_pred):.2f}")
        st.session_state.model = model

# -----------------------------
# Step 6: Insights & Predictive Dashboard
# -----------------------------
elif options == "Insights" and st.session_state.cleaned_df is not None:
    st.header("Data Storytelling & Predictive Insights")
    df = st.session_state.cleaned_df.copy()
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Dataset overview
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    for col in numeric_cols:
        st.write(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    for col in cat_cols:
        st.write(f"{col}: Top values:\n{df[col].value_counts().head(3)}")
    
    # Predictive Dashboard
    if st.session_state.model is not None and st.session_state.target is not None:
        st.subheader("Predictive Insights Dashboard")
        target = st.session_state.target
        model = st.session_state.model
        problem_type = st.session_state.model_type
        X = df.drop(columns=[target])
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df['Prediction'] = model.predict(X_scaled)
        
        # Top predictions
        top_n = st.slider("Top predictions to show", 5, 20, 10)
        if problem_type=="regression":
            st.dataframe(df.nlargest(top_n,'Prediction'))
        else:
            st.dataframe(df['Prediction'].value_counts().head(top_n))
        
        # Feature importance
        feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        fig = go.Figure(go.Bar(x=feat_imp['Importance'], y=feat_imp['Feature'], orientation='h', marker_color='#38bdf8'))
        fig.update_layout(yaxis=dict(autorange='reversed'), height=500)
        st.plotly_chart(fig)
        
        # Top 3 features
        st.subheader("Top 3 Features")
        top_feats = feat_imp.head(3)
        c1,c2,c3 = st.columns(3)
        c1.metric(top_feats.iloc[0]['Feature'], f"{top_feats.iloc[0]['Importance']:.2f}")
        c2.metric(top_feats.iloc[1]['Feature'], f"{top_feats.iloc[1]['Importance']:.2f}")
        c3.metric(top_feats.iloc[2]['Feature'], f"{top_feats.iloc[2]['Importance']:.2f}")
        
        # Download predictions
        st.download_button("Download Dataset with Predictions", df.to_csv(index=False).encode('utf-8'), "predictions.csv","text/csv")

# -----------------------------
# Footer
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Solution**\nCreated with Streamlit, Pandas, Scikit-learn")
if st.session_state.cleaned_df is not None:
    st.sidebar.download_button("Download Cleaned Data CSV", st.session_state.cleaned_df.to_csv(index=False).encode('utf-8'), "cleaned_data.csv","text/csv")
