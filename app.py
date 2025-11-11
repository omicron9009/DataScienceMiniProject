import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures

# Page Configuration
st.set_page_config(
    page_title="MGNREGA Comprehensive Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load Data Function
@st.cache_data
def load_data():
    """Load the MGNREGA dataset"""
    try:
        df = pd.read_csv('data/combined.csv')
        return df
    except:
        st.error("⚠️ Please ensure 'data/combined.csv' exists!")
        return None

# Preprocessing Functions
def preprocess_for_regression(df):
    """Prepare data for regression analysis"""
    df_clean = df.select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).median())
    return df_clean

def preprocess_for_clustering(df):
    """Prepare data for clustering"""
    exclude_cols = ['fin_year', 'month', 'state_code', 'state_name', 
                   'district_code', 'district_name', 'Remarks']
    numeric_data = df.drop(columns=exclude_cols, errors='ignore')
    numeric_data = numeric_data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.fillna(numeric_data.median())
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    return numeric_data, scaled_data

def preprocess_for_decision_tree(df):
    """Prepare data for decision tree analysis"""
    df_processed = df.copy()
    
    # Encode categorical columns
    le_district = LabelEncoder()
    le_month = LabelEncoder()
    
    if 'district_name' in df.columns:
        df_processed['district_encoded'] = le_district.fit_transform(df['district_name'])
    if 'month' in df.columns:
        df_processed['month_encoded'] = le_month.fit_transform(df['month'])
    
    # Create target variables
    if 'Total_Individuals_Worked' in df.columns and 'Total_Exp' in df.columns:
        df_processed['Productivity'] = df['Total_Individuals_Worked'] / (df['Total_Exp'] + 1)
        median_prod = df_processed['Productivity'].median()
        df_processed['High_Productivity'] = (df_processed['Productivity'] > median_prod).astype(int)
    
    if 'Total_Adm_Expenditure' in df.columns and 'Total_Exp' in df.columns:
        df_processed['Admin_Cost_Ratio'] = df['Total_Adm_Expenditure'] / (df['Total_Exp'] + 1)
        df_processed['High_Admin_Spending'] = (df_processed['Admin_Cost_Ratio'] > 0.15).astype(int)
    
    return df_processed, le_district, le_month

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🌟 MGNREGA Analysis Dashboard 🌟</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar Navigation
    st.sidebar.title("📊 Navigation Panel")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Analysis Type:",
        ["🏠 Home & Overview",
         "📈 Exploratory Data Analysis (EDA)",
         "📊 Regression Analysis",
         "🎯 Clustering Analysis", 
         "🌳 Decision Tree Analysis",
         "🔮 Advanced Predictions"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Use the menu to navigate between different analyses")
    
    # Display dataset info in sidebar
    with st.sidebar.expander("📋 Dataset Info"):
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Total Columns:** {len(df.columns)}")
        st.write(f"**Date Range:** {df['fin_year'].min()} - {df['fin_year'].max()}")
    
    # Page Routing
    if page == "🏠 Home & Overview":
        show_home(df)
    elif page == "📈 Exploratory Data Analysis (EDA)":
        show_eda(df)
    elif page == "📊 Regression Analysis":
        show_regression(df)
    elif page == "🎯 Clustering Analysis":
        show_clustering(df)
    elif page == "🌳 Decision Tree Analysis":
        show_decision_tree(df)
    elif page == "🔮 Advanced Predictions":
        show_predictions(df)

# Home Page
def show_home(df):
    st.markdown('<h2 class="sub-header">Welcome to MGNREGA Analysis Dashboard</h2>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📁 Total Records", f"{len(df):,}")
    with col2:
        st.metric("📊 Features", len(df.columns))
    with col3:
        st.metric("🏛️ Districts", df['district_name'].nunique() if 'district_name' in df.columns else "N/A")
    with col4:
        st.metric("💰 Total Expenditure", f"₹{df['Total_Exp'].sum()/1e7:.2f}Cr" if 'Total_Exp' in df.columns else "N/A")
    
    st.markdown("---")
    
    # Dataset Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Quick Stats
    st.subheader("📊 Quick Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numerical Summary:**")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.markdown("**Missing Values:**")
        missing = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing[missing['Missing'] > 0], use_container_width=True)

# EDA Page
def show_eda(df):
    st.markdown('<h2 class="sub-header">📈 Exploratory Data Analysis</h2>', 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📉 Top Districts", "📅 Temporal Analysis", "🔗 Correlations"])
    
    with tab1:
        st.subheader("Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Average_Wage_rate_per_day_per_person' in df.columns:
                fig = px.histogram(df, x='Average_Wage_rate_per_day_per_person',
                                 nbins=30, title="Average Daily Wage Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Key Insights:**
                - Shows wage equity across regions
                - Most districts maintain similar wage rates per policy
                """)
        
        with col2:
            if 'Average_days_of_employment_provided_per_Household' in df.columns:
                fig = px.histogram(df, x='Average_days_of_employment_provided_per_Household',
                                 nbins=25, title="Employment Days Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Key Insights:**
                - Distribution of employment generation
                - Shows program reach effectiveness
                """)
    
    with tab2:
        st.subheader("Top Performing Districts")
        
        if 'district_name' in df.columns and 'Total_Exp' in df.columns:
            top_districts = df.groupby('district_name')['Total_Exp'].sum().nlargest(10).reset_index()
            
            fig = px.bar(top_districts, x='district_name', y='Total_Exp',
                        title="Top 10 Districts by Total Expenditure",
                        color='Total_Exp', color_continuous_scale='Blues')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 10 Districts:**")
                st.dataframe(top_districts, use_container_width=True)
            
            with col2:
                st.success("""
                **Analysis:**
                - Districts with highest expenditure show maximum MGNREGA activity
                - Indicates higher employment generation
                - Represents better fund utilization
                """)
    
    with tab3:
        st.subheader("Temporal Analysis")
        
        if 'fin_year' in df.columns and 'month' in df.columns and 'Total_Exp' in df.columns:
            monthly_data = df.groupby(['fin_year', 'month'])['Total_Exp'].mean().reset_index()
            
            fig = px.line(monthly_data, x='month', y='Total_Exp', color='fin_year',
                         title="Monthly Expenditure Trends",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Temporal Patterns:**
            - Seasonal variations in expenditure
            - Year-over-year comparisons
            - Peak activity months identification
            """)
    
    with tab4:
        st.subheader("Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:15]  # Top 15 for readability
        
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("""
            **Correlation Insights:**
            - Strong positive correlations indicate related metrics
            - Helps identify redundant features
            - Guides feature selection for modeling
            """)

# Regression Analysis Page
def show_regression(df):
    st.markdown('<h2 class="sub-header">📊 Regression Analysis</h2>', 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📉 Simple Linear Regression", "📈 Multiple Regression", "🔄 Polynomial Regression"])
    
    with tab1:
        st.subheader("Simple Linear Regression: Wages vs Total Expenditure")
        
        if 'Total_Exp' in df.columns and 'Wages' in df.columns:
            X = df[['Total_Exp']].values
            y = df['Wages'].values
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2:.4f}")
            with col2:
                st.metric("Intercept", f"{model.intercept_:.2f}")
            with col3:
                st.metric("Coefficient", f"{model.coef_[0]:.6f}")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X, y, alpha=0.5, label='Actual Data')
            ax.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
            ax.set_xlabel('Total Expenditure', fontsize=12)
            ax.set_ylabel('Wages', fontsize=12)
            ax.set_title('Simple Linear Regression', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.success(f"""
            **Model Performance:**
            - R² Score: {r2:.4f} (Explains {r2*100:.2f}% of variance)
            - RMSE: {rmse:.2f}
            - Formula: Wages = {model.intercept_:.2f} + {model.coef_[0]:.6f} × Total_Exp
            """)
    
    with tab2:
        st.subheader("Multiple Linear Regression")
        
        feature_cols = ['Total_Exp', 'Approved_Labour_Budget', 'Women_Persondays']
        
        if all(col in df.columns for col in feature_cols) and 'Wages' in df.columns:
            X = df[feature_cols].fillna(df[feature_cols].median())
            y = df['Wages']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training R²", f"{r2_train:.4f}")
            with col2:
                st.metric("Testing R²", f"{r2_test:.4f}")
            with col3:
                st.metric("Intercept", f"{model.intercept_:.2f}")
            
            # Coefficient table
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Feature Coefficients:**")
                st.dataframe(coef_df, use_container_width=True)
            
            with col2:
                fig = px.bar(coef_df, x='Feature', y='Coefficient',
                           title='Feature Importance (Coefficients)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Actual vs Predicted
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred_test, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Wages', fontsize=12)
            ax.set_ylabel('Predicted Wages', fontsize=12)
            ax.set_title('Actual vs Predicted Values', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.success(f"""
            **Model Performance:**
            - Training R²: {r2_train:.4f}
            - Testing R²: {r2_test:.4f}
            - Model explains {r2_test*100:.2f}% of wage variance
            """)
    
    with tab3:
        st.subheader("Polynomial Regression")
        
        degree = st.slider("Select Polynomial Degree", 2, 5, 2)
        
        if 'Total_Exp' in df.columns and 'Wages' in df.columns:
            X = df[['Total_Exp']].values
            y = df['Wages'].values
            
            # Polynomial features
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            
            # Train model
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            
            r2 = r2_score(y, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{r2:.4f}")
            with col2:
                st.metric("Polynomial Degree", degree)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort for smooth curve
            sort_idx = X.flatten().argsort()
            X_sorted = X[sort_idx]
            y_pred_sorted = y_pred[sort_idx]
            
            ax.scatter(X, y, alpha=0.3, label='Actual Data')
            ax.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, 
                   label=f'Polynomial Fit (degree={degree})')
            ax.set_xlabel('Total Expenditure', fontsize=12)
            ax.set_ylabel('Wages', fontsize=12)
            ax.set_title(f'Polynomial Regression (Degree {degree})', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.success(f"""
            **Polynomial Model Performance:**
            - R² Score: {r2:.4f}
            - Captures non-linear relationships
            - Degree {degree} polynomial fit
            """)

# Clustering Analysis Page
def show_clustering(df):
    st.markdown('<h2 class="sub-header">🎯 Clustering Analysis</h2>', 
                unsafe_allow_html=True)
    
    numeric_data, scaled_data = preprocess_for_clustering(df)
    
    # Sidebar controls
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 3)
    
    # Perform K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    df['Cluster'] = clusters
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["🗺️ Cluster Visualization", "📊 Cluster Statistics", "📋 Detailed Analysis"])
    
    with tab1:
        st.subheader(f"District Clusters (K={n_clusters})")
        
        # Create visualization dataframe
        viz_df = pd.DataFrame({
            'PCA1': pca_data[:, 0],
            'PCA2': pca_data[:, 1],
            'Cluster': clusters.astype(str),
            'District': df['district_name'] if 'district_name' in df.columns else range(len(df)),
            'Total_Exp': df['Total_Exp'] if 'Total_Exp' in df.columns else 0
        })
        
        fig = px.scatter(viz_df, x='PCA1', y='PCA2', color='Cluster',
                        hover_data=['District', 'Total_Exp'],
                        title='Cluster Visualization (PCA Projection)',
                        color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Clustering Information:**
        - Number of clusters: {n_clusters}
        - Variance explained by PCA: {sum(pca.explained_variance_ratio_)*100:.2f}%
        - Algorithm: K-Means
        """)
    
    with tab2:
        st.subheader("Cluster Statistics")
        
        summary = df.groupby('Cluster').agg({
            col: 'mean' for col in numeric_data.columns[:10]  # Top 10 features
        }).round(2)
        
        st.dataframe(summary.T, use_container_width=True)
        
        # Cluster sizes
        cluster_sizes = pd.DataFrame({
            'Cluster': range(n_clusters),
            'Count': [sum(clusters == i) for i in range(n_clusters)]
        })
        
        fig = px.bar(cluster_sizes, x='Cluster', y='Count',
                    title='Districts per Cluster',
                    color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Detailed Cluster Analysis")
        
        selected_cluster = st.selectbox("Select Cluster to Analyze", range(n_clusters))
        
        cluster_data = df[df['Cluster'] == selected_cluster]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Cluster {selected_cluster} Characteristics:**")
            st.write(f"Number of Districts: {len(cluster_data)}")
            
            if 'Total_Exp' in df.columns:
                st.write(f"Average Total Exp: ₹{cluster_data['Total_Exp'].mean():,.2f}")
            
            if 'district_name' in df.columns:
                st.markdown("**Sample Districts:**")
                st.write(cluster_data['district_name'].head(10).tolist())
        
        with col2:
            if 'Total_Exp' in df.columns:
                fig = px.box(df, x='Cluster', y='Total_Exp',
                           title=f'Expenditure Distribution Across Clusters')
                st.plotly_chart(fig, use_container_width=True)

# Decision Tree Analysis Page
def show_decision_tree(df):
    st.markdown('<h2 class="sub-header">🌳 Decision Tree Analysis</h2>', 
                unsafe_allow_html=True)
    
    df_processed, le_district, le_month = preprocess_for_decision_tree(df)
    
    # Target selection
    target_option = st.selectbox(
        "Select Target Variable",
        ["High_Productivity", "High_Admin_Spending"]
    )
    
    if target_option not in df_processed.columns:
        st.error(f"Target variable {target_option} could not be created. Check your data.")
        return
    
    # Feature selection based on target
    if target_option == "High_Productivity":
        features = ['Material_and_skilled_Wages', 
                   'Average_days_of_employment_provided_per_Household',
                   'Average_Wage_rate_per_day_per_person',
                   'Total_Households_Worked']
    else:
        features = ['Persondays_of_Central_Liability_so_far',
                   'percent_of_Expenditure_on_Agriculture_Allied_Works',
                   'Material_and_skilled_Wages',
                   'Average_days_of_employment_provided_per_Household']
    
    # Filter available features
    available_features = [f for f in features if f in df_processed.columns]
    
    if len(available_features) == 0:
        st.error("Required features not found in dataset")
        return
    
    X = df_processed[available_features].fillna(df_processed[available_features].median())
    y = df_processed[target_option]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Model parameters
    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider("Max Depth", 2, 10, 5)
    with col2:
        min_samples_leaf = st.slider("Min Samples per Leaf", 5, 50, 10)
    
    # Train model
    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    # Display Results
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🌳 Tree Visualization", "📈 Feature Importance"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
        with col2:
            st.metric("Testing Accuracy", f"{test_acc*100:.2f}%")
        
        st.markdown("**Classification Report:**")
        report = classification_report(y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        st.markdown("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred_test)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Decision Tree Structure")
        
        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(clf, feature_names=available_features, 
                 class_names=['Low', 'High'],
                 filled=True, rounded=True, fontsize=10, ax=ax)
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Feature Importance Analysis")
        
        importance_df = pd.DataFrame({
            'Feature': available_features,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.bar(importance_df, x='Feature', y='Importance',
                        title='Feature Importance Scores',
                        color='Importance', color_continuous_scale='Greens')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(importance_df, use_container_width=True)
            
            st.success("""
            **Interpretation:**
            - Higher values indicate more important features
            - Top features have strongest predictive power
            - Helps identify key performance drivers
            """)

# Advanced Predictions Page
def show_predictions(df):
    st.markdown('<h2 class="sub-header">🔮 Advanced Prediction System</h2>', 
                unsafe_allow_html=True)
    
    df_processed, le_district, le_month = preprocess_for_decision_tree(df)
    
    st.subheader("🎯 Predict District Productivity")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wages = st.number_input("Material & Skilled Wages (₹)", 
                               min_value=0.0, max_value=20000.0, value=1000.0, step=100.0)
    with col2:
        emp_days = st.number_input("Average Employment Days", 
                                   min_value=1, max_value=100, value=30)
    with col3:
        wage_rate = st.number_input("Average Daily Wage Rate (₹)", 
                                    min_value=100.0, max_value=500.0, value=300.0)
    
    if 'district_name' in df.columns:
        district = st.selectbox("Select District", sorted(df['district_name'].unique()))
    else:
        district = "Unknown"
        st.warning("District information not available")
    
    if st.button("🔮 Predict Productivity", type="primary"):
        # Prepare features
        features = ['Material_and_skilled_Wages', 
                   'Average_days_of_employment_provided_per_Household',
                   'Average_Wage_rate_per_day_per_person']
        
        if 'district_encoded' in df_processed.columns:
            features.append('district_encoded')
            district_encoded = le_district.transform([district])[0] if district != "Unknown" else 0
        else:
            district_encoded = 0
        
        available_features = [f for f in features if f in df_processed.columns]
        
        X = df_processed[available_features].fillna(df_processed[available_features].median())
        y = df_processed['High_Productivity'] if 'High_Productivity' in df_processed.columns else pd.Series([0]*len(df))
        
        # Train model
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X, y)
        
        # Make prediction
        if len(available_features) == 4:
            input_data = np.array([[wages, emp_days, wage_rate, district_encoded]])
        else:
            input_data = np.array([[wages, emp_days, wage_rate]])
        
        prediction = clf.predict(input_data)[0]
        probability = clf.predict_proba(input_data)[0]
        
        st.markdown("---")
        
        if prediction == 1:
            st.success(f"✅ **High Productivity Expected!** (Confidence: {probability[1]*100:.1f}%)")
            st.balloons()
        else:
            st.warning(f"⚠️ **Lower Productivity Expected** (Confidence: {probability[0]*100:.1f}%)")
        
        # Show feature contribution
        st.markdown("**Feature Contributions:**")
        feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance, x='Feature', y='Importance',
                    title='What drives this prediction?',
                    color='Importance', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
