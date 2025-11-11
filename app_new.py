#
# Copied code from app.py
#
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
# For displaying matplotlib plots in Streamlit
from matplotlib.patches import Patch 

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
        # Try loading relative to script location
        df = pd.read_csv('data/combined.csv')
        return df
    except FileNotFoundError:
        try:
            # Try loading relative to potential parent directory if run differently
            df = pd.read_csv('../data/combined.csv')
            return df
        except FileNotFoundError:
            st.error("⚠️ Could not find 'combined.csv'. Please ensure it's in a 'data' folder relative to the script or its parent folder!")
            return None
    except Exception as e:
        st.error(f"⚠️ An error occurred while loading the data: {e}")
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
         "🧠 Neural Network Analysis", # <-- Added New Option
         "🔮 Advanced Predictions"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Use the menu to navigate between different analyses")

    # Display dataset info in sidebar
    with st.sidebar.expander("📋 Dataset Info"):
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Total Columns:** {len(df.columns)}")
        if 'fin_year' in df.columns:
             st.write(f"**Date Range:** {df['fin_year'].min()} - {df['fin_year'].max()}")
        else:
             st.write("**Date Range:** N/A")


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
    elif page == "🧠 Neural Network Analysis": # <-- Added New Route
        show_neural_network(df)
    elif page == "🔮 Advanced Predictions":
        show_predictions(df)

# --- Existing Page Functions (show_home, show_eda, show_regression, etc.) ---
# --- These functions remain unchanged from your original code. ---
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
        st.dataframe(df.describe(include=np.number), use_container_width=True) # Ensure only numerical describe

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
            else:
                st.warning("Column 'Average_Wage_rate_per_day_per_person' not found.")

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
            else:
                 st.warning("Column 'Average_days_of_employment_provided_per_Household' not found.")

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
        else:
            st.warning("Columns 'district_name' or 'Total_Exp' not found for this analysis.")

    with tab3:
        st.subheader("Temporal Analysis")

        if 'fin_year' in df.columns and 'month' in df.columns and 'Total_Exp' in df.columns:
            try:
                # Attempt conversion, handling potential errors if format is unexpected
                df['Date'] = pd.to_datetime(df['fin_year'].str.split('-').str[0] + df['month'], format='%Y%B', errors='coerce')
                monthly_data = df.dropna(subset=['Date']).groupby(pd.Grouper(key='Date', freq='M'))['Total_Exp'].mean().reset_index()
                monthly_data['Year'] = monthly_data['Date'].dt.year.astype(str) # For coloring

                fig = px.line(monthly_data, x='Date', y='Total_Exp', color='Year',
                            title="Monthly Average Expenditure Trends",
                            markers=True)
                st.plotly_chart(fig, use_container_width=True)

                st.info("""
                **Temporal Patterns:**
                - Seasonal variations in expenditure
                - Year-over-year comparisons
                - Peak activity months identification
                """)
            except Exception as e:
                st.warning(f"Could not perform temporal analysis. Error creating date: {e}")
        else:
            st.warning("Columns 'fin_year', 'month', or 'Total_Exp' not found for this analysis.")


    with tab4:
        st.subheader("Correlation Analysis")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Limit to first 20 numeric columns if more exist, for readability
        if len(numeric_cols) > 20:
             numeric_cols_subset = numeric_cols[:20]
             st.caption("Showing correlation heatmap for the first 20 numerical features.")
        else:
             numeric_cols_subset = numeric_cols

        if len(numeric_cols_subset) > 1:
            corr = df[numeric_cols_subset].corr()

            fig, ax = plt.subplots(figsize=(14, 12)) # Adjusted size
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, ax=ax, cbar_kws={'label': 'Correlation'}, annot_kws={"size": 8}) # Smaller annotation size
            plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=8) # Rotate x-labels and adjust size
            plt.yticks(fontsize=8) # Adjust y-label size
            plt.tight_layout()
            st.pyplot(fig)

            st.info("""
            **Correlation Insights:**
            - Strong positive correlations indicate related metrics
            - Helps identify redundant features
            - Guides feature selection for modeling
            """)
        else:
            st.warning("Not enough numerical columns found for correlation analysis.")


# Regression Analysis Page
def show_regression(df):
    st.markdown('<h2 class="sub-header">📊 Regression Analysis</h2>',
                unsafe_allow_html=True)

    df_reg = preprocess_for_regression(df) # Use preprocessed data

    tab1, tab2, tab3 = st.tabs(["📉 Simple Linear Regression", "📈 Multiple Regression", "🔄 Polynomial Regression"])

    with tab1:
        st.subheader("Simple Linear Regression: Wages vs Total Expenditure")

        if 'Total_Exp' in df_reg.columns and 'Wages' in df_reg.columns:
            X = df_reg[['Total_Exp']].values
            y = df_reg['Wages'].values

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
        else:
            st.warning("Columns 'Total_Exp' or 'Wages' not found for Simple Linear Regression.")

    with tab2:
        st.subheader("Multiple Linear Regression")

        # Define potential feature columns
        potential_feature_cols = ['Total_Exp', 'Approved_Labour_Budget', 'Women_Persondays']
        target_col = 'Wages'

        # Check if all columns exist
        available_feature_cols = [col for col in potential_feature_cols if col in df_reg.columns]

        if available_feature_cols and target_col in df_reg.columns:
            st.write(f"Using features: {', '.join(available_feature_cols)}")
            X = df_reg[available_feature_cols] # Use available features
            y = df_reg[target_col]

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
                'Feature': available_feature_cols, # Use available features
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
            ax.plot([min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())],
                   [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())], # Adjust line limits
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
            - Model explains {r2_test*100:.2f}% of wage variance using selected features.
            """)
        else:
             st.warning("Required columns ('Total_Exp', 'Approved_Labour_Budget', 'Women_Persondays', 'Wages') not found for Multiple Linear Regression.")

    with tab3:
        st.subheader("Polynomial Regression")

        degree = st.slider("Select Polynomial Degree", 2, 5, 2, key='poly_degree_slider') # Added key

        if 'Total_Exp' in df_reg.columns and 'Wages' in df_reg.columns:
            X = df_reg[['Total_Exp']].values
            y = df_reg['Wages'].values

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
            - Captures potential non-linear relationships
            - Degree {degree} polynomial fit
            """)
        else:
            st.warning("Columns 'Total_Exp' or 'Wages' not found for Polynomial Regression.")


# Clustering Analysis Page
def show_clustering(df):
    st.markdown('<h2 class="sub-header">🎯 Clustering Analysis</h2>',
                unsafe_allow_html=True)

    try:
        numeric_data, scaled_data = preprocess_for_clustering(df)
        if numeric_data.empty:
             st.warning("No numerical data found for clustering after preprocessing.")
             return
    except Exception as e:
        st.error(f"Error during clustering preprocessing: {e}")
        return

    # Sidebar controls
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 3, key='cluster_slider') # Added key

    # Perform K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42) # Set n_init explicitly
    clusters = kmeans.fit_predict(scaled_data)

    # Use original DataFrame index for cluster assignment
    df_clustered = df.loc[numeric_data.index].copy() # Ensure working with relevant rows
    df_clustered['Cluster'] = clusters

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["🗺️ Cluster Visualization", "📊 Cluster Statistics", "📋 Detailed Analysis"])

    with tab1:
        st.subheader(f"District Clusters (K={n_clusters})")

        # Create visualization dataframe
        viz_df_data = {
            'PCA1': pca_data[:, 0],
            'PCA2': pca_data[:, 1],
            'Cluster': clusters.astype(str),
            'District': df_clustered['district_name'] if 'district_name' in df_clustered.columns else range(len(df_clustered)),
            'Total_Exp': df_clustered['Total_Exp'] if 'Total_Exp' in df_clustered.columns else 0
        }
        viz_df = pd.DataFrame(viz_df_data)


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

        # Aggregate using the DataFrame with cluster assignments
        cluster_summary_cols = [col for col in numeric_data.columns if col in df_clustered.columns]
        # Use only first 10 available numeric cols for summary display
        if len(cluster_summary_cols) > 10:
             cluster_summary_cols_subset = cluster_summary_cols[:10]
        else:
             cluster_summary_cols_subset = cluster_summary_cols

        if cluster_summary_cols_subset:
            summary = df_clustered.groupby('Cluster')[cluster_summary_cols_subset].mean().round(2)
            st.dataframe(summary.T, use_container_width=True)
        else:
             st.warning("No numeric columns available to calculate cluster statistics.")

        # Cluster sizes
        cluster_sizes = df_clustered['Cluster'].value_counts().reset_index()
        cluster_sizes.columns = ['Cluster', 'Count']
        cluster_sizes = cluster_sizes.sort_values('Cluster')


        fig_sizes = px.bar(cluster_sizes, x='Cluster', y='Count',
                    title='Records per Cluster',
                    color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_sizes, use_container_width=True)

    with tab3:
        st.subheader("Detailed Cluster Analysis")

        selected_cluster = st.selectbox("Select Cluster to Analyze", sorted(df_clustered['Cluster'].unique()), key='cluster_select') # Added key

        cluster_data_selected = df_clustered[df_clustered['Cluster'] == selected_cluster]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Cluster {selected_cluster} Characteristics:**")
            st.write(f"Number of Records: {len(cluster_data_selected)}")

            if 'Total_Exp' in cluster_data_selected.columns:
                st.write(f"Average Total Exp: ₹{cluster_data_selected['Total_Exp'].mean():,.2f}")

            if 'district_name' in cluster_data_selected.columns:
                st.markdown("**Sample Districts (if available):**")
                st.write(cluster_data_selected['district_name'].head(10).tolist())
            else:
                st.write("District names not available.")

        with col2:
            plot_col = st.selectbox("Select column for distribution across clusters:",
                                    [col for col in cluster_summary_cols if col != 'Cluster'],
                                    index=cluster_summary_cols.index('Total_Exp') if 'Total_Exp' in cluster_summary_cols else 0,
                                    key='cluster_dist_select') # Added key

            if plot_col:
                fig_box = px.box(df_clustered, x='Cluster', y=plot_col,
                               title=f'{plot_col} Distribution Across Clusters')
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                 st.warning("No column selected for box plot.")


# Decision Tree Analysis Page
def show_decision_tree(df):
    st.markdown('<h2 class="sub-header">🌳 Decision Tree Analysis</h2>',
                unsafe_allow_html=True)

    try:
        df_processed, le_district, le_month = preprocess_for_decision_tree(df)
    except Exception as e:
        st.error(f"Error during decision tree preprocessing: {e}")
        return

    # Target selection
    available_targets = [t for t in ["High_Productivity", "High_Admin_Spending"] if t in df_processed.columns]
    if not available_targets:
         st.error("Target variables ('High_Productivity', 'High_Admin_Spending') could not be created. Check input data and preprocessing logic.")
         return

    target_option = st.selectbox(
        "Select Target Variable",
        available_targets,
        key='dt_target_select' # Added key
    )

    # Feature selection based on target
    if target_option == "High_Productivity":
        features = ['Material_and_skilled_Wages',
                   'Average_days_of_employment_provided_per_Household',
                   'Average_Wage_rate_per_day_per_person',
                   'Total_Households_Worked']
    else: # High_Admin_Spending
        features = ['Persondays_of_Central_Liability_so_far',
                   'percent_of_Expenditure_on_Agriculture_Allied_Works',
                   'Material_and_skilled_Wages',
                   'Average_days_of_employment_provided_per_Household']

    # Filter available features from the processed dataframe
    available_features = [f for f in features if f in df_processed.columns]

    if len(available_features) == 0:
        st.error(f"None of the required features for target '{target_option}' found in dataset: {', '.join(features)}")
        return
    elif len(available_features) < len(features):
        st.warning(f"Using available features: {', '.join(available_features)}. Missing: {', '.join(set(features) - set(available_features))}")


    X = df_processed[available_features].fillna(df_processed[available_features].median())
    y = df_processed[target_option]

    if len(X) != len(y):
        st.error("Feature and target data lengths do not match after preprocessing.")
        return
    if y.nunique() < 2:
         st.error(f"Target variable '{target_option}' has only one class after preprocessing. Cannot train classifier.")
         return


    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError as e:
        st.error(f"Error during train-test split (possibly due to insufficient samples for stratification): {e}")
        st.write(f"Target variable distribution:\n{y.value_counts()}")
        return


    # Model parameters
    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider("Max Depth", 2, 10, 5, key='dt_depth_slider') # Added key
    with col2:
        min_samples_leaf = st.slider("Min Samples per Leaf", 5, 50, 10, key='dt_leaf_slider') # Added key

    # Train model
    try:
        clf = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        clf.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Error training Decision Tree model: {e}")
        return

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
        try:
            report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0) # Handle zero division
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate classification report: {e}")

        st.markdown("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred_test)

        fig, ax = plt.subplots(figsize=(6, 4)) # Smaller size
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=clf.classes_, yticklabels=clf.classes_) # Add labels
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    with tab2:
        st.subheader("Decision Tree Structure")

        try:
            fig, ax = plt.subplots(figsize=(20, 12)) # Consider making size adjustable or limiting depth for display
            plot_tree(clf, feature_names=available_features,
                     class_names=['Low', 'High'], # Assuming binary classification 0/1
                     filled=True, rounded=True, fontsize=10, ax=ax, max_depth=5) # Limit depth for initial display
            st.pyplot(fig)
            st.caption("Displaying tree up to depth 5 for readability. Adjust Max Depth slider for model training.")
        except Exception as e:
            st.warning(f"Could not plot decision tree: {e}")


    with tab3:
        st.subheader("Feature Importance Analysis")

        importance_df = pd.DataFrame({
            'Feature': available_features,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_imp = px.bar(importance_df, x='Importance', y='Feature', # Swapped axes for better label reading
                        orientation='h', # Horizontal bar chart
                        title='Feature Importance Scores',
                        color='Importance', color_continuous_scale='Greens')
            # fig_imp.update_layout(xaxis_tickangle=-45) # Not needed for horizontal
            st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            st.dataframe(importance_df, use_container_width=True)

            st.success("""
            **Interpretation:**
            - Higher values indicate more important features for this model split.
            - Top features have strongest predictive power according to the tree.
            - Helps identify key drivers for the selected target variable.
            """)


# Advanced Predictions Page
def show_predictions(df):
    st.markdown('<h2 class="sub-header">🔮 Advanced Prediction System</h2>',
                unsafe_allow_html=True)

    try:
        df_processed, le_district, le_month = preprocess_for_decision_tree(df)
        if 'High_Productivity' not in df_processed.columns:
             st.error("Target variable 'High_Productivity' could not be created for predictions.")
             return
    except Exception as e:
        st.error(f"Error during prediction preprocessing: {e}")
        return


    st.subheader("🎯 Predict District Productivity (Using Decision Tree)")

    # Define features used for High_Productivity prediction in Decision Tree section
    productivity_features = ['Material_and_skilled_Wages',
                             'Average_days_of_employment_provided_per_Household',
                             'Average_Wage_rate_per_day_per_person',
                             'Total_Households_Worked'] # Assuming this was used

    # Check which features are actually available after preprocessing
    available_pred_features = [f for f in productivity_features if f in df_processed.columns]

    if len(available_pred_features) == 0:
        st.error("No features available for productivity prediction.")
        return
    elif len(available_pred_features) < len(productivity_features):
         st.warning(f"Using available features for prediction: {', '.join(available_pred_features)}. Missing: {', '.join(set(productivity_features) - set(available_pred_features))}")


    st.markdown("**Enter Feature Values:**")
    input_values = {}
    cols = st.columns(len(available_pred_features))
    for i, feature in enumerate(available_pred_features):
         # Try to get median as default, else use 0
         default_val = float(df_processed[feature].median()) if pd.notna(df_processed[feature].median()) else 0.0
         # Define reasonable min/max based on feature name or use broad defaults
         min_val = 0.0
         max_val = float(df_processed[feature].max()) * 1.5 if pd.notna(df_processed[feature].max()) else 10000.0
         step = 1.0 if np.issubdtype(df_processed[feature].dtype, np.integer) else 10.0

         input_values[feature] = cols[i].number_input(
             f"{feature}",
             min_value=min_val,
             max_value=max_val,
             value=default_val,
             step=step,
             key=f"pred_input_{feature}" # Added key
         )


    # District input only if district_encoded was available and used
    district_encoded_val = None
    if 'district_encoded' in df_processed.columns and 'district_name' in df.columns:
        if st.checkbox("Include District Information (if available)?", key='pred_district_check'): # Added key
             district_list = ["Unknown"] + sorted(df['district_name'].unique())
             district = st.selectbox("Select District", district_list, key='pred_district_select') # Added key
             if district != "Unknown":
                 try:
                     district_encoded_val = le_district.transform([district])[0]
                     available_pred_features.append('district_encoded') # Add if used
                 except ValueError:
                     st.warning(f"District '{district}' not seen during initial encoding. Using 0.")
                     district_encoded_val = 0
             else:
                 district_encoded_val = 0 # Or use median/mode if preferred for 'Unknown'
             input_values['district_encoded'] = district_encoded_val

    if st.button("🔮 Predict Productivity", type="primary", key='pred_button'): # Added key
        # Prepare data only with available features
        X = df_processed[available_pred_features].fillna(df_processed[available_pred_features].median())
        y = df_processed['High_Productivity']

        # Train model (using the same parameters as in the Decision Tree section for consistency)
        try:
             clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, random_state=42)
             clf.fit(X, y)
        except Exception as e:
             st.error(f"Error training prediction model: {e}")
             return

        # Prepare input data array based ONLY on the features used for training THIS prediction model
        input_data_list = []
        final_features_used = available_pred_features # Features used in clf.fit(X, y)
        for feature in final_features_used:
             input_data_list.append(input_values[feature])

        input_data = np.array([input_data_list])


        # Make prediction
        try:
             prediction = clf.predict(input_data)[0]
             probability = clf.predict_proba(input_data)[0]

             st.markdown("---")

             if prediction == 1:
                 st.success(f"✅ **High Productivity Expected!** (Confidence: {probability[1]*100:.1f}%)")
                 st.balloons()
             else:
                 st.warning(f"⚠️ **Lower Productivity Expected** (Confidence: {probability[0]*100:.1f}%)")

             # Show feature contribution (Importance from THIS specific prediction model)
             st.markdown("**Feature Importances (for this prediction model):**")
             feature_importance = pd.DataFrame({
                 'Feature': final_features_used,
                 'Importance': clf.feature_importances_
             }).sort_values('Importance', ascending=False)

             fig_pred_imp = px.bar(feature_importance, x='Importance', y='Feature', # Swapped axes
                             orientation='h', # Horizontal
                             title='Feature Importance Driving Prediction',
                             color='Importance', color_continuous_scale='RdYlGn')
             st.plotly_chart(fig_pred_imp, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Input Data used:", input_data)


# --- >>> NEW SECTION FOR NEURAL NETWORK RESULTS <<< ---
def show_neural_network(df):
    """Displays the results from the offline Neural Network training."""
    st.markdown('<h2 class="sub-header">🧠 Neural Network Wage Prediction Results</h2>',
                unsafe_allow_html=True)

    st.info("""
    This section presents the **results** obtained from training Neural Network models (specifically MLPRegressor from Scikit-learn) **offline** using the provided Jupyter Notebook.
    - ✅ **No TensorFlow/DLL issues:** This approach works seamlessly on standard Python setups.
    - 📊 The models were trained to predict the **'Wages'** column based on selected numerical features.
    - 💡 The table below compares the performance against traditional ML models.
    """)
    st.markdown("---")

    # --- Hardcoded Results from the Jupyter Notebook ---
    nn_results_data = {
        'Model': [
            'Random Forest',
            'Gradient Boosting',
            'Optimized NN',
            'Deep NN (3 layers)',
            'Simple NN (1 layer)',
            'Ridge Regression'
        ],
        'RMSE': [
            226.67, 381.37, 570.74, 606.41, 884.13, 899.70
        ],
        'MAE': [
            67.06, 232.19, 347.20, 338.60, 600.48, 543.76
        ],
        'R² Score': [
            0.9989, 0.9970, 0.9933, 0.9924, 0.9839, 0.9834
        ],
        'Type': [
            'Traditional ML', 'Traditional ML', 'Neural Network',
            'Neural Network', 'Neural Network', 'Traditional ML'
        ]
    }
    results_df = pd.DataFrame(nn_results_data)

    st.subheader("📊 Complete Model Comparison")
    st.dataframe(results_df.style.format({
        'RMSE': '{:,.2f}',
        'MAE': '{:,.2f}',
        'R² Score': '{:.4f}'
    }).highlight_max(subset=['R² Score'], color='lightgreen')
      .apply(lambda x: ['background-color: lightblue' if x.Type == 'Neural Network' else '' for i in x], axis=1),
      use_container_width=True
    )

    st.markdown("---")

    # --- Details of the Best Neural Network ---
    st.subheader("🏆 Best Neural Network Model: Optimized NN")
    st.write("Hyperparameters found via GridSearchCV:")
    st.json({
        'alpha': 0.001,
        'hidden_layer_sizes': [100, 50, 25],
        'learning_rate_init': 0.01,
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 300,
        'early_stopping': True
    })

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimized NN R² Score", f"{0.9933:.4f}")
    with col2:
        st.metric("Optimized NN RMSE", f"{570.74:,.2f}")
    with col3:
        st.metric("Optimized NN MAE", f"{347.20:,.2f}")

    st.markdown("---")

    # --- Feature Importance (from Random Forest in the notebook) ---
    st.subheader("📈 Feature Importance (from Comparative Random Forest Model)")
    feature_importance_data = {
        'Feature': [
            'Total_Exp', 'Persondays_of_Central_Liability_so_far', 'Approved_Labour_Budget',
            'Number_of_Ongoing_Works', 'Women_Persondays', 'Total_Individuals_Worked',
            'Number_of_Completed_Works', 'Total_Households_Worked', 'ST_persondays',
            'SC_persondays'
        ],
        'Importance': [
            0.543737, 0.295124, 0.139887, 0.011026, 0.002967, 0.002652,
            0.001643, 0.001466, 0.000932, 0.000567
        ]
    }
    importance_df = pd.DataFrame(feature_importance_data)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_imp = px.bar(importance_df, x='Importance', y='Feature',
                         orientation='h', title='Feature Importance Scores (RF)',
                         color='Importance', color_continuous_scale='Greens')
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}) # Show most important at top
        st.plotly_chart(fig_imp, use_container_width=True)
    with col2:
        st.dataframe(importance_df, use_container_width=True)
        st.caption("Importance scores indicate contribution to the Random Forest model's wage prediction.")

    st.markdown("---")

    # --- Performance Comparison Visualization ---
    st.subheader("📉 Model Performance Visualization")
    fig_comp, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['skyblue' if t == 'Neural Network' else 'lightcoral' for t in results_df['Type']]
    # RMSE
    axes[0].barh(results_df['Model'], results_df['RMSE'], color=colors)
    axes[0].set_xlabel('RMSE (Lower is Better)', fontweight='bold')
    axes[0].set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()
    # MAE
    axes[1].barh(results_df['Model'], results_df['MAE'], color=colors)
    axes[1].set_xlabel('MAE (Lower is Better)', fontweight='bold')
    axes[1].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].invert_yaxis()
    # R² Score
    axes[2].barh(results_df['Model'], results_df['R² Score'], color=colors)
    axes[2].set_xlabel('R² Score (Higher is Better)', fontweight='bold')
    axes[2].set_title('R² Score', fontsize=13, fontweight='bold')
    axes[2].set_xlim([0.98, 1.0]) # Adjust xlim for better R² visualization
    axes[2].grid(axis='x', alpha=0.3)
    axes[2].invert_yaxis()
    # Legend
    legend_elements = [Patch(facecolor='skyblue', label='Neural Network'), Patch(facecolor='lightcoral', label='Traditional ML')]
    axes[2].legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    st.pyplot(fig_comp)


    st.markdown("---")

    # --- Illustrative Prediction Input/Output ---
    st.subheader("💡 Illustrative Wage Prediction (Optimized NN)")
    st.write("Enter hypothetical values for the features used by the Neural Network model:")

    nn_feature_columns = [
        'Total_Exp', 'Approved_Labour_Budget', 'Women_Persondays',
        'Persondays_of_Central_Liability_so_far', 'Total_Households_Worked',
        'Total_Individuals_Worked', 'Number_of_Completed_Works',
        'Number_of_Ongoing_Works', 'SC_persondays', 'ST_persondays'
    ]

    input_nn_values = {}
    cols = st.columns(3) # Use columns for better layout

    for i, feature in enumerate(nn_feature_columns):
         col_index = i % 3
         # Use median from the original dataframe if available, otherwise 0
         default_val = float(df[feature].median()) if feature in df.columns and pd.notna(df[feature].median()) else 0.0
         input_nn_values[feature] = cols[col_index].number_input(
             f"{feature}",
             min_value=0.0,
             value=default_val,
             step=100.0,
             key=f"nn_input_{feature}" # Unique key
         )

    if st.button("Predict Wage (Illustrative)", type="primary", key='nn_predict_button'): # Unique key
        # --- This is NOT a real prediction. It's a placeholder. ---
        # A real implementation would load the scaler, scale the input, load the model, and predict.
        # Here, we just display a plausible-looking output based on the input values' magnitude.
        
        # Simple heuristic for illustration: scale total_exp and add some base
        predicted_wage_placeholder = (input_nn_values.get('Total_Exp', 0) * 0.5) + \
                                     (input_nn_values.get('Approved_Labour_Budget', 0) * 0.1) + \
                                     (input_nn_values.get('Persondays_of_Central_Liability_so_far', 0) * 0.2) + 500

        st.success(f"💡 **Illustrative Predicted Wage:** ₹ {predicted_wage_placeholder:,.2f}")
        st.caption("⚠️ **Note:** This is a sample prediction for demonstration purposes only. It uses a simple heuristic based on inputs and does **not** reflect the actual trained Neural Network's calculation.")


# --- END OF NEW SECTION ---


if __name__ == "__main__":
    main()