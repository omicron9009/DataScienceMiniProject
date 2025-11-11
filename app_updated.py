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
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

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
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
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
    exclude_cols = ['fin_year', 'month', 'state_code', 'state_name', 'district_code', 'district_name', 'Remarks']
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

def perform_backward_elimination(df, X_cols, y_col, sl=0.05):
    """
    Perform backward elimination on the given features

    Parameters:
    - df: DataFrame containing the data
    - X_cols: List of independent variable column names
    - y_col: Dependent variable column name
    - sl: Significance level (default 0.05)

    Returns:
    - final_model: The final OLS model after elimination
    - features_kept: List of features that remained significant
    - elimination_history: List of eliminated features with their p-values
    """
    # Prepare data
    X = df[X_cols].fillna(df[X_cols].median())
    y = df[y_col].fillna(df[y_col].median())

    # Add constant
    X = sm.add_constant(X)

    # Track elimination
    features = list(X.columns)
    elimination_history = []

    while len(features) > 0:
        # Fit model
        model = sm.OLS(y, X[features]).fit()

        # Get p-values
        p_values = model.pvalues

        # Find max p-value
        max_p_value = p_values.max()

        if max_p_value > sl:
            feature_to_remove = p_values.idxmax()
            elimination_history.append({
                'feature': feature_to_remove,
                'p_value': max_p_value
            })
            features.remove(feature_to_remove)
        else:
            break

    # Final model
    final_model = sm.OLS(y, X[features]).fit()
    features_kept = [f for f in features if f != 'const']

    return final_model, features_kept, elimination_history

# Main Dashboard
def main():
    st.markdown('<h1 style="text-align: center; color: #2E7D32;">🌾 MGNREGA Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Comprehensive Statistical & Machine Learning Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load data
    df = load_data()

    if df is None:
        return

    # Sidebar Navigation
    st.sidebar.title("📊 Navigation Menu")

    menu_options = [
        "🏠 Home & Overview",
        "📈 Univariate Analysis",
        "🔄 Bivariate Analysis", 
        "🎯 Multivariate Analysis",
        "📉 Regression Analysis",
        "⚙️ Backward Elimination",
        "🌳 Decision Tree",
        "🎪 Clustering Analysis",
        "📊 PCA Analysis"
    ]

    choice = st.sidebar.radio("Select Analysis Type:", menu_options)

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Use the menu to navigate between different analyses")

    # ============================================================================
    # HOME & OVERVIEW
    # ============================================================================
    if choice == "🏠 Home & Overview":
        st.header("📋 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("📊 Data Types & Missing Values")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values
            })
            st.dataframe(dtype_df, use_container_width=True)

        with col2:
            st.write("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            st.dataframe(missing_df, use_container_width=True)

        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    # ============================================================================
    # UNIVARIATE ANALYSIS
    # ============================================================================
    elif choice == "📈 Univariate Analysis":
        st.header("📈 Univariate Analysis")
        st.write("Analyze the distribution and characteristics of individual variables")

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        tab1, tab2 = st.tabs(["📊 Numeric Variables", "📋 Categorical Variables"])

        with tab1:
            st.subheader("Numeric Variable Analysis")

            selected_numeric = st.selectbox("Select a numeric variable:", numeric_cols)

            if selected_numeric:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Mean", f"{df[selected_numeric].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[selected_numeric].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{df[selected_numeric].std():.2f}")

                # Create visualizations
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))

                # Histogram
                axes[0, 0].hist(df[selected_numeric].dropna(), bins=30, color='skyblue', edgecolor='black')
                axes[0, 0].set_title(f'Histogram of {selected_numeric}')
                axes[0, 0].set_xlabel(selected_numeric)
                axes[0, 0].set_ylabel('Frequency')

                # Box Plot
                axes[0, 1].boxplot(df[selected_numeric].dropna(), vert=True)
                axes[0, 1].set_title(f'Box Plot of {selected_numeric}')
                axes[0, 1].set_ylabel(selected_numeric)

                # Density Plot
                df[selected_numeric].dropna().plot(kind='density', ax=axes[1, 0], color='green')
                axes[1, 0].set_title(f'Density Plot of {selected_numeric}')
                axes[1, 0].set_xlabel(selected_numeric)

                # Q-Q Plot
                sm.qqplot(df[selected_numeric].dropna(), line='s', ax=axes[1, 1])
                axes[1, 1].set_title(f'Q-Q Plot of {selected_numeric}')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Statistical tests
                st.subheader("📊 Descriptive Statistics")
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Kurtosis'],
                    'Value': [
                        df[selected_numeric].count(),
                        df[selected_numeric].mean(),
                        df[selected_numeric].std(),
                        df[selected_numeric].min(),
                        df[selected_numeric].quantile(0.25),
                        df[selected_numeric].quantile(0.50),
                        df[selected_numeric].quantile(0.75),
                        df[selected_numeric].max(),
                        df[selected_numeric].skew(),
                        df[selected_numeric].kurtosis()
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)

        with tab2:
            st.subheader("Categorical Variable Analysis")

            if len(categorical_cols) > 0:
                selected_cat = st.selectbox("Select a categorical variable:", categorical_cols)

                if selected_cat:
                    # Value counts
                    value_counts = df[selected_cat].value_counts().head(20)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Top Categories:**")
                        st.dataframe(value_counts, use_container_width=True)

                    with col2:
                        st.metric("Unique Values", df[selected_cat].nunique())
                        st.metric("Most Common", value_counts.index[0])
                        st.metric("Mode Frequency", value_counts.values[0])

                    # Bar plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    value_counts.plot(kind='bar', ax=ax, color='coral')
                    ax.set_title(f'Frequency Distribution of {selected_cat}')
                    ax.set_xlabel(selected_cat)
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            else:
                st.info("No categorical variables found in the dataset.")

    # ============================================================================
    # BIVARIATE ANALYSIS
    # ============================================================================
    elif choice == "🔄 Bivariate Analysis":
        st.header("🔄 Bivariate Analysis")
        st.write("Explore relationships between two variables")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        tab1, tab2, tab3 = st.tabs(["📊 Numeric vs Numeric", "📋 Numeric vs Categorical", "🔥 Correlation Matrix"])

        with tab1:
            st.subheader("Numeric vs Numeric Analysis")

            col1, col2 = st.columns(2)

            with col1:
                x_var = st.selectbox("Select X variable:", numeric_cols, key='bivar_x')
            with col2:
                y_var = st.selectbox("Select Y variable:", numeric_cols, key='bivar_y', index=1 if len(numeric_cols) > 1 else 0)

            if x_var and y_var and x_var != y_var:
                # Calculate correlation
                correlation = df[[x_var, y_var]].corr().iloc[0, 1]

                st.metric("Correlation Coefficient", f"{correlation:.4f}")

                # Create visualizations
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                # Scatter plot
                axes[0].scatter(df[x_var], df[y_var], alpha=0.5, color='blue')
                axes[0].set_xlabel(x_var)
                axes[0].set_ylabel(y_var)
                axes[0].set_title(f'Scatter Plot: {x_var} vs {y_var}')

                # Add regression line
                z = np.polyfit(df[x_var].dropna(), df[y_var].dropna(), 1)
                p = np.poly1d(z)
                axes[0].plot(df[x_var], p(df[x_var]), "r--", alpha=0.8, linewidth=2)

                # Hexbin plot
                axes[1].hexbin(df[x_var], df[y_var], gridsize=20, cmap='YlOrRd')
                axes[1].set_xlabel(x_var)
                axes[1].set_ylabel(y_var)
                axes[1].set_title(f'Hexbin Density Plot')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with tab2:
            st.subheader("Numeric vs Categorical Analysis")

            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            if len(categorical_cols) > 0:
                col1, col2 = st.columns(2)

                with col1:
                    num_var = st.selectbox("Select numeric variable:", numeric_cols, key='bivar_num')
                with col2:
                    cat_var = st.selectbox("Select categorical variable:", categorical_cols, key='bivar_cat')

                if num_var and cat_var:
                    # Limit to top categories
                    top_cats = df[cat_var].value_counts().head(10).index
                    df_filtered = df[df[cat_var].isin(top_cats)]

                    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                    # Box plot
                    df_filtered.boxplot(column=num_var, by=cat_var, ax=axes[0])
                    axes[0].set_title(f'{num_var} by {cat_var}')
                    axes[0].set_xlabel(cat_var)
                    axes[0].set_ylabel(num_var)
                    plt.sca(axes[0])
                    plt.xticks(rotation=45, ha='right')

                    # Violin plot
                    positions = range(len(top_cats))
                    data_to_plot = [df_filtered[df_filtered[cat_var] == cat][num_var].dropna() for cat in top_cats]
                    axes[1].violinplot(data_to_plot, positions=positions, showmeans=True)
                    axes[1].set_xticks(positions)
                    axes[1].set_xticklabels(top_cats, rotation=45, ha='right')
                    axes[1].set_title(f'Violin Plot: {num_var} by {cat_var}')
                    axes[1].set_ylabel(num_var)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            else:
                st.info("No categorical variables available for analysis.")

        with tab3:
            st.subheader("Correlation Matrix")

            # Allow user to select variables
            selected_vars = st.multiselect(
                "Select variables for correlation matrix:",
                numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))]
            )

            if len(selected_vars) >= 2:
                corr_matrix = df[selected_vars].corr()

                # Heatmap
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
                ax.set_title('Correlation Matrix Heatmap')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Show correlation table
                st.subheader("Correlation Coefficients")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None), 
                           use_container_width=True)
            else:
                st.warning("Please select at least 2 variables for correlation analysis.")

    # ============================================================================
    # MULTIVARIATE ANALYSIS
    # ============================================================================
    elif choice == "🎯 Multivariate Analysis":
        st.header("🎯 Multivariate Analysis")
        st.write("Analyze relationships among multiple variables simultaneously")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        tab1, tab2, tab3 = st.tabs(["📊 Pair Plot", "🎨 3D Scatter", "🔥 Advanced Correlations"])

        with tab1:
            st.subheader("Pair Plot Analysis")

            # Select variables
            max_vars = min(6, len(numeric_cols))
            selected_vars = st.multiselect(
                "Select variables for pair plot (max 6 recommended):",
                numeric_cols,
                default=numeric_cols[:max_vars]
            )

            if len(selected_vars) >= 2:
                if st.button("Generate Pair Plot"):
                    with st.spinner("Generating pair plot..."):
                        # Sample data if too large
                        sample_size = min(1000, len(df))
                        df_sample = df[selected_vars].sample(n=sample_size, random_state=42)

                        fig = sns.pairplot(df_sample, diag_kind='kde', plot_kws={'alpha': 0.6})
                        st.pyplot(fig)
                        plt.close()
            else:
                st.warning("Please select at least 2 variables.")

        with tab2:
            st.subheader("3D Scatter Plot")

            if len(numeric_cols) >= 3:
                col1, col2, col3 = st.columns(3)

                with col1:
                    x_var = st.selectbox("X axis:", numeric_cols, key='3d_x')
                with col2:
                    y_var = st.selectbox("Y axis:", numeric_cols, key='3d_y', index=1 if len(numeric_cols) > 1 else 0)
                with col3:
                    z_var = st.selectbox("Z axis:", numeric_cols, key='3d_z', index=2 if len(numeric_cols) > 2 else 0)

                if x_var and y_var and z_var:
                    # Sample for performance
                    sample_size = min(1000, len(df))
                    df_sample = df[[x_var, y_var, z_var]].sample(n=sample_size, random_state=42).dropna()

                    fig = go.Figure(data=[go.Scatter3d(
                        x=df_sample[x_var],
                        y=df_sample[y_var],
                        z=df_sample[z_var],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=df_sample[z_var],
                            colorscale='Viridis',
                            showscale=True
                        )
                    )])

                    fig.update_layout(
                        title=f'3D Scatter Plot',
                        scene=dict(
                            xaxis_title=x_var,
                            yaxis_title=y_var,
                            zaxis_title=z_var
                        ),
                        width=800,
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 3 numeric variables for 3D visualization.")

        with tab3:
            st.subheader("Advanced Correlation Analysis")

            selected_vars = st.multiselect(
                "Select variables for advanced analysis:",
                numeric_cols,
                default=numeric_cols[:min(8, len(numeric_cols))]
            )

            if len(selected_vars) >= 3:
                corr_matrix = df[selected_vars].corr()

                # Clustered heatmap
                st.write("**Clustered Correlation Heatmap:**")
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.clustermap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                             square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                             figsize=(12, 10))
                st.pyplot(plt.gcf())
                plt.close()

                # Find highly correlated pairs
                st.write("**Highly Correlated Variable Pairs (|r| > 0.7):**")
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })

                if high_corr:
                    st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
                else:
                    st.info("No highly correlated pairs found (threshold: |r| > 0.7)")
            else:
                st.warning("Please select at least 3 variables.")

    # ============================================================================
    # REGRESSION ANALYSIS
    # ============================================================================
    elif choice == "📉 Regression Analysis":
        st.header("📉 Multiple Linear Regression Analysis")

        df_clean = preprocess_for_regression(df)
        numeric_columns = df_clean.columns.tolist()

        st.subheader("🎯 Select Variables")

        col1, col2 = st.columns(2)

        with col1:
            dependent_var = st.selectbox("Select Dependent Variable (Y):", numeric_columns)

        with col2:
            independent_vars = st.multiselect(
                "Select Independent Variables (X):",
                [col for col in numeric_columns if col != dependent_var],
                default=[col for col in numeric_columns if col != dependent_var][:3]
            )

        if dependent_var and independent_vars:
            if st.button("🚀 Run Regression Analysis"):
                # Prepare data
                X = df_clean[independent_vars]
                y = df_clean[dependent_var]

                # Remove any rows with NaN
                mask = ~(X.isna().any(axis=1) | y.isna())
                X = X[mask]
                y = y[mask]

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
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)
                rmse_train = np.sqrt(mse_train)
                rmse_test = np.sqrt(mse_test)

                # Display metrics
                st.subheader("📊 Model Performance")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("R² Score (Train)", f"{r2_train:.4f}")
                    st.metric("R² Score (Test)", f"{r2_test:.4f}")

                with col2:
                    st.metric("RMSE (Train)", f"{rmse_train:.2f}")
                    st.metric("RMSE (Test)", f"{rmse_test:.2f}")

                with col3:
                    st.metric("MSE (Train)", f"{mse_train:.2f}")
                    st.metric("MSE (Test)", f"{mse_test:.2f}")

                # Coefficients
                st.subheader("📈 Model Coefficients")
                coef_df = pd.DataFrame({
                    'Feature': independent_vars,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)

                st.dataframe(coef_df, use_container_width=True)

                # Visualizations
                st.subheader("📊 Regression Diagnostics")

                fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                # Actual vs Predicted
                axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, color='blue')
                axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[0, 0].set_xlabel('Actual Values')
                axes[0, 0].set_ylabel('Predicted Values')
                axes[0, 0].set_title('Actual vs Predicted Values')

                # Residuals vs Predicted
                residuals_test = y_test - y_pred_test
                axes[0, 1].scatter(y_pred_test, residuals_test, alpha=0.5, color='green')
                axes[0, 1].axhline(y=0, color='r', linestyle='--')
                axes[0, 1].set_xlabel('Predicted Values')
                axes[0, 1].set_ylabel('Residuals')
                axes[0, 1].set_title('Residual Plot')

                # Distribution of Residuals
                axes[1, 0].hist(residuals_test, bins=30, color='purple', edgecolor='black')
                axes[1, 0].set_xlabel('Residuals')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Distribution of Residuals')

                # Q-Q Plot
                sm.qqplot(residuals_test, line='s', ax=axes[1, 1])
                axes[1, 1].set_title('Q-Q Plot of Residuals')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Feature Importance
                st.subheader("🎯 Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                coef_df['abs_coef'] = abs(coef_df['Coefficient'])
                coef_df_sorted = coef_df.sort_values('abs_coef', ascending=True)
                ax.barh(coef_df_sorted['Feature'], coef_df_sorted['Coefficient'], color='teal')
                ax.set_xlabel('Coefficient Value')
                ax.set_title('Feature Coefficients')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # ============================================================================
    # BACKWARD ELIMINATION
    # ============================================================================
    elif choice == "⚙️ Backward Elimination":
        st.header("⚙️ Backward Elimination - Feature Selection")
        st.write("Automatically remove non-significant features based on p-values")

        df_clean = preprocess_for_regression(df)
        numeric_columns = df_clean.columns.tolist()

        st.subheader("🎯 Configure Backward Elimination")

        col1, col2 = st.columns(2)

        with col1:
            dependent_var = st.selectbox("Select Dependent Variable (Y):", numeric_columns, key='be_y')

        with col2:
            significance_level = st.slider("Significance Level (α):", 0.01, 0.10, 0.05, 0.01)

        independent_vars = st.multiselect(
            "Select Initial Independent Variables (X):",
            [col for col in numeric_columns if col != dependent_var],
            default=[col for col in numeric_columns if col != dependent_var][:5]
        )

        if dependent_var and independent_vars and len(independent_vars) > 0:
            if st.button("🔄 Perform Backward Elimination"):
                with st.spinner("Running backward elimination..."):
                    try:
                        # Perform backward elimination
                        final_model, features_kept, elimination_history = perform_backward_elimination(
                            df_clean, independent_vars, dependent_var, significance_level
                        )

                        # Display results
                        st.success("✅ Backward Elimination Complete!")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Initial Features", len(independent_vars))
                            st.metric("Features Removed", len(elimination_history))

                        with col2:
                            st.metric("Features Retained", len(features_kept))
                            st.metric("Significance Level", f"{significance_level}")

                        # Show elimination history
                        if elimination_history:
                            st.subheader("📋 Elimination History")
                            elim_df = pd.DataFrame(elimination_history)
                            elim_df['Step'] = range(1, len(elim_df) + 1)
                            elim_df = elim_df[['Step', 'feature', 'p_value']]
                            st.dataframe(elim_df, use_container_width=True)
                        else:
                            st.info("No features were eliminated. All features were significant.")

                        # Show final features
                        st.subheader("✅ Final Selected Features")
                        if features_kept:
                            st.write(", ".join(features_kept))
                        else:
                            st.warning("No features remained significant.")

                        # Model Summary
                        st.subheader("📊 Final Model Summary")

                        # Extract key metrics
                        st.write(f"**R-squared:** {final_model.rsquared:.4f}")
                        st.write(f"**Adjusted R-squared:** {final_model.rsquared_adj:.4f}")
                        st.write(f"**F-statistic:** {final_model.fvalue:.4f}")
                        st.write(f"**Prob (F-statistic):** {final_model.f_pvalue:.4e}")
                        st.write(f"**AIC:** {final_model.aic:.2f}")
                        st.write(f"**BIC:** {final_model.bic:.2f}")

                        # Coefficients table
                        st.subheader("📈 Final Model Coefficients")
                        coef_summary = pd.DataFrame({
                            'Feature': final_model.params.index,
                            'Coefficient': final_model.params.values,
                            'Std Error': final_model.bse.values,
                            't-value': final_model.tvalues.values,
                            'P-value': final_model.pvalues.values
                        })
                        st.dataframe(coef_summary, use_container_width=True)

                        # Diagnostic Plots
                        st.subheader("📊 Diagnostic Plots")

                        # Prepare data for plotting
                        if features_kept:
                            X = df_clean[features_kept].fillna(df_clean[features_kept].median())
                            y = df_clean[dependent_var].fillna(df_clean[dependent_var].median())
                            X_with_const = sm.add_constant(X)

                            # Get predictions and residuals
                            fitted_values = final_model.fittedvalues
                            residuals = final_model.resid

                            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                            # 1. Correlation Heatmap
                            corr_vars = features_kept + [dependent_var]
                            corr_matrix = df_clean[corr_vars].corr()
                            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                                       ax=axes[0, 0], square=True, linewidths=1)
                            axes[0, 0].set_title('Correlation Matrix of Selected Variables')

                            # 2. Residuals vs Fitted
                            axes[0, 1].scatter(fitted_values, residuals, alpha=0.5)
                            axes[0, 1].axhline(y=0, color='r', linestyle='--')
                            axes[0, 1].set_xlabel('Fitted Values')
                            axes[0, 1].set_ylabel('Residuals')
                            axes[0, 1].set_title('Residuals vs. Fitted Values')

                            # Add lowess line
                            try:
                                from statsmodels.nonparametric.smoothers_lowess import lowess
                                lowess_result = lowess(residuals, fitted_values, frac=0.2)
                                axes[0, 1].plot(lowess_result[:, 0], lowess_result[:, 1], 
                                              color='red', linewidth=2)
                            except:
                                pass

                            # 3. Q-Q Plot
                            sm.qqplot(residuals, line='s', ax=axes[1, 0])
                            axes[1, 0].set_title('Q-Q Plot of Residuals')

                            # 4. Scale-Location Plot
                            standardized_residuals = residuals / np.std(residuals)
                            axes[1, 1].scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
                            axes[1, 1].set_xlabel('Fitted Values')
                            axes[1, 1].set_ylabel('√|Standardized Residuals|')
                            axes[1, 1].set_title('Scale-Location Plot')

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        # Full model summary (expandable)
                        with st.expander("📄 View Full Statistical Summary"):
                            st.text(str(final_model.summary()))

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.write("Please ensure your data has no missing values and features are numeric.")
        else:
            st.info("👆 Please select dependent variable and at least one independent variable.")

    # ============================================================================
    # DECISION TREE
    # ============================================================================
    elif choice == "🌳 Decision Tree":
        st.header("🌳 Decision Tree Classification")

        df_processed, le_district, le_month = preprocess_for_decision_tree(df)

        st.subheader("🎯 Select Target Variable")

        target_options = []
        if 'High_Productivity' in df_processed.columns:
            target_options.append('High_Productivity')
        if 'High_Admin_Spending' in df_processed.columns:
            target_options.append('High_Admin_Spending')

        if not target_options:
            st.error("No suitable target variables found. Please check your data.")
            return

        target = st.selectbox("Select Target Variable:", target_options)

        # Feature selection
        numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [f for f in numeric_features if f != target and 'Productivity' not in f and 'Admin_Cost_Ratio' not in f]

        selected_features = st.multiselect(
            "Select Features for Classification:",
            numeric_features,
            default=numeric_features[:5]
        )

        if selected_features and st.button("🚀 Train Decision Tree"):
            # Prepare data
            X = df_processed[selected_features].fillna(df_processed[selected_features].median())
            y = df_processed[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            clf = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=20)
            clf.fit(X_train, y_train)

            # Predictions
            y_pred = clf.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)

            st.subheader("📊 Model Performance")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("Training Samples", len(X_train))
            with col3:
                st.metric("Test Samples", len(X_test))

            # Classification Report
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

            # Confusion Matrix
            st.subheader("🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            plt.close()

            # Feature Importance
            st.subheader("🎯 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'], color='forestgreen')
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Decision Tree Visualization
            st.subheader("🌳 Decision Tree Visualization")
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(clf, feature_names=selected_features, class_names=['Low', 'High'],
                     filled=True, rounded=True, ax=ax)
            st.pyplot(fig)
            plt.close()

    # ============================================================================
    # CLUSTERING ANALYSIS
    # ============================================================================
    elif choice == "🎪 Clustering Analysis":
        st.header("🎪 K-Means Clustering Analysis")

        numeric_data, scaled_data = preprocess_for_clustering(df)

        st.subheader("⚙️ Configure Clustering")

        col1, col2 = st.columns(2)

        with col1:
            n_clusters = st.slider("Number of Clusters:", 2, 10, 3)

        with col2:
            random_state = st.number_input("Random State:", 0, 100, 42)

        if st.button("🚀 Perform Clustering"):
            # Perform K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)

            # Add clusters to dataframe
            df_clustered = numeric_data.copy()
            df_clustered['Cluster'] = clusters

            # Display cluster statistics
            st.subheader("📊 Cluster Statistics")

            cluster_stats = df_clustered.groupby('Cluster').agg(['mean', 'count'])
            st.dataframe(cluster_stats, use_container_width=True)

            # Cluster sizes
            st.subheader("📈 Cluster Distribution")
            cluster_counts = pd.DataFrame(df_clustered['Cluster'].value_counts().sort_index())
            cluster_counts.columns = ['Count']

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(cluster_counts.index, cluster_counts['Count'], color='skyblue')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of Records')
            ax.set_title('Distribution of Records Across Clusters')
            ax.set_xticks(range(n_clusters))
            st.pyplot(fig)
            plt.close()

            # 2D Visualization using first 2 principal components
            st.subheader("🎨 Cluster Visualization (PCA)")

            pca_2d = PCA(n_components=2, random_state=random_state)
            coords_2d = pca_2d.fit_transform(scaled_data)

            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=clusters, 
                               cmap='viridis', alpha=0.6, s=50)
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                      c='red', marker='X', s=200, edgecolors='black', label='Centroids')
            ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_title('K-Means Clustering Visualization (2D PCA)')
            plt.colorbar(scatter, label='Cluster')
            ax.legend()
            st.pyplot(fig)
            plt.close()

            # Elbow Method
            st.subheader("📉 Elbow Method for Optimal Clusters")

            inertias = []
            K_range = range(2, 11)

            for k in K_range:
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                km.fit(scaled_data)
                inertias.append(km.inertia_)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(K_range, inertias, 'bo-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method For Optimal k')
            ax.grid(True)
            st.pyplot(fig)
            plt.close()

    # ============================================================================
    # PCA ANALYSIS
    # ============================================================================
    elif choice == "📊 PCA Analysis":
        st.header("📊 Principal Component Analysis (PCA)")

        numeric_data, scaled_data = preprocess_for_clustering(df)

        st.subheader("⚙️ Configure PCA")

        n_components = st.slider("Number of Components:", 2, min(10, scaled_data.shape[1]), 3)

        if st.button("🚀 Perform PCA"):
            # Perform PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(scaled_data)

            # Variance explained
            st.subheader("📈 Variance Explained")

            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_components)],
                'Variance Explained': pca.explained_variance_ratio_,
                'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
            })

            st.dataframe(variance_df, use_container_width=True)

            # Scree Plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Individual variance
            axes[0].bar(range(1, n_components+1), pca.explained_variance_ratio_, color='steelblue')
            axes[0].set_xlabel('Principal Component')
            axes[0].set_ylabel('Variance Explained Ratio')
            axes[0].set_title('Scree Plot')
            axes[0].set_xticks(range(1, n_components+1))

            # Cumulative variance
            axes[1].plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), 
                        'bo-', linewidth=2)
            axes[1].set_xlabel('Number of Components')
            axes[1].set_ylabel('Cumulative Variance Explained')
            axes[1].set_title('Cumulative Variance Explained')
            axes[1].grid(True)
            axes[1].set_xticks(range(1, n_components+1))

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Component Loadings
            st.subheader("🎯 Component Loadings")

            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=numeric_data.columns
            )

            st.dataframe(loadings.style.background_gradient(cmap='coolwarm', axis=None), 
                        use_container_width=True)

            # Biplot (for first 2 components)
            if n_components >= 2:
                st.subheader("📊 PCA Biplot (PC1 vs PC2)")

                fig, ax = plt.subplots(figsize=(12, 8))

                # Plot data points
                ax.scatter(components[:, 0], components[:, 1], alpha=0.5, s=30)

                # Plot loading vectors
                for i, feature in enumerate(numeric_data.columns):
                    ax.arrow(0, 0, 
                            pca.components_[0, i]*3, 
                            pca.components_[1, i]*3,
                            head_width=0.1, head_length=0.1, 
                            fc='red', ec='red', alpha=0.7)
                    ax.text(pca.components_[0, i]*3.2, 
                           pca.components_[1, i]*3.2,
                           feature, fontsize=8, ha='center')

                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                ax.set_title('PCA Biplot')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linewidth=0.5)
                ax.axvline(x=0, color='k', linewidth=0.5)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

if __name__ == "__main__":
    main()
