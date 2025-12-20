"""
AI Governance Dashboard
Streamlit application for monitoring bias and ethical violations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import COMPASDataProcessor
from src.ml_model import RecidivismPredictor
from src.bias_analyzer import BiasAnalyzer
from pyspark.sql import SparkSession

# Page configuration
st.set_page_config(
    page_title="AI Governance Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Spark Session
@st.cache_resource
def init_spark():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("AI_Governance_Dashboard") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    return spark

# Load and process data
@st.cache_resource
def load_and_process_data(file_path):
    """Load and process COMPAS data"""
    spark = init_spark()
    processor = COMPASDataProcessor(spark)
    processor.load_data(file_path)
    processor.clean_data()
    processor.prepare_features()
    return processor

# Train model
@st.cache_resource
def train_model(_processor, model_type='random_forest'):
    """Train ML model"""
    spark = init_spark()
    
    # Split data
    train_df, test_df = _processor.processed_df.randomSplit([0.8, 0.2], seed=42)
    
    # Train model
    predictor = RecidivismPredictor(spark)
    if model_type == 'random_forest':
        predictor.train_random_forest(train_df)
    else:
        predictor.train_logistic_regression(train_df)
    
    # Make predictions
    predictions = predictor.predict(test_df)
    
    # Evaluate
    metrics = predictor.evaluate_model(predictions)
    
    # Calculate fairness
    fairness_metrics = predictor.calculate_fairness_metrics(predictions)
    
    return predictor, predictions, metrics, fairness_metrics, test_df

def main():
    """Main dashboard application"""
    
    # Title
    st.title("⚖️ AI Governance Dashboard: Monitoring Bias & Ethical Violations")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📊 Overview", "🔍 Data Analysis", "🤖 Model Performance", "⚖️ Bias Analysis", "📈 Fairness Metrics"]
    )
    
    # File path - check multiple locations
    data_file = None
    possible_paths = [
        "compas-scores-two-years.csv",
        "data/compas-scores-two-years.csv",
        "/app/data/compas-scores-two-years.csv",
        "/app/compas-scores-two-years.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if data_file is None:
        st.error("❌ Data file not found! Please ensure 'compas-scores-two-years.csv' is in the project root.")
        st.info("Current directory: " + os.getcwd())
        st.info("Looking for file in: " + str(possible_paths))
        st.stop()
    
    # Load data
    with st.spinner("Loading and processing data..."):
        try:
            processor = load_and_process_data(data_file)
            stats = processor.get_statistics()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()
    
    # Page routing
    if page == "📊 Overview":
        show_overview(processor, stats)
    elif page == "🔍 Data Analysis":
        show_data_analysis(processor)
    elif page == "🤖 Model Performance":
        show_model_performance(processor)
    elif page == "⚖️ Bias Analysis":
        show_bias_analysis(processor)
    elif page == "📈 Fairness Metrics":
        show_fairness_metrics(processor)

def show_overview(processor, stats):
    """Show overview page"""
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{stats.get('total_records', 0):,}")
    
    with col2:
        recid_rate = stats.get('recidivism_rate', 0)
        st.metric("Recidivism Rate", f"{recid_rate:.2%}")
    
    with col3:
        race_dist = stats.get('race_distribution', {})
        total_race = sum(race_dist.values())
        if total_race > 0:
            aa_pct = (race_dist.get('African-American', 0) / total_race) * 100
            st.metric("African-American %", f"{aa_pct:.1f}%")
    
    with col4:
        gender_dist = stats.get('gender_distribution', {})
        total_gender = sum(gender_dist.values())
        if total_gender > 0:
            male_pct = (gender_dist.get('Male', 0) / total_gender) * 100
            st.metric("Male %", f"{male_pct:.1f}%")
    
    st.markdown("---")
    
    # Race distribution chart
    st.subheader("Race Distribution")
    race_dist = stats.get('race_distribution', {})
    if race_dist:
        race_df = pd.DataFrame(list(race_dist.items()), columns=['Race', 'Count'])
        fig = px.pie(race_df, values='Count', names='Race', 
                    title="Distribution by Race")
        st.plotly_chart(fig, use_container_width=True)
    
    # Gender distribution chart
    st.subheader("Gender Distribution")
    gender_dist = stats.get('gender_distribution', {})
    if gender_dist:
        gender_df = pd.DataFrame(list(gender_dist.items()), columns=['Gender', 'Count'])
        fig = px.bar(gender_df, x='Gender', y='Count', 
                    title="Distribution by Gender")
        st.plotly_chart(fig, use_container_width=True)

def show_data_analysis(processor):
    """Show data analysis page"""
    st.header("🔍 Data Analysis")
    
    # Convert to pandas for display - only select columns that exist
    display_columns = ['age', 'race_binary', 'sex', 'decile_score', 'two_year_recid']
    # Add priors_count only if it exists
    if 'priors_count' in processor.processed_df.columns:
        display_columns.insert(3, 'priors_count')
    # Filter to only existing columns
    available_columns = [col for col in display_columns if col in processor.processed_df.columns]
    
    pdf = processor.processed_df.select(*available_columns).limit(1000).toPandas()
    
    st.subheader("Sample Data")
    st.dataframe(pdf.head(100), use_container_width=True)
    
    st.markdown("---")
    
    # Age distribution
    st.subheader("Age Distribution")
    age_data = processor.processed_df.select('age').toPandas()
    fig = px.histogram(age_data, x='age', nbins=30, title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recidivism by race
    st.subheader("Recidivism Rate by Race")
    recid_by_race = processor.processed_df.groupBy('race_binary', 'two_year_recid').count().toPandas()
    if not recid_by_race.empty:
        pivot_df = recid_by_race.pivot(index='race_binary', columns='two_year_recid', values='count').fillna(0)
        if 1 in pivot_df.columns and 0 in pivot_df.columns:
            pivot_df['recidivism_rate'] = pivot_df[1] / (pivot_df[0] + pivot_df[1])
            fig = px.bar(pivot_df.reset_index(), x='race_binary', y='recidivism_rate',
                        title="Recidivism Rate by Race")
            st.plotly_chart(fig, use_container_width=True)

def show_model_performance(processor):
    """Show model performance page"""
    st.header("🤖 Model Performance")
    
    model_type = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                predictor, predictions, metrics, fairness_metrics, test_df = train_model(
                    processor, 
                    model_type.lower().replace(' ', '_')
                )
                
                st.success("Model trained successfully!")
                
                # Display metrics
                st.subheader("Model Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                with col5:
                    st.metric("AUC", f"{metrics['auc']:.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = predictor.get_confusion_matrix(predictions)
                cm_df = pd.DataFrame(cm, 
                                    index=['Actual: No Recidivism', 'Actual: Recidivism'],
                                    columns=['Predicted: No Recidivism', 'Predicted: Recidivism'])
                
                fig = px.imshow(cm_df, text_auto=True, aspect="auto",
                               title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                # Store in session state
                st.session_state['predictor'] = predictor
                st.session_state['predictions'] = predictions
                st.session_state['metrics'] = metrics
                st.session_state['fairness_metrics'] = fairness_metrics
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                st.exception(e)
    
    # Show stored results
    if 'metrics' in st.session_state:
        st.subheader("Model Metrics")
        metrics = st.session_state['metrics']
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
        with col5:
            st.metric("AUC", f"{metrics['auc']:.4f}")

def show_bias_analysis(processor):
    """Show bias analysis page"""
    st.header("⚖️ Bias Analysis")
    
    if 'predictions' not in st.session_state:
        st.warning("Please train a model first in the 'Model Performance' page.")
        return
    
    predictions = st.session_state['predictions']
    fairness_metrics = st.session_state.get('fairness_metrics', {})
    
    protected_attribute = st.selectbox("Protected Attribute", ["race_binary", "sex"])
    
    if st.button("Analyze Bias"):
        with st.spinner("Analyzing bias..."):
            try:
                spark = init_spark()
                analyzer = BiasAnalyzer(spark)
                
                bias_report = analyzer.comprehensive_bias_report(
                    predictions, 
                    protected_attribute=protected_attribute
                )
                
                st.success("Bias analysis completed!")
                
                # Display results
                st.subheader("Demographic Parity")
                dp = bias_report.get('demographic_parity', {})
                if 'disparity' in dp:
                    st.metric("Disparity", f"{dp['disparity']:.4f}")
                
                for group, metrics in dp.items():
                    if group != 'disparity':
                        st.write(f"**{group}**: Prediction Rate = {metrics.get('prediction_rate', 0):.4f}")
                
                st.subheader("Equalized Odds")
                eo = bias_report.get('equalized_odds', {})
                if 'disparity' in eo:
                    st.write(f"**TPR Disparity**: {eo['disparity'].get('tpr_disparity', 0):.4f}")
                    st.write(f"**FPR Disparity**: {eo['disparity'].get('fpr_disparity', 0):.4f}")
                
                for group, metrics in eo.items():
                    if group != 'disparity':
                        st.write(f"**{group}**: TPR = {metrics.get('true_positive_rate', 0):.4f}, "
                               f"FPR = {metrics.get('false_positive_rate', 0):.4f}")
                
                st.subheader("Equal Opportunity")
                eopp = bias_report.get('equal_opportunity', {})
                if 'disparity' in eopp:
                    st.metric("Disparity", f"{eopp['disparity']:.4f}")
                
                for group, metrics in eopp.items():
                    if group != 'disparity':
                        st.write(f"**{group}**: TPR = {metrics.get('true_positive_rate', 0):.4f}")
                
            except Exception as e:
                st.error(f"Error analyzing bias: {str(e)}")
                st.exception(e)

def show_fairness_metrics(processor):
    """Show fairness metrics page"""
    st.header("📈 Fairness Metrics")
    
    if 'fairness_metrics' not in st.session_state:
        st.warning("Please train a model first in the 'Model Performance' page.")
        return
    
    fairness_metrics = st.session_state['fairness_metrics']
    
    st.subheader("Fairness Metrics by Race")
    
    # Create visualization
    metrics_df = []
    for group, metrics in fairness_metrics.items():
        if group != 'disparity' and isinstance(metrics, dict):
            metrics_df.append({
                'Group': group,
                'TPR': metrics.get('true_positive_rate', 0),
                'FPR': metrics.get('false_positive_rate', 0),
                'PPV': metrics.get('positive_predictive_value', 0),
                'Accuracy': metrics.get('accuracy', 0)
            })
    
    if metrics_df:
        df = pd.DataFrame(metrics_df)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('True Positive Rate', 'False Positive Rate', 
                          'Positive Predictive Value', 'Accuracy'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(x=df['Group'], y=df['TPR'], name='TPR'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['Group'], y=df['FPR'], name='FPR'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=df['Group'], y=df['PPV'], name='PPV'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['Group'], y=df['Accuracy'], name='Accuracy'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Fairness Metrics Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display disparity metrics
        if 'disparity' in fairness_metrics:
            st.subheader("Disparity Metrics")
            disp = fairness_metrics['disparity']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Equalized Odds Disparity", 
                         f"{disp.get('equalized_odds_disparity', 0):.4f}")
            with col2:
                st.metric("Demographic Parity Disparity", 
                         f"{disp.get('demographic_parity_disparity', 0):.4f}")
            with col3:
                st.metric("Equal Opportunity Disparity", 
                         f"{disp.get('equal_opportunity_disparity', 0):.4f}")

if __name__ == "__main__":
    main()

