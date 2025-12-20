# AI Governance Dashboard: Monitoring Bias & Ethical Violations

A comprehensive Big Data project for analyzing bias and fairness in AI systems using the COMPAS recidivism prediction dataset, Apache Spark, and Docker.

## 📋 Project Overview

This project implements an AI Governance Dashboard that:
- Analyzes the COMPAS dataset for bias and fairness issues
- Trains machine learning models for recidivism prediction
- Calculates comprehensive fairness metrics (Demographic Parity, Equalized Odds, Equal Opportunity)
- Provides an interactive Streamlit dashboard for visualization
- Runs on Apache Spark for distributed processing
- Containerized with Docker for easy deployment

## 🛠️ Technologies Used

- **Apache Spark 3.5.0**: Distributed data processing
- **Python 3.9**: Programming language
- **PySpark**: Spark Python API
- **Streamlit**: Interactive web dashboard
- **scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive visualizations
- **Docker**: Containerization
- **AIF360**: Fairness metrics library

## 📁 Project Structure

```
ai-governance-dashboard/
├── compas-scores-two-years.csv    # Dataset
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose configuration
├── requirements.txt                # Python dependencies
├── dashboard.py                    # Streamlit dashboard
├── run_analysis.py                 # Main analysis script
├── src/
│   ├── __init__.py
│   ├── data_processor.py          # Data loading and preprocessing
│   ├── ml_model.py                 # ML model training and evaluation
│   └── bias_analyzer.py            # Bias analysis and fairness metrics
├── models/                         # Saved ML models
├── results/                        # Analysis results
└── phase 1/                        # Phase 1 documentation
```

## 🚀 Quick Start Guide

### Prerequisites

- Docker Desktop installed and running
- At least 8GB RAM available for Docker
- Windows 10/11 (or Linux/Mac)

### Step-by-Step Installation

#### Step 1: Verify Docker Installation

Open PowerShell and run:
```powershell
docker --version
docker-compose --version
```

If not installed, download Docker Desktop from: https://www.docker.com/products/docker-desktop

#### Step 2: Navigate to Project Directory

```powershell
cd "C:\Users\khale\OneDrive\Desktop\ai-governance-dashboard"
```

#### Step 3: Build Docker Image

```powershell
docker-compose build
```

This will:
- Download Python 3.9 base image
- Install Java 11
- Install Apache Spark 3.5.0
- Install all Python dependencies
- Set up the application

**Expected time: 5-10 minutes (first time)**

#### Step 4: Start the Application

```powershell
docker-compose up
```

This will:
- Start the container
- Launch the Streamlit dashboard
- Make it available at http://localhost:8501

#### Step 5: Access the Dashboard

Open your web browser and navigate to:
```
http://localhost:8501
```

### Running Analysis Script

To run the complete analysis pipeline:

```powershell
docker-compose exec ai-governance-dashboard python run_analysis.py
```

Or run locally (if you have Python and Spark installed):

```powershell
python run_analysis.py
```

## 📊 Using the Dashboard

### Navigation Pages

1. **📊 Overview**: Dataset statistics and distributions
2. **🔍 Data Analysis**: Exploratory data analysis
3. **🤖 Model Performance**: Train and evaluate ML models
4. **⚖️ Bias Analysis**: Comprehensive bias detection
5. **📈 Fairness Metrics**: Detailed fairness metrics visualization

### Workflow

1. Start with **Overview** to understand the dataset
2. Explore **Data Analysis** to see data distributions
3. Go to **Model Performance** and click "Train Model"
4. Navigate to **Bias Analysis** to see bias detection results
5. Check **Fairness Metrics** for detailed comparisons

## 🔧 Manual Setup (Without Docker)

If you prefer to run without Docker:

### 1. Install Python 3.9

Download from: https://www.python.org/downloads/

### 2. Install Java 11

Download from: https://adoptium.net/

Set JAVA_HOME environment variable:
```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-11"
```

### 3. Install Apache Spark

Download Spark 3.5.0 from: https://spark.apache.org/downloads.html

Extract and set SPARK_HOME:
```powershell
$env:SPARK_HOME = "C:\spark-3.5.0-bin-hadoop3"
$env:PATH += ";$env:SPARK_HOME\bin"
```

### 4. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the Application

```powershell
streamlit run dashboard.py
```

## 📈 Understanding the Results

### Model Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area Under ROC Curve

### Fairness Metrics

- **Demographic Parity**: Equal prediction rates across groups
- **Equalized Odds**: Equal TPR and FPR across groups
- **Equal Opportunity**: Equal TPR across groups
- **Disparity**: Difference in metrics between groups (lower is better)

### Bias Indicators

- **High Disparity (>0.1)**: Significant bias detected
- **Medium Disparity (0.05-0.1)**: Moderate bias
- **Low Disparity (<0.05)**: Relatively fair

## 🐛 Troubleshooting

### Docker Issues

**Problem**: Docker container fails to start
- **Solution**: Ensure Docker Desktop is running
- Check: `docker ps` should show running containers

**Problem**: Port 8501 already in use
- **Solution**: Change port in `docker-compose.yml`:
  ```yaml
  ports:
    - "8502:8501"  # Use port 8502 instead
  ```

**Problem**: Out of memory errors
- **Solution**: Increase Docker memory limit in Docker Desktop settings (Settings → Resources → Memory)

### Spark Issues

**Problem**: Java not found
- **Solution**: Ensure JAVA_HOME is set correctly in Dockerfile

**Problem**: Slow processing
- **Solution**: Increase Spark executor memory in `data_processor.py`:
  ```python
  .config("spark.executor.memory", "2g")
  ```

### Data Issues

**Problem**: File not found error
- **Solution**: Ensure `compas-scores-two-years.csv` is in the project root directory

## 📝 Project Documentation

See `Project_Documentation.md` for complete project documentation following the course requirements.

## 🎓 Course Requirements Checklist

- ✅ Dataset: COMPAS recidivism dataset
- ✅ Apache Spark: Distributed processing
- ✅ Docker: Containerization
- ✅ ML Model: Random Forest and Logistic Regression
- ✅ Evaluation Metrics: Accuracy, Precision, Recall, F1, AUC
- ✅ Dashboard: Streamlit interactive dashboard
- ✅ Bias Analysis: Comprehensive fairness metrics
- ✅ Documentation: Complete project documentation

## 📚 References

1. Baumeister, J., et al. (2025). Stream-Based Monitoring of Algorithmic Fairness. TACAS 2025.
2. Chen, P., Wu, L., & Wang, L. (2023). AI fairness in data management and analytics. Applied Sciences, 13(18), 10258.
3. Funda, V. (2025). A systematic review of algorithm auditing processes. Journal of Infrastructure, Policy and Development.

## 👥 Authors

- Khaled Amr (ID: 237857)
- Abraam Refaat (ID: 234459)
- Omar Samh (ID: 231969)

## 📄 License

This project is for educational purposes as part of the Big Data course.

## 🙏 Acknowledgments

- COMPAS dataset providers
- Apache Spark community
- Streamlit team
- AIF360 developers



