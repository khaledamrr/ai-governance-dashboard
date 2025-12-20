"""
Main script to run the complete analysis pipeline
"""

import os
import sys
from pyspark.sql import SparkSession
from src.data_processor import COMPASDataProcessor
from src.ml_model import RecidivismPredictor
from src.bias_analyzer import BiasAnalyzer

def main():
    """Run complete analysis pipeline"""
    
    print("=" * 60)
    print("AI Governance Dashboard - Analysis Pipeline")
    print("=" * 60)
    
    print("\n[1/5] Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("AI_Governance_Analysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    print("\n[2/5] Loading and processing data...")
    data_file = "compas-scores-two-years.csv"
    if not os.path.exists(data_file):
        data_file = "data/compas-scores-two-years.csv"
    
    processor = COMPASDataProcessor(spark)
    processor.load_data(data_file)
    processor.clean_data()
    processor.prepare_features()
    
    stats = processor.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Total Records: {stats.get('total_records', 0):,}")
    print(f"  Recidivism Rate: {stats.get('recidivism_rate', 0):.2%}")
    
    print("\n[3/5] Splitting data into train/test sets...")
    train_df, test_df = processor.processed_df.randomSplit([0.8, 0.2], seed=42)
    print(f"  Training set: {train_df.count()} records")
    print(f"  Test set: {test_df.count()} records")
    
    print("\n[4/5] Training Random Forest model...")
    predictor = RecidivismPredictor(spark)
    predictor.train_random_forest(train_df)
    
    print("\n[5/5] Making predictions and evaluating...")
    predictions = predictor.predict(test_df)
    
    metrics = predictor.evaluate_model(predictions)
    
    print("\nCalculating fairness metrics...")
    fairness_metrics = predictor.calculate_fairness_metrics(predictions)
    
    print("\nPerforming bias analysis...")
    analyzer = BiasAnalyzer(spark)
    bias_report = analyzer.comprehensive_bias_report(predictions)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\nModel Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    print("\nFairness Metrics by Race:")
    for group, group_metrics in fairness_metrics.items():
        if group != 'disparity' and isinstance(group_metrics, dict):
            print(f"\n  {group}:")
            print(f"    TPR: {group_metrics.get('true_positive_rate', 0):.4f}")
            print(f"    FPR: {group_metrics.get('false_positive_rate', 0):.4f}")
            print(f"    Accuracy: {group_metrics.get('accuracy', 0):.4f}")
    
    if 'disparity' in fairness_metrics:
        print("\nDisparity Metrics:")
        disp = fairness_metrics['disparity']
        print(f"  Equalized Odds Disparity: {disp.get('equalized_odds_disparity', 0):.4f}")
        print(f"  Demographic Parity Disparity: {disp.get('demographic_parity_disparity', 0):.4f}")
        print(f"  Equal Opportunity Disparity: {disp.get('equal_opportunity_disparity', 0):.4f}")
    
    print("\nSaving results...")
    os.makedirs("results", exist_ok=True)
    
    model_path = "models/recidivism_model"
    os.makedirs("models", exist_ok=True)
    predictor.save_model(model_path)
    
    predictions.select('id', 'two_year_recid', 'prediction', 'probability', 'race_binary').write \
        .mode('overwrite') \
        .option("header", "true") \
        .csv("results/predictions")
    
    print("\nAnalysis complete! Results saved to 'results/' directory")
    print("=" * 60)
    
    spark.stop()

if __name__ == "__main__":
    main()




