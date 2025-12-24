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
    
    print("\n" + "=" * 60)
    print("AI Governance Dashboard - Random Forest Analysis Pipeline")
    print("=" * 60)
    
    # Initialize Spark
    print("\n[1/5] Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("AI_Governance_Analysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Load and process data
    print("\n[2/5] Loading and processing data...")
    data_file = "compas-scores-two-years.csv"
    if not os.path.exists(data_file):
        data_file = "data/compas-scores-two-years.csv"
    
    processor = COMPASDataProcessor(spark)
    processor.load_data(data_file)
    processor.clean_data()
    processor.prepare_features()
    
    # Get statistics
    stats = processor.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Total Records: {stats.get('total_records', 0):,}")
    print(f"  Recidivism Rate: {stats.get('recidivism_rate', 0):.2%}")
    
    # Split data: 80% train, 20% test (balanced split for reliable evaluation)
    print("\n[3/5] Splitting data into train/test sets...")
    from pyspark.sql.functions import col
    
    positive = processor.processed_df.filter(col('two_year_recid') == 1)
    negative = processor.processed_df.filter(col('two_year_recid') == 0)
    
    pos_train, pos_test = positive.randomSplit([0.80, 0.20], seed=42)
    neg_train, neg_test = negative.randomSplit([0.80, 0.20], seed=42)
    
    train_df = pos_train.union(neg_train)
    test_df = pos_test.union(neg_test)
    
    print(f"  Training set: {train_df.count()} records")
    print(f"  Test set: {test_df.count()} records")
    
    # Train model
    print("\n[4/5] Training Random Forest model...")
    predictor = RecidivismPredictor(spark)
    predictor.train_random_forest(train_df)
    
    # Make predictions
    print("\n[5/5] Making predictions and evaluating...")
    predictions = predictor.predict(test_df)
    
    # Evaluate model
    metrics = predictor.evaluate_model(predictions)
    
    # Calculate fairness metrics
    print("\nCalculating fairness metrics...")
    fairness_metrics = predictor.calculate_fairness_metrics(predictions)
    
    # Bias analysis
    print("\nPerforming bias analysis...")
    analyzer = BiasAnalyzer(spark)
    bias_report = analyzer.comprehensive_bias_report(predictions)
    
    # Print results
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
    
    # Save results
    print("\nSaving results...")
    os.makedirs("results", exist_ok=True)
    
    # Save model
    model_path = "models/recidivism_model"
    os.makedirs("models", exist_ok=True)
    predictor.save_model(model_path)
    
    # Save predictions
    # Convert vector columns to string for CSV saving
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    
    vec_to_str = udf(lambda x: str(x.tolist()) if hasattr(x, "tolist") else str(x), StringType())
    
    predictions.select('id', 'two_year_recid', 'prediction', 'race_binary') \
        .withColumn('probability_str', vec_to_str('probability')) \
        .write \
        .mode('overwrite') \
        .option("header", "true") \
        .csv("results/predictions")
    
    print("\nAnalysis complete! Results saved to 'results/' directory")
    print("=" * 60)
    
    spark.stop()

if __name__ == "__main__":
    main()





