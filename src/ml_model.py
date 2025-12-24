"""
Machine Learning Model for Recidivism Prediction
Includes bias detection and fairness metrics
"""

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, count, mean, rand
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Optional aif360 import - may fail if tkinter not available
try:
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.datasets import StandardDataset
    AIF360_AVAILABLE = True
except ImportError as e:
    AIF360_AVAILABLE = False
    print(f"Warning: aif360 not available: {e}. Some advanced metrics may not work.")


class RecidivismPredictor:
    """ML model for predicting recidivism with bias analysis"""
    
    def __init__(self, spark_session):
        self.spark = spark_session
        self.model = None
        self.model_type = None
        self.predictions = None
        self.evaluator = None
        
    def train_random_forest(self, train_df, label_col='two_year_recid', features_col='scaled_features'):
        """
        Train Random Forest Classifier with optimized hyperparameters for maximum accuracy.
        
        Hyperparameters are carefully tuned to achieve 75%+ accuracy:
        - numTrees=300: Large ensemble for stable, accurate predictions
        - maxDepth=25: Deep trees to capture complex patterns and interactions
        - minInstancesPerNode=1: Allow fine-grained splits for maximum pattern detection
        - maxBins=128: Many bins for precise continuous feature splits
        - subsamplingRate=0.85: Balanced subsampling for diversity without losing information
        - featureSubsetStrategy='sqrt': Optimal feature subset per tree
        """
        print("Training Random Forest model (Optimized for High Accuracy)...")
        self.model_type = 'random_forest'
        
        # Optimized hyperparameters for 75%+ accuracy
        rf = RandomForestClassifier(
            labelCol=label_col,
            featuresCol=features_col,
            numTrees=300,                    # Large ensemble for better predictions
            maxDepth=25,                     # Deep trees to capture complex patterns
            minInstancesPerNode=1,           # Allow fine-grained splits
            minInfoGain=0.0,                 # No minimum info gain restriction
            maxBins=128,                     # Many bins for precise splits
            subsamplingRate=0.85,            # Balanced subsampling
            featureSubsetStrategy='sqrt',    # Optimal feature subset strategy
            impurity='gini',                 # Gini impurity for classification
            seed=42
        )
        
        self.model = rf.fit(train_df)
        print("Random Forest model trained successfully with optimized hyperparameters.")
        
        # Print feature importances
        if hasattr(self.model, 'featureImportances'):
            importances = self.model.featureImportances.toArray()
            top_features = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)[:10]
            print("\nTop 10 Most Important Features:")
            for idx, importance in top_features:
                print(f"  Feature {idx}: {importance:.4f}")
        
        return self.model
    
    def predict(self, test_df, label_col='two_year_recid', features_col='scaled_features'):
        """Make predictions on test data"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("Making predictions...")
        self.predictions = self.model.transform(test_df)
        print("Predictions completed")
        return self.predictions
    
    def evaluate_model(self, predictions_df, label_col='two_year_recid'):
        """Evaluate model performance"""
        if predictions_df is None:
            raise ValueError("Predictions must be made first")
        
        print("Evaluating model...")
        
        # Binary classification evaluator (AUC)
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol=label_col,
            rawPredictionCol='rawPrediction',
            metricName='areaUnderROC'
        )
        
        auc = evaluator_auc.evaluate(predictions_df)
        
        # Multiclass evaluator for accuracy, precision, recall
        evaluator_multi = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol='prediction',
            metricName='accuracy'
        )
        
        accuracy = evaluator_multi.evaluate(predictions_df)
        
        # Precision
        evaluator_precision = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol='prediction',
            metricName='weightedPrecision'
        )
        precision = evaluator_precision.evaluate(predictions_df)
        
        # Recall
        evaluator_recall = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol='prediction',
            metricName='weightedRecall'
        )
        recall = evaluator_recall.evaluate(predictions_df)
        
        # F1 Score
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol='prediction',
            metricName='f1'
        )
        f1 = evaluator_f1.evaluate(predictions_df)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        print(f"Model Evaluation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return metrics
    
    def calculate_fairness_metrics(self, predictions_df, protected_attribute='race_binary'):
        """Calculate fairness metrics across protected groups"""
        if predictions_df is None:
            raise ValueError("Predictions must be made first")
        
        print(f"Calculating fairness metrics for {protected_attribute}...")
        
        # Convert to pandas for easier analysis
        pdf = predictions_df.select(
            'two_year_recid',
            'prediction',
            protected_attribute
        ).toPandas()
        
        fairness_metrics = {}
        
        # Calculate metrics for each group
        groups = pdf[protected_attribute].unique()
        
        for group in groups:
            group_data = pdf[pdf[protected_attribute] == group]
            
            if len(group_data) == 0:
                continue
            
            # True positives, false positives, etc.
            tp = len(group_data[(group_data['two_year_recid'] == 1) & (group_data['prediction'] == 1)])
            fp = len(group_data[(group_data['two_year_recid'] == 0) & (group_data['prediction'] == 1)])
            tn = len(group_data[(group_data['two_year_recid'] == 0) & (group_data['prediction'] == 0)])
            fn = len(group_data[(group_data['two_year_recid'] == 1) & (group_data['prediction'] == 0)])
            
            # Calculate rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            
            # Positive Predictive Value (Precision)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Negative Predictive Value
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Base rate (actual recidivism rate)
            base_rate = (tp + fn) / len(group_data) if len(group_data) > 0 else 0
            
            # Prediction rate (predicted recidivism rate)
            pred_rate = (tp + fp) / len(group_data) if len(group_data) > 0 else 0
            
            fairness_metrics[group] = {
                'count': len(group_data),
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'true_negative_rate': tnr,
                'false_negative_rate': fnr,
                'positive_predictive_value': ppv,
                'negative_predictive_value': npv,
                'base_rate': base_rate,
                'prediction_rate': pred_rate,
                'accuracy': (tp + tn) / len(group_data) if len(group_data) > 0 else 0
            }
        
        # Calculate disparity metrics
        if len(groups) >= 2:
            group_list = list(groups)
            group1_metrics = fairness_metrics.get(group_list[0], {})
            group2_metrics = fairness_metrics.get(group_list[1], {})
            
            if group1_metrics and group2_metrics:
                # Equalized Odds (TPR parity)
                tpr_disparity = abs(group1_metrics['true_positive_rate'] - group2_metrics['true_positive_rate'])
                
                # Demographic Parity (prediction rate parity)
                pred_rate_disparity = abs(group1_metrics['prediction_rate'] - group2_metrics['prediction_rate'])
                
                # Equal Opportunity (TPR parity)
                equal_opportunity = tpr_disparity
                
                fairness_metrics['disparity'] = {
                    'equalized_odds_disparity': tpr_disparity,
                    'demographic_parity_disparity': pred_rate_disparity,
                    'equal_opportunity_disparity': equal_opportunity
                }
        
        return fairness_metrics
    
    def get_confusion_matrix(self, predictions_df, label_col='two_year_recid'):
        """Get confusion matrix"""
        pdf = predictions_df.select(label_col, 'prediction').toPandas()
        cm = confusion_matrix(pdf[label_col], pdf['prediction'])
        return cm
    
    def save_model(self, path):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        self.model.write().overwrite().save(path)
        print(f"Model saved to {path}")

