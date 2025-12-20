"""
Bias Analysis Module
Implements comprehensive bias detection and fairness metrics
"""

from pyspark.sql.functions import col, when, count, mean, stddev, sum as spark_sum
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np


class BiasAnalyzer:
    """Analyze bias and fairness in predictions"""
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def demographic_parity(self, predictions_df, protected_attribute='race_binary', prediction_col='prediction'):
        """Calculate Demographic Parity (Statistical Parity)"""
        print(f"Calculating Demographic Parity for {protected_attribute}...")
        
        # Calculate prediction rate for each group
        parity_df = predictions_df.groupBy(protected_attribute).agg(
            mean(prediction_col).alias('prediction_rate'),
            count('*').alias('count')
        ).collect()
        
        parity_metrics = {}
        for row in parity_df:
            parity_metrics[row[protected_attribute]] = {
                'prediction_rate': row['prediction_rate'],
                'count': row['count']
            }
        
        # Calculate disparity
        if len(parity_metrics) >= 2:
            rates = [v['prediction_rate'] for v in parity_metrics.values()]
            disparity = max(rates) - min(rates)
            parity_metrics['disparity'] = disparity
        
        return parity_metrics
    
    def equalized_odds(self, predictions_df, protected_attribute='race_binary', 
                      label_col='two_year_recid', prediction_col='prediction'):
        """Calculate Equalized Odds (TPR and FPR parity)"""
        print(f"Calculating Equalized Odds for {protected_attribute}...")
        
        pdf = predictions_df.select(
            protected_attribute,
            label_col,
            prediction_col
        ).toPandas()
        
        odds_metrics = {}
        groups = pdf[protected_attribute].unique()
        
        for group in groups:
            group_data = pdf[pdf[protected_attribute] == group]
            
            # True Positive Rate (TPR)
            tp = len(group_data[(group_data[label_col] == 1) & (group_data[prediction_col] == 1)])
            fn = len(group_data[(group_data[label_col] == 1) & (group_data[prediction_col] == 0)])
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # False Positive Rate (FPR)
            fp = len(group_data[(group_data[label_col] == 0) & (group_data[prediction_col] == 1)])
            tn = len(group_data[(group_data[label_col] == 0) & (group_data[prediction_col] == 0)])
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            odds_metrics[group] = {
                'true_positive_rate': tpr,
                'false_positive_rate': fpr
            }
        
        # Calculate disparities
        if len(odds_metrics) >= 2:
            tprs = [v['true_positive_rate'] for v in odds_metrics.values()]
            fprs = [v['false_positive_rate'] for v in odds_metrics.values()]
            
            odds_metrics['disparity'] = {
                'tpr_disparity': max(tprs) - min(tprs),
                'fpr_disparity': max(fprs) - min(fprs)
            }
        
        return odds_metrics
    
    def equal_opportunity(self, predictions_df, protected_attribute='race_binary',
                         label_col='two_year_recid', prediction_col='prediction'):
        """Calculate Equal Opportunity (TPR parity)"""
        print(f"Calculating Equal Opportunity for {protected_attribute}...")
        
        pdf = predictions_df.select(
            protected_attribute,
            label_col,
            prediction_col
        ).toPandas()
        
        opportunity_metrics = {}
        groups = pdf[protected_attribute].unique()
        
        for group in groups:
            group_data = pdf[pdf[protected_attribute] == group]
            positive_group = group_data[group_data[label_col] == 1]
            
            if len(positive_group) > 0:
                tpr = len(positive_group[positive_group[prediction_col] == 1]) / len(positive_group)
            else:
                tpr = 0
            
            opportunity_metrics[group] = {
                'true_positive_rate': tpr,
                'positive_count': len(positive_group)
            }
        
        # Calculate disparity
        if len(opportunity_metrics) >= 2:
            tprs = [v['true_positive_rate'] for v in opportunity_metrics.values() if v['positive_count'] > 0]
            if tprs:
                opportunity_metrics['disparity'] = max(tprs) - min(tprs)
        
        return opportunity_metrics
    
    def calibration_analysis(self, predictions_df, protected_attribute='race_binary',
                            label_col='two_year_recid', probability_col='probability'):
        """Analyze calibration across groups"""
        print(f"Analyzing calibration for {protected_attribute}...")
        
        pdf = predictions_df.select(
            protected_attribute,
            label_col,
            probability_col
        ).toPandas()
        
        # Extract probability of positive class
        if isinstance(pdf[probability_col].iloc[0], (list, np.ndarray)):
            pdf['prob_positive'] = pdf[probability_col].apply(lambda x: float(x[1]) if len(x) > 1 else float(x[0]))
        else:
            pdf['prob_positive'] = pdf[probability_col]
        
        calibration_metrics = {}
        groups = pdf[protected_attribute].unique()
        
        for group in groups:
            group_data = pdf[pdf[protected_attribute] == group]
            
            # Bin probabilities
            bins = np.linspace(0, 1, 11)
            group_data['prob_bin'] = pd.cut(group_data['prob_positive'], bins=bins)
            
            calibration_data = []
            for bin_range in group_data['prob_bin'].unique():
                if pd.isna(bin_range):
                    continue
                bin_data = group_data[group_data['prob_bin'] == bin_range]
                if len(bin_data) > 0:
                    avg_predicted = bin_data['prob_positive'].mean()
                    avg_actual = bin_data[label_col].mean()
                    calibration_data.append({
                        'predicted': avg_predicted,
                        'actual': avg_actual,
                        'count': len(bin_data)
                    })
            
            calibration_metrics[group] = {
                'calibration_data': calibration_data,
                'mean_predicted': group_data['prob_positive'].mean(),
                'mean_actual': group_data[label_col].mean()
            }
        
        return calibration_metrics
    
    def comprehensive_bias_report(self, predictions_df, protected_attribute='race_binary',
                                 label_col='two_year_recid', prediction_col='prediction'):
        """Generate comprehensive bias analysis report"""
        print("Generating comprehensive bias report...")
        
        report = {
            'demographic_parity': self.demographic_parity(predictions_df, protected_attribute, prediction_col),
            'equalized_odds': self.equalized_odds(predictions_df, protected_attribute, label_col, prediction_col),
            'equal_opportunity': self.equal_opportunity(predictions_df, protected_attribute, label_col, prediction_col)
        }
        
        # Try calibration if probability column exists
        try:
            if 'probability' in predictions_df.columns:
                report['calibration'] = self.calibration_analysis(
                    predictions_df, protected_attribute, label_col, 'probability'
                )
        except Exception as e:
            print(f"Calibration analysis skipped: {e}")
        
        return report



