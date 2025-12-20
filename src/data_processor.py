"""
Data Processing Module for COMPAS Dataset
Handles data loading, cleaning, and preprocessing using Apache Spark
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, count, mean, stddev
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import pandas as pd


class COMPASDataProcessor:
    """Process COMPAS dataset for bias analysis and ML modeling"""
    
    def __init__(self, spark_session=None):
        """Initialize Spark session and data processor"""
        if spark_session is None:
            self.spark = SparkSession.builder \
                .appName("AI_Governance_Dashboard") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
        else:
            self.spark = spark_session
        
        self.df = None
        self.processed_df = None
        
    def load_data(self, file_path):
        """Load COMPAS dataset from CSV file"""
        print(f"Loading data from {file_path}...")
        # Read CSV - Spark automatically renames duplicate columns (e.g., priors_count becomes priors_count14)
        df_raw = self.spark.read.csv(
            file_path,
            header=True,
            inferSchema=True,
            nullValue="",
            nanValue=""
        )
        
        # Spark renames duplicate columns by appending numbers
        # We need to find the first occurrence of priors_count (the one without a number suffix)
        # and rename any numbered versions back to the original name if needed
        column_mapping = {}
        priors_count_found = False
        
        for col_name in df_raw.columns:
            # If this is a numbered duplicate of priors_count (e.g., priors_count14), skip it
            # We'll use the first priors_count (without number) if it exists
            if col_name.startswith('priors_count') and col_name != 'priors_count':
                # This is a duplicate, we'll skip it and use the first one
                continue
            # Keep all other columns as-is
            column_mapping[col_name] = col_name
        
        # Select only the columns we want to keep
        columns_to_select = list(column_mapping.keys())
        self.df = df_raw.select(*columns_to_select)
        
        print(f"Data loaded successfully. Rows: {self.df.count()}, Columns: {len(self.df.columns)}")
        # Check if priors_count exists
        if 'priors_count' in self.df.columns:
            print("priors_count column found")
        else:
            print("WARNING: priors_count column not found. Available columns with 'priors':", 
                  [c for c in self.df.columns if 'priors' in c.lower()])
        return self.df
    
    def clean_data(self):
        """Clean and preprocess the COMPAS dataset"""
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data() before clean_data()")
        
        print("Cleaning data...")
        
        # Select relevant columns for analysis
        relevant_columns = [
            'id', 'sex', 'age', 'age_cat', 'race', 'juv_fel_count',
            'juv_misd_count', 'juv_other_count', 'priors_count',
            'decile_score', 'score_text', 'is_recid', 'two_year_recid',
            'is_violent_recid', 'v_decile_score', 'v_score_text'
        ]
        
        # Filter to only include columns that exist
        available_columns = [col for col in relevant_columns if col in self.df.columns]
        df_cleaned = self.df.select(available_columns)
        
        # Remove rows with missing critical values
        df_cleaned = df_cleaned.filter(
            col('two_year_recid').isNotNull() &
            col('race').isNotNull() &
            col('sex').isNotNull() &
            col('age').isNotNull()
        )
        
        # Handle missing values in numeric columns
        numeric_columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 
                          'priors_count', 'decile_score', 'v_decile_score']
        
        for col_name in numeric_columns:
            if col_name in df_cleaned.columns:
                df_cleaned = df_cleaned.withColumn(
                    col_name,
                    when(col(col_name).isNull(), 0).otherwise(col(col_name))
                )
        
        # Debug: Print available columns after cleaning
        print(f"Columns after cleaning ({len(df_cleaned.columns)}): {', '.join(df_cleaned.columns[:15])}...")
        if 'priors_count' not in df_cleaned.columns:
            # Check if there's a numbered version
            priors_cols = [c for c in df_cleaned.columns if 'priors' in c.lower()]
            if priors_cols:
                print(f"WARNING: priors_count not found, but found: {priors_cols}")
                # Use the first priors column found
                if len(priors_cols) > 0:
                    print(f"Using {priors_cols[0]} as priors_count")
                    df_cleaned = df_cleaned.withColumnRenamed(priors_cols[0], 'priors_count')
        
        # Convert to appropriate types
        if 'decile_score' in df_cleaned.columns:
            df_cleaned = df_cleaned.withColumn('decile_score', col('decile_score').cast(IntegerType()))
        if 'v_decile_score' in df_cleaned.columns:
            df_cleaned = df_cleaned.withColumn('v_decile_score', col('v_decile_score').cast(IntegerType()))
        if 'two_year_recid' in df_cleaned.columns:
            df_cleaned = df_cleaned.withColumn('two_year_recid', col('two_year_recid').cast(IntegerType()))
        if 'is_recid' in df_cleaned.columns:
            df_cleaned = df_cleaned.withColumn('is_recid', col('is_recid').cast(IntegerType()))
        
        # Create binary race categories (simplified for analysis)
        if 'race' in df_cleaned.columns:
            df_cleaned = df_cleaned.withColumn(
                'race_binary',
                when(col('race') == 'African-American', 'African-American')
                .when(col('race') == 'Caucasian', 'Caucasian')
                .otherwise('Other')
            )
        
        # Create age groups
        if 'age' in df_cleaned.columns:
            df_cleaned = df_cleaned.withColumn(
                'age_group',
                when(col('age') < 25, 'Young')
                .when(col('age') < 45, 'Middle')
                .otherwise('Old')
            )
        
        self.processed_df = df_cleaned
        print(f"Data cleaned. Final rows: {self.processed_df.count()}")
        return self.processed_df
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        if self.processed_df is None:
            raise ValueError("Data must be cleaned first. Call clean_data() before prepare_features()")
        
        print("Preparing features for ML...")
        
        # Select features for modeling
        feature_columns = ['age', 'juv_fel_count', 'juv_misd_count', 
                          'juv_other_count', 'priors_count', 'decile_score']
        
        # Filter to existing columns
        available_features = [col for col in feature_columns if col in self.processed_df.columns]
        
        # Index categorical variables
        indexers = []
        if 'sex' in self.processed_df.columns:
            sex_indexer = StringIndexer(inputCol='sex', outputCol='sex_indexed')
            indexers.append(sex_indexer)
        
        if 'race_binary' in self.processed_df.columns:
            race_indexer = StringIndexer(inputCol='race_binary', outputCol='race_indexed')
            indexers.append(race_indexer)
        
        # Assemble features
        feature_list = available_features.copy()
        if 'sex_indexed' in [idx.getOutputCol() for idx in indexers]:
            feature_list.append('sex_indexed')
        if 'race_indexed' in [idx.getOutputCol() for idx in indexers]:
            feature_list.append('race_indexed')
        
        assembler = VectorAssembler(
            inputCols=feature_list,
            outputCol='features',
            handleInvalid='skip'
        )
        
        # Create pipeline
        stages = indexers + [assembler]
        pipeline = Pipeline(stages=stages)
        
        # Fit and transform
        model = pipeline.fit(self.processed_df)
        df_features = model.transform(self.processed_df)
        
        # Scale features
        scaler = StandardScaler(
            inputCol='features',
            outputCol='scaled_features',
            withStd=True,
            withMean=True
        )
        
        scaler_model = scaler.fit(df_features)
        df_scaled = scaler_model.transform(df_features)
        
        self.processed_df = df_scaled
        print("Features prepared successfully")
        return self.processed_df
    
    def get_statistics(self):
        """Get basic statistics about the dataset"""
        if self.processed_df is None:
            raise ValueError("Data must be processed first")
        
        stats = {}
        
        # Overall statistics
        total_count = self.processed_df.count()
        stats['total_records'] = total_count
        
        # Recidivism statistics
        if 'two_year_recid' in self.processed_df.columns:
            recid_count = self.processed_df.filter(col('two_year_recid') == 1).count()
            stats['recidivism_rate'] = recid_count / total_count if total_count > 0 else 0
        
        # Race distribution
        if 'race_binary' in self.processed_df.columns:
            race_dist = self.processed_df.groupBy('race_binary').count().collect()
            stats['race_distribution'] = {row['race_binary']: row['count'] for row in race_dist}
        
        # Gender distribution
        if 'sex' in self.processed_df.columns:
            gender_dist = self.processed_df.groupBy('sex').count().collect()
            stats['gender_distribution'] = {row['sex']: row['count'] for row in gender_dist}
        
        return stats
    
    def to_pandas(self):
        """Convert Spark DataFrame to Pandas DataFrame for analysis"""
        if self.processed_df is None:
            raise ValueError("Data must be processed first")
        return self.processed_df.toPandas()
    
    def save_processed_data(self, output_path):
        """Save processed data to parquet format"""
        if self.processed_df is None:
            raise ValueError("Data must be processed first")
        self.processed_df.write.mode('overwrite').parquet(output_path)
        print(f"Processed data saved to {output_path}")



