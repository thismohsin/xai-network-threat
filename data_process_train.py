#!/usr/bin/env python3
"""
Part 1: XAI Network IDS - Data Preprocessing and Model Training
Improved SMOTE handling with balanced sampling per attack category

Requirements:
pip install scikit-learn pandas numpy matplotlib seaborn
pip install tensorflow xgboost lightgbm imbalanced-learn shap

Input: UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
Output: part1_results.pkl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import time
import json
import os
from datetime import datetime
from collections import Counter
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           accuracy_score, precision_recall_fscore_support)
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Imbalanced learning
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not available, class balancing disabled")

# Gradient boosting
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
    print("TensorFlow available - Deep learning enabled")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available - Using sklearn only")

plt.style.use('default')
sns.set_palette("husl")

def log_checkpoint(message, elapsed_time=None):
    """Log progress with timestamps"""
    current_time = datetime.now().strftime("%H:%M:%S")
    if elapsed_time:
        print(f"[{current_time}] {message} - Elapsed: {elapsed_time:.2f}s")
    else:
        print(f"[{current_time}] {message}")

class ImprovedXAITrainer:
    """Improved XAI trainer with better SMOTE handling"""
    
    def __init__(self):
        self.reset_state()
        
    def reset_state(self):
        """Reset all internal state"""
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.scalers = {}
        self.label_encoders = {}
        self.attack_encoder = None
        self.feature_names = []
        self.needs_smote = False
        self.n_classes = 2
        self.class_names = []
        self.class_distribution = {}
        
    def load_and_preprocess_data(self, train_path="UNSW_NB15_training-set.csv", 
                                test_path="UNSW_NB15_testing-set.csv", 
                                sample_size=75000):
        """Enhanced data loading and preprocessing"""
        start_time = time.time()
        log_checkpoint("Starting enhanced data loading and preprocessing...")
        
        try:
            # Load data
            log_checkpoint(f"Loading training data: {train_path}")
            train_df = pd.read_csv(train_path)
            
            log_checkpoint(f"Loading test data: {test_path}")
            test_df = pd.read_csv(test_path)
            
            # Intelligent sampling to maintain class distribution
            if len(train_df) > sample_size:
                log_checkpoint(f"Performing stratified sampling from {len(train_df)} to {sample_size}")
                # Get class distribution
                class_counts = train_df['attack_cat'].value_counts()
                log_checkpoint(f"Original class distribution: {dict(class_counts)}")
                
                # Calculate sampling ratios while maintaining representation
                sample_ratios = {}
                min_samples_per_class = max(100, sample_size // (len(class_counts) * 10))  # Minimum samples per class
                
                for attack_cat, count in class_counts.items():
                    if count > min_samples_per_class:
                        # For large classes, sample proportionally but cap at reasonable size
                        max_samples = min(count, sample_size // 3)  # No single class dominates
                        sample_ratios[attack_cat] = min(max_samples, int(count * (sample_size / len(train_df))))
                    else:
                        # Keep all samples for rare classes
                        sample_ratios[attack_cat] = count
                
                # Adjust to meet total sample size
                total_samples = sum(sample_ratios.values())
                if total_samples > sample_size:
                    # Scale down proportionally
                    scale_factor = sample_size / total_samples
                    sample_ratios = {k: max(min_samples_per_class, int(v * scale_factor)) 
                                   for k, v in sample_ratios.items()}
                
                log_checkpoint(f"Target sampling: {sample_ratios}")
                
                # Perform stratified sampling
                sampled_dfs = []
                for attack_cat, n_samples in sample_ratios.items():
                    class_data = train_df[train_df['attack_cat'] == attack_cat]
                    if len(class_data) > n_samples:
                        sampled_data = class_data.sample(n=n_samples, random_state=42)
                    else:
                        sampled_data = class_data
                    sampled_dfs.append(sampled_data)
                
                train_df = pd.concat(sampled_dfs, ignore_index=True)
                train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
                
                log_checkpoint(f"Final training size: {len(train_df)}")
            
            # Sample test data proportionally
            if len(test_df) > sample_size // 2:
                test_size = sample_size // 2
                test_class_counts = test_df['attack_cat'].value_counts()
                test_sample_ratios = {}
                
                for attack_cat, count in test_class_counts.items():
                    proportion = count / len(test_df)
                    test_sample_ratios[attack_cat] = max(50, int(test_size * proportion))
                
                # Adjust to meet test size
                total_test = sum(test_sample_ratios.values())
                if total_test > test_size:
                    scale_factor = test_size / total_test
                    test_sample_ratios = {k: max(20, int(v * scale_factor)) 
                                        for k, v in test_sample_ratios.items()}
                
                sampled_test_dfs = []
                for attack_cat, n_samples in test_sample_ratios.items():
                    class_data = test_df[test_df['attack_cat'] == attack_cat]
                    if len(class_data) > n_samples:
                        sampled_data = class_data.sample(n=n_samples, random_state=42)
                    else:
                        sampled_data = class_data
                    sampled_test_dfs.append(sampled_data)
                
                test_df = pd.concat(sampled_test_dfs, ignore_index=True)
                test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                log_checkpoint(f"Final test size: {len(test_df)}")
            
            # Combine for preprocessing
            train_df['dataset'] = 'train'
            test_df['dataset'] = 'test'
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            
            log_checkpoint(f"Combined dataset shape: {combined_df.shape}")
            
            # Remove unnecessary columns
            cols_to_drop = ['id'] if 'id' in combined_df.columns else []
            if cols_to_drop:
                combined_df = combined_df.drop(columns=cols_to_drop)
            
            # Check for target column
            if 'attack_cat' not in combined_df.columns:
                raise ValueError("Target column 'attack_cat' not found in dataset")
            
            # Analyze final class distribution
            class_dist = combined_df[combined_df['dataset'] == 'train']['attack_cat'].value_counts()
            self.n_classes = len(class_dist)
            self.class_names = class_dist.index.tolist()
            self.class_distribution = dict(class_dist)
            
            log_checkpoint(f"Final training class distribution:")
            for cat, count in class_dist.items():
                percentage = (count / class_dist.sum()) * 100
                log_checkpoint(f"  {cat}: {count} ({percentage:.1f}%)")
            
            # Determine SMOTE strategy
            min_samples = class_dist.min()
            max_samples = class_dist.max()
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
            
            # More conservative SMOTE application
            self.needs_smote = imbalance_ratio > 5 and IMBLEARN_AVAILABLE and min_samples >= 50
            
            log_checkpoint(f"Class imbalance ratio: {imbalance_ratio:.1f}")
            log_checkpoint(f"SMOTE will be applied: {self.needs_smote}")
            
            # Enhanced preprocessing
            log_checkpoint("Starting enhanced preprocessing...")
            
            # Handle categorical features with better encoding
            categorical_cols = ['proto', 'service', 'state']
            for col in categorical_cols:
                if col in combined_df.columns:
                    # Fill missing values with mode
                    mode_val = combined_df[col].mode()
                    if len(mode_val) > 0:
                        combined_df[col] = combined_df[col].fillna(mode_val[0])
                    else:
                        combined_df[col] = combined_df[col].fillna('unknown')
                    
                    # Encode with frequency-based approach for rare categories
                    value_counts = combined_df[col].value_counts()
                    rare_categories = value_counts[value_counts < 10].index
                    combined_df[col] = combined_df[col].replace(rare_categories, 'rare_category')
                    
                    # Label encode
                    le = LabelEncoder()
                    combined_df[col] = le.fit_transform(combined_df[col].astype(str))
                    self.label_encoders[col] = le
            
            # Enhanced numerical feature processing
            numerical_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col not in ['attack_cat']]
            
            for col in numerical_cols:
                # Handle missing values with median
                median_val = combined_df[col].median()
                combined_df[col] = combined_df[col].fillna(median_val)
                
                # More robust outlier handling using IQR method
                Q1 = combined_df[col].quantile(0.25)
                Q3 = combined_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap extreme outliers
                combined_df[col] = combined_df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Enhanced feature engineering
            log_checkpoint("Performing enhanced feature engineering...")
            
            # Create comprehensive ratio features
            if 'sbytes' in combined_df.columns and 'dbytes' in combined_df.columns:
                combined_df['bytes_ratio'] = combined_df['sbytes'] / (combined_df['dbytes'] + 1)
                combined_df['total_bytes'] = combined_df['sbytes'] + combined_df['dbytes']
            
            if 'spkts' in combined_df.columns and 'dpkts' in combined_df.columns:
                combined_df['pkts_ratio'] = combined_df['spkts'] / (combined_df['dpkts'] + 1)
                combined_df['total_pkts'] = combined_df['spkts'] + combined_df['dpkts']
            
            # Duration-based features
            if 'dur' in combined_df.columns:
                combined_df['dur_log'] = np.log1p(combined_df['dur'])
                if 'total_bytes' in combined_df.columns:
                    combined_df['bytes_per_second'] = combined_df['total_bytes'] / (combined_df['dur'] + 1)
            
            # Encode target variable
            self.attack_encoder = LabelEncoder()
            combined_df['attack_cat_encoded'] = self.attack_encoder.fit_transform(combined_df['attack_cat'])
            
            # Prepare features with intelligent selection
            feature_cols = [col for col in combined_df.columns 
                          if col not in ['attack_cat', 'attack_cat_encoded', 'dataset']]
            
            X = combined_df[feature_cols]
            y = combined_df['attack_cat_encoded']
            
            # Feature selection for high-dimensional data
            if len(feature_cols) > 40:
                log_checkpoint(f"Selecting top 40 features from {len(feature_cols)}...")
                try:
                    selector = SelectKBest(mutual_info_classif, k=40)
                    X_selected = selector.fit_transform(X, y)
                    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                    X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                    feature_cols = selected_features
                    log_checkpoint(f"Selected features: {selected_features[:10]}...")
                except Exception as e:
                    log_checkpoint(f"Feature selection failed: {e}, using all features")
            
            self.feature_names = feature_cols
            
            # Split back to train/test
            train_mask = combined_df['dataset'] == 'train'
            test_mask = combined_df['dataset'] == 'test'
            
            self.X_train = X[train_mask].reset_index(drop=True)
            self.X_test = X[test_mask].reset_index(drop=True)
            self.y_train = y[train_mask].reset_index(drop=True)
            self.y_test = y[test_mask].reset_index(drop=True)
            
            # Create multiple scalers for different model types
            log_checkpoint("Creating optimized data scalers...")
            
            # Standard scaler for tree models
            scaler_standard = StandardScaler()
            scaler_standard.fit(self.X_train)
            self.scalers['standard'] = scaler_standard
            
            # MinMax scaler for neural networks
            scaler_minmax = MinMaxScaler()
            scaler_minmax.fit(self.X_train)
            self.scalers['minmax'] = scaler_minmax
            
            elapsed = time.time() - start_time
            log_checkpoint(f"Enhanced preprocessing completed", elapsed)
            log_checkpoint(f"Training samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
            log_checkpoint(f"Features: {len(self.feature_names)}, Classes: {self.n_classes}")
            
            return True
            
        except Exception as e:
            log_checkpoint(f"Error in data loading/preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False

    def handle_class_imbalance_improved(self):
        """Improved SMOTE with controlled synthetic data generation"""
        if not self.needs_smote:
            log_checkpoint("No class balancing needed - using original data")
            self.X_train_resampled = self.X_train.copy()
            self.y_train_resampled = self.y_train.copy()
            return True
        
        try:
            log_checkpoint("Applying improved SMOTE strategy...")
            
            # Analyze current class distribution
            class_counts = pd.Series(self.y_train).value_counts().sort_index()
            log_checkpoint(f"Original distribution: {dict(class_counts)}")
            
            # Calculate balanced target sizes
            median_size = int(class_counts.median())
            max_allowed_size = median_size * 2  # Prevent any class from being too large
            min_target_size = max(100, median_size // 2)  # Minimum viable size
            
            # Create sampling strategy
            sampling_strategy = {}
            for class_label, count in class_counts.items():
                if count < min_target_size:
                    # Boost very small classes
                    target_size = min(min_target_size, count * 3)
                    sampling_strategy[class_label] = target_size
                elif count > max_allowed_size:
                    # Don't oversample large classes
                    continue
                else:
                    # Moderate boost for medium classes
                    target_size = min(max_allowed_size, int(count * 1.5))
                    if target_size > count:
                        sampling_strategy[class_label] = target_size
            
            log_checkpoint(f"Sampling strategy: {sampling_strategy}")
            
            if not sampling_strategy:
                log_checkpoint("No classes need resampling")
                self.X_train_resampled = self.X_train.copy()
                self.y_train_resampled = self.y_train.copy()
                return True
            
            # Calculate appropriate k_neighbors
            min_class_size = min([class_counts[label] for label in sampling_strategy.keys()])
            k_neighbors = max(1, min(5, min_class_size - 1))
            
            # Apply SMOTE with the calculated strategy
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42, 
                k_neighbors=k_neighbors
            )
            
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                self.X_train, self.y_train
            )
            
            # Verify results
            new_class_counts = pd.Series(self.y_train_resampled).value_counts().sort_index()
            log_checkpoint(f"After SMOTE distribution: {dict(new_class_counts)}")
            
            total_synthetic = len(self.X_train_resampled) - len(self.X_train)
            log_checkpoint(f"SMOTE completed: {len(self.X_train)} -> {len(self.X_train_resampled)} samples")
            log_checkpoint(f"Generated {total_synthetic} synthetic samples")
            
            return True
            
        except Exception as e:
            log_checkpoint(f"Improved SMOTE failed: {e}, using original data")
            self.X_train_resampled = self.X_train.copy()
            self.y_train_resampled = self.y_train.copy()
            return True
    
    def create_tensorflow_model(self, input_dim, n_classes, architecture='optimized'):
        """Create optimized TensorFlow/Keras model"""
        if not TF_AVAILABLE:
            return None
        
        try:
            model = Sequential()
            
            if architecture == 'optimized':
                # Optimized architecture for network intrusion detection
                model.add(Dense(256, activation='relu', input_dim=input_dim))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                
                model.add(Dense(128, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                
                model.add(Dense(64, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.2))
                
                model.add(Dense(32, activation='relu'))
                model.add(Dropout(0.2))
            
            # Output layer
            if n_classes > 2:
                model.add(Dense(n_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
            else:
                model.add(Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            
            # Compile with optimized settings
            model.compile(
                optimizer=Adam(learning_rate=0.001, decay=1e-6),
                loss=loss,
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            log_checkpoint(f"Error creating TensorFlow model: {e}")
            return None
    
    def train_models_comprehensive(self, max_time_minutes=60):
        """Comprehensive model training with enhanced hyperparameter tuning"""
        start_time = time.time()
        max_time_seconds = max_time_minutes * 60
        
        log_checkpoint(f"Starting comprehensive model training - Max time: {max_time_minutes} minutes")
        
        # Use resampled data if available
        X_train = self.X_train_resampled if hasattr(self, 'X_train_resampled') else self.X_train
        y_train = self.y_train_resampled if hasattr(self, 'y_train_resampled') else self.y_train
        
        log_checkpoint(f"Training with {len(X_train)} samples, {len(self.X_test)} test samples")
        
        results = {}
        
        # Random Forest with expanded parameter grid
        if time.time() - start_time < max_time_seconds:
            try:
                log_checkpoint("Training Random Forest with enhanced parameters...")
                
                rf_params = {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [15, 25, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', 'balanced_subsample'],
                    'max_features': ['sqrt', 'log2']
                }
                
                rf = RandomForestClassifier(random_state=42, n_jobs=-1)
                rf_search = RandomizedSearchCV(
                    rf, rf_params, n_iter=15, cv=3, 
                    scoring='f1_weighted', random_state=42, n_jobs=-1,
                    verbose=1
                )
                
                rf_search.fit(X_train, y_train)
                
                # Test prediction
                y_pred = rf_search.best_estimator_.predict(self.X_test)
                acc = accuracy_score(self.y_test, y_pred)
                f1 = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')[2]
                
                results['Random Forest'] = {
                    'model': rf_search.best_estimator_,
                    'accuracy': acc,
                    'f1': f1,
                    'params': rf_search.best_params_,
                    'cv_score': rf_search.best_score_
                }
                
                log_checkpoint(f"Random Forest: Accuracy={acc:.4f}, F1={f1:.4f}, CV={rf_search.best_score_:.4f}")
                
            except Exception as e:
                log_checkpoint(f"Random Forest training failed: {e}")
        
        # XGBoost with advanced tuning
        if XGBOOST_AVAILABLE and time.time() - start_time < max_time_seconds * 0.8:
            try:
                log_checkpoint("Training XGBoost with advanced parameters...")
                
                xgb_params = {
                    'n_estimators': [200, 400, 600],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.01, 0.1],
                    'reg_lambda': [1, 1.5, 2]
                }
                
                xgb = XGBClassifier(
                    random_state=42, 
                    n_jobs=-1, 
                    use_label_encoder=False, 
                    eval_metric='mlogloss' if self.n_classes > 2 else 'logloss'
                )
                
                xgb_search = RandomizedSearchCV(
                    xgb, xgb_params, n_iter=12, cv=3,
                    scoring='f1_weighted', random_state=42, n_jobs=-1,
                    verbose=1
                )
                
                xgb_search.fit(X_train, y_train)
                
                y_pred = xgb_search.best_estimator_.predict(self.X_test)
                acc = accuracy_score(self.y_test, y_pred)
                f1 = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')[2]
                
                results['XGBoost'] = {
                    'model': xgb_search.best_estimator_,
                    'accuracy': acc,
                    'f1': f1,
                    'params': xgb_search.best_params_,
                    'cv_score': xgb_search.best_score_
                }
                
                log_checkpoint(f"XGBoost: Accuracy={acc:.4f}, F1={f1:.4f}, CV={xgb_search.best_score_:.4f}")
                
            except Exception as e:
                log_checkpoint(f"XGBoost training failed: {e}")
        
        # LightGBM with optimized parameters
        if LIGHTGBM_AVAILABLE and time.time() - start_time < max_time_seconds * 0.8:
            try:
                log_checkpoint("Training LightGBM with optimized parameters...")
                
                lgb_params = {
                    'n_estimators': [200, 400, 600],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'num_leaves': [31, 50, 100],
                    'feature_fraction': [0.8, 0.9],
                    'bagging_fraction': [0.8, 0.9],
                    'reg_alpha': [0, 0.01, 0.1],
                    'reg_lambda': [0, 0.01, 0.1]
                }
                
                lgb = LGBMClassifier(
                    random_state=42, 
                    n_jobs=-1, 
                    verbose=-1, 
                    class_weight='balanced'
                )
                
                lgb_search = RandomizedSearchCV(
                    lgb, lgb_params, n_iter=12, cv=3,
                    scoring='f1_weighted', random_state=42, n_jobs=-1,
                    verbose=1
                )
                
                lgb_search.fit(X_train, y_train)
                
                y_pred = lgb_search.best_estimator_.predict(self.X_test)
                acc = accuracy_score(self.y_test, y_pred)
                f1 = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')[2]
                
                results['LightGBM'] = {
                    'model': lgb_search.best_estimator_,
                    'accuracy': acc,
                    'f1': f1,
                    'params': lgb_search.best_params_,
                    'cv_score': lgb_search.best_score_
                }
                
                log_checkpoint(f"LightGBM: Accuracy={acc:.4f}, F1={f1:.4f}, CV={lgb_search.best_score_:.4f}")
                
            except Exception as e:
                log_checkpoint(f"LightGBM training failed: {e}")
        
        # Enhanced Neural Network (sklearn)
        if time.time() - start_time < max_time_seconds * 0.9:
            try:
                log_checkpoint("Training Enhanced Neural Network (sklearn)...")
                
                # Scale data for neural network
                X_train_scaled = self.scalers['minmax'].transform(X_train)
                X_test_scaled = self.scalers['minmax'].transform(self.X_test)
                
                nn_params = {
                    'hidden_layer_sizes': [(150, 100, 50), (200, 100), (128, 64, 32)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01],
                    'beta_1': [0.9, 0.95],
                    'beta_2': [0.999, 0.99]
                }
                
                nn = MLPClassifier(
                    random_state=42, 
                    max_iter=300, 
                    early_stopping=True,
                    validation_fraction=0.1
                )
                
                nn_search = RandomizedSearchCV(
                    nn, nn_params, n_iter=8, cv=3,
                    scoring='f1_weighted', random_state=42, n_jobs=-1,
                    verbose=1
                )
                
                nn_search.fit(X_train_scaled, y_train)
                
                y_pred = nn_search.best_estimator_.predict(X_test_scaled)
                acc = accuracy_score(self.y_test, y_pred)
                f1 = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')[2]
                
                results['Neural Network'] = {
                    'model': nn_search.best_estimator_,
                    'accuracy': acc,
                    'f1': f1,
                    'params': nn_search.best_params_,
                    'scaler': 'minmax',
                    'cv_score': nn_search.best_score_
                }
                
                log_checkpoint(f"Neural Network: Accuracy={acc:.4f}, F1={f1:.4f}, CV={nn_search.best_score_:.4f}")
                
            except Exception as e:
                log_checkpoint(f"Neural Network training failed: {e}")
        
        # Deep Learning (TensorFlow) - Enhanced
        if TF_AVAILABLE and time.time() - start_time < max_time_seconds * 0.95:
            try:
                log_checkpoint("Training Deep Neural Network with enhanced architecture...")
                
                # Prepare data
                X_train_scaled = self.scalers['minmax'].transform(X_train)
                X_test_scaled = self.scalers['minmax'].transform(self.X_test)
                
                # Create model
                model = self.create_tensorflow_model(
                    input_dim=X_train_scaled.shape[1],
                    n_classes=self.n_classes,
                    architecture='optimized'
                )
                
                if model is not None:
                    # Enhanced training with callbacks
                    callbacks = [
                        EarlyStopping(patience=15, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
                    ]
                    
                    history = model.fit(
                        X_train_scaled, y_train,
                        epochs=100,
                        batch_size=128,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Evaluate
                    if self.n_classes > 2:
                        y_pred_proba = model.predict(X_test_scaled)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                    else:
                        y_pred_proba = model.predict(X_test_scaled)
                        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                    
                    acc = accuracy_score(self.y_test, y_pred)
                    f1 = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')[2]
                    
                    results['Deep Neural Network'] = {
                        'model': model,
                        'accuracy': acc,
                        'f1': f1,
                        'params': {'architecture': 'optimized', 'epochs': len(history.history['loss'])},
                        'scaler': 'minmax',
                        'history': history.history
                    }
                    
                    log_checkpoint(f"Deep Neural Network: Accuracy={acc:.4f}, F1={f1:.4f}")
                
            except Exception as e:
                log_checkpoint(f"Deep Neural Network training failed: {e}")
        
        elapsed = time.time() - start_time
        log_checkpoint(f"Comprehensive model training completed in {elapsed/60:.2f} minutes")
        
        return results

def run_part1():
    """Enhanced Part 1: Model training and tuning"""
    log_checkpoint("Starting Enhanced Part 1: Model Training and Tuning")
    
    trainer = ImprovedXAITrainer()
    
    # Check for dataset files
    if not (os.path.exists("UNSW_NB15_training-set.csv") and 
            os.path.exists("UNSW_NB15_testing-set.csv")):
        log_checkpoint("ERROR: Dataset files not found!")
        log_checkpoint("Required files:")
        log_checkpoint("  - UNSW_NB15_training-set.csv")
        log_checkpoint("  - UNSW_NB15_testing-set.csv")
        return False
    
    try:
        # Load and preprocess data with enhanced methods
        if not trainer.load_and_preprocess_data():
            return False
        
        # Apply improved class balancing
        if not trainer.handle_class_imbalance_improved():
            return False
        
        # Train models with comprehensive tuning
        results = trainer.train_models_comprehensive()
        
        if not results:
            log_checkpoint("No models were successfully trained")
            return False
        
        # Enhanced results summary
        print("\n" + "="*80)
        print("PART 1 TRAINING SUMMARY")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'F1-Score': result['f1'],
                'CV Score': result.get('cv_score', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        print(comparison_df.to_string(index=False))
        
        best_model = comparison_df.iloc[0]['Model']
        print(f"\nBest Performing Model: {best_model}")
        print(f"F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
        print(f"Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
        
        # Save enhanced results
        save_data = {
            'results': results,
            'comparison_df': comparison_df,
            'trainer_state': {
                'X_train': trainer.X_train,
                'X_test': trainer.X_test, 
                'y_train': trainer.y_train,
                'y_test': trainer.y_test,
                'X_train_resampled': getattr(trainer, 'X_train_resampled', None),
                'y_train_resampled': getattr(trainer, 'y_train_resampled', None),
                'scalers': trainer.scalers,
                'label_encoders': trainer.label_encoders,
                'attack_encoder': trainer.attack_encoder,
                'feature_names': trainer.feature_names,
                'class_names': trainer.class_names,
                'n_classes': trainer.n_classes,
                'class_distribution': trainer.class_distribution
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'train_samples': len(trainer.X_train),
                    'test_samples': len(trainer.X_test),
                    'features': len(trainer.feature_names),
                    'classes': trainer.n_classes
                },
                'preprocessing': {
                    'smote_applied': trainer.needs_smote,
                    'feature_selection': len(trainer.feature_names) <= 40
                }
            }
        }
        
        with open('part1_results.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        
        # Generate training visualization
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Model comparison
        plt.subplot(2, 2, 1)
        models = comparison_df['Model']
        f1_scores = comparison_df['F1-Score']
        bars = plt.bar(range(len(models)), f1_scores, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'][:len(models)])
        plt.xlabel('Models')
        plt.ylabel('F1-Score')
        plt.title('Model Performance Comparison')
        plt.xticks(range(len(models)), models, rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 2: Class distribution
        plt.subplot(2, 2, 2)
        class_dist = pd.Series(trainer.class_distribution)
        plt.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', startangle=90)
        plt.title('Training Class Distribution')
        
        # Plot 3: Feature importance (for best tree model)
        best_tree_model = None
        for name, result in results.items():
            if hasattr(result['model'], 'feature_importances_'):
                best_tree_model = result['model']
                best_tree_name = name
                break
        
        if best_tree_model is not None:
            plt.subplot(2, 2, 3)
            feature_imp = pd.DataFrame({
                'feature': trainer.feature_names,
                'importance': best_tree_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            plt.barh(range(len(feature_imp)), feature_imp['importance'])
            plt.yticks(range(len(feature_imp)), feature_imp['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top Features ({best_tree_name})')
            plt.gca().invert_yaxis()
        
        # Plot 4: Training history for deep learning
        if 'Deep Neural Network' in results and 'history' in results['Deep Neural Network']:
            plt.subplot(2, 2, 4)
            history = results['Deep Neural Network']['history']
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Deep Learning Training History')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('part1_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        log_checkpoint("Part 1 completed successfully!")
        log_checkpoint("Generated files:")
        log_checkpoint("  - part1_results.pkl (Model results and state)")
        log_checkpoint("  - part1_training_results.png (Visualization)")
        
        return True
        
    except Exception as e:
        log_checkpoint(f"Part 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("XAI Network IDS - Part 1: Enhanced Training and Preprocessing")
    print("=" * 60)
    
    # Display system information
    print(f"TensorFlow: {'Available' if TF_AVAILABLE else 'Not Available'}")
    print(f"XGBoost: {'Available' if XGBOOST_AVAILABLE else 'Not Available'}")
    print(f"LightGBM: {'Available' if LIGHTGBM_AVAILABLE else 'Not Available'}")
    print(f"Imbalanced-Learn: {'Available' if IMBLEARN_AVAILABLE else 'Not Available'}")
    print()
    
    success = run_part1()
    
    if success:
        print("\n✅ Part 1 completed successfully!")
        print("Ready to proceed to Part 2: Model Analysis")
    else:
        print("\n❌ Part 1 failed!")
        print("Please check the error messages above.")