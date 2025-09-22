#!/usr/bin/env python3
"""
Part 2: XAI Network IDS - Model Analysis and Evaluation
Enhanced model comparison with detailed performance metrics

Requirements:
pip install scikit-learn pandas numpy matplotlib seaborn plotly

Input: part1_results.pkl
Output: part2_results.pkl
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
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           accuracy_score, precision_recall_fscore_support,
                           roc_curve, precision_recall_curve, auc)
from sklearn.preprocessing import LabelBinarizer
warnings.filterwarnings('ignore')

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available - using matplotlib only")

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Try to import TensorFlow for deep learning model handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

def log_checkpoint(message, elapsed_time=None):
    """Log progress with timestamps"""
    current_time = datetime.now().strftime("%H:%M:%S")
    if elapsed_time:
        print(f"[{current_time}] {message} - Elapsed: {elapsed_time:.2f}s")
    else:
        print(f"[{current_time}] {message}")

class ModelAnalyzer:
    """Comprehensive model analysis and evaluation"""
    
    def __init__(self):
        self.results = None
        self.trainer_state = None
        self.comparison_df = None
        self.metadata = None
        self.analysis_results = {}
    
    def load_part1_results(self, filepath='part1_results.pkl'):
        """Load results from Part 1"""
        try:
            log_checkpoint(f"Loading Part 1 results from {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Part 1 results file not found: {filepath}")
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.results = data['results']
            self.trainer_state = data['trainer_state']
            self.comparison_df = data['comparison_df']
            self.metadata = data.get('metadata', {})
            
            log_checkpoint(f"Loaded {len(self.results)} trained models")
            log_checkpoint(f"Dataset info: {len(self.trainer_state['X_test'])} test samples, {len(self.trainer_state['feature_names'])} features")
            
            return True
            
        except Exception as e:
            log_checkpoint(f"Error loading Part 1 results: {e}")
            return False
    
    def detailed_model_evaluation(self):
        """Perform detailed evaluation of all models"""
        log_checkpoint("Starting detailed model evaluation...")
        
        detailed_results = {}
        
        for model_name, model_info in self.results.items():
            try:
                log_checkpoint(f"Evaluating {model_name}...")
                
                # Get model and data
                model = model_info['model']
                X_test = self.trainer_state['X_test']
                y_test = self.trainer_state['y_test']
                
                # Scale data if needed
                if model_name in ['Neural Network', 'Deep Neural Network']:
                    scaler = self.trainer_state['scalers']['minmax']
                    X_test_scaled = scaler.transform(X_test)
                    
                    if model_name == 'Deep Neural Network' and TF_AVAILABLE:
                        # Handle TensorFlow model predictions
                        if self.trainer_state['n_classes'] > 2:
                            y_pred_proba = model.predict(X_test_scaled, verbose=0)
                            y_pred = np.argmax(y_pred_proba, axis=1)
                        else:
                            y_pred_proba = model.predict(X_test_scaled, verbose=0)
                            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                    else:
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate comprehensive metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
                macro_f1 = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)[2]
                weighted_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)[2]
                
                # Per-class metrics
                class_report = classification_report(
                    y_test, y_pred,
                    target_names=self.trainer_state['class_names'],
                    output_dict=True,
                    zero_division=0
                )
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # ROC AUC for multi-class
                roc_auc = None
                if y_pred_proba is not None and self.trainer_state['n_classes'] > 1:
                    try:
                        if self.trainer_state['n_classes'] == 2:
                            if y_pred_proba.ndim > 1:
                                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                            else:
                                roc_auc = roc_auc_score(y_test, y_pred_proba)
                        else:
                            # Multi-class ROC AUC
                            lb = LabelBinarizer()
                            y_test_binary = lb.fit_transform(y_test)
                            if y_test_binary.shape[1] == 1:
                                y_test_binary = np.hstack([1 - y_test_binary, y_test_binary])
                            roc_auc = roc_auc_score(y_test_binary, y_pred_proba, multi_class='ovr', average='weighted')
                    except Exception as e:
                        log_checkpoint(f"ROC AUC calculation failed for {model_name}: {e}")
                
                # Store detailed results
                detailed_results[model_name] = {
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'metrics': {
                        'accuracy': accuracy,
                        'macro_f1': macro_f1,
                        'weighted_f1': weighted_f1,
                        'per_class_precision': precision,
                        'per_class_recall': recall,
                        'per_class_f1': f1,
                        'roc_auc': roc_auc
                    },
                    'classification_report': class_report,
                    'confusion_matrix': cm,
                    'model_info': model_info
                }
                
                log_checkpoint(f"{model_name} evaluation completed - Accuracy: {accuracy:.4f}, F1: {weighted_f1:.4f}")
                
            except Exception as e:
                log_checkpoint(f"Error evaluating {model_name}: {e}")
                continue
        
        self.analysis_results = detailed_results
        return detailed_results
    
    def generate_performance_comparison(self):
        """Generate comprehensive performance comparison"""
        log_checkpoint("Generating performance comparison...")
        
        if not self.analysis_results:
            log_checkpoint("No analysis results available")
            return None
        
        # Create enhanced comparison DataFrame
        comparison_data = []
        for model_name, result in self.analysis_results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Macro F1': metrics['macro_f1'],
                'Weighted F1': metrics['weighted_f1'],
                'ROC AUC': metrics.get('roc_auc', 'N/A'),
                'CV Score': result['model_info'].get('cv_score', 'N/A')
            })
        
        enhanced_comparison = pd.DataFrame(comparison_data)
        enhanced_comparison = enhanced_comparison.sort_values('Weighted F1', ascending=False)
        
        print("\n" + "="*100)
        print("ENHANCED MODEL PERFORMANCE COMPARISON")
        print("="*100)
        print(enhanced_comparison.to_string(index=False, float_format='%.4f'))
        
        # Performance analysis
        best_model = enhanced_comparison.iloc[0]
        print(f"\nBest Overall Model: {best_model['Model']}")
        print(f"Weighted F1-Score: {best_model['Weighted F1']:.4f}")
        print(f"Accuracy: {best_model['Accuracy']:.4f}")
        
        if best_model['Accuracy'] > 0.95:
            performance_level = "EXCELLENT"
        elif best_model['Accuracy'] > 0.90:
            performance_level = "VERY GOOD"
        elif best_model['Accuracy'] > 0.85:
            performance_level = "GOOD"
        elif best_model['Accuracy'] > 0.80:
            performance_level = "ACCEPTABLE"
        else:
            performance_level = "NEEDS IMPROVEMENT"
        
        print(f"Performance Level: {performance_level}")
        
        return enhanced_comparison
    
    def analyze_per_class_performance(self):
        """Detailed per-class performance analysis"""
        log_checkpoint("Analyzing per-class performance...")
        
        if not self.analysis_results:
            return None, None
        
        best_model_name = max(self.analysis_results.keys(), 
                             key=lambda x: self.analysis_results[x]['metrics']['weighted_f1'])
        best_result = self.analysis_results[best_model_name]
        
        print(f"\n" + "="*80)
        print(f"DETAILED PER-CLASS ANALYSIS - {best_model_name}")
        print("="*80)
        
        # Create per-class DataFrame
        class_metrics = []
        class_report = best_result['classification_report']
        
        for class_name in self.trainer_state['class_names']:
            if class_name in class_report:
                metrics = class_report[class_name]
                class_metrics.append({
                    'Attack Type': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': int(metrics['support'])
                })
        
        class_df = pd.DataFrame(class_metrics)
        class_df = class_df.sort_values('F1-Score', ascending=False)
        
        print(class_df.to_string(index=False, float_format='%.4f'))
        
        # Identify best and worst performing classes
        best_class = class_df.iloc[0]
        worst_class = class_df.iloc[-1]
        
        print(f"\nBest Detected Attack: {best_class['Attack Type']} (F1: {best_class['F1-Score']:.4f})")
        print(f"Most Challenging Attack: {worst_class['Attack Type']} (F1: {worst_class['F1-Score']:.4f})")
        
        # Calculate class difficulty score
        class_df['Difficulty_Score'] = (1 - class_df['F1-Score']) * 100
        
        return class_df, best_model_name
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        log_checkpoint("Creating comprehensive visualizations...")
        
        if not self.analysis_results:
            return False
        
        # Create subplot figure
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('XAI Network IDS - Comprehensive Model Analysis', fontsize=16, fontweight='bold')
        
        # 1. Model Performance Comparison (Bar Chart)
        ax1 = axes[0, 0]
        enhanced_comparison = self.generate_performance_comparison()
        models = enhanced_comparison['Model']
        f1_scores = enhanced_comparison['Weighted F1']
        accuracies = enhanced_comparison['Accuracy']
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, f1_scores, width, label='Weighted F1', alpha=0.8)
        bars2 = ax1.bar(x + width/2, accuracies, width, label='Accuracy', alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        
        # Add value labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                    f'{f1_scores.iloc[i]:.3f}', ha='center', va='bottom', fontsize=8)
            ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                    f'{accuracies.iloc[i]:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Best Model Confusion Matrix
        ax2 = axes[0, 1]
        best_model_name = enhanced_comparison.iloc[0]['Model']
        cm = self.analysis_results[best_model_name]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.trainer_state['class_names'],
                   yticklabels=self.trainer_state['class_names'],
                   ax=ax2)
        ax2.set_title(f'Confusion Matrix - {best_model_name}')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # 3. Per-Class Performance
        ax3 = axes[0, 2]
        class_df, _ = self.analyze_per_class_performance()
        if class_df is not None:
            top_classes = class_df.head(8)
            bars = ax3.barh(range(len(top_classes)), top_classes['F1-Score'])
            ax3.set_yticks(range(len(top_classes)))
            ax3.set_yticklabels(top_classes['Attack Type'])
            ax3.set_xlabel('F1-Score')
            ax3.set_title('Per-Class F1-Score (Top 8)')
            ax3.invert_yaxis()
            
            # Color coding for performance
            for i, bar in enumerate(bars):
                score = top_classes['F1-Score'].iloc[i]
                if score > 0.9:
                    bar.set_color('#2ecc71')  # Green for excellent
                elif score > 0.8:
                    bar.set_color('#3498db')  # Blue for good
                elif score > 0.7:
                    bar.set_color('#f39c12')  # Orange for fair
                else:
                    bar.set_color('#e74c3c')  # Red for poor
        
        # 4. Feature Importance (for best tree-based model)
        ax4 = axes[1, 0]
        best_tree_model = None
        best_tree_name = None
        
        for name, result in self.analysis_results.items():
            model = result['model_info']['model']
            if hasattr(model, 'feature_importances_'):
                best_tree_model = model
                best_tree_name = name
                break
        
        if best_tree_model is not None:
            feature_imp = pd.DataFrame({
                'feature': self.trainer_state['feature_names'],
                'importance': best_tree_model.feature_importances_
            }).sort_values('importance', ascending=False).head(12)
            
            bars = ax4.barh(range(len(feature_imp)), feature_imp['importance'])
            ax4.set_yticks(range(len(feature_imp)))
            ax4.set_yticklabels(feature_imp['feature'])
            ax4.set_xlabel('Importance')
            ax4.set_title(f'Feature Importance - {best_tree_name}')
            ax4.invert_yaxis()
        else:
            ax4.text(0.5, 0.5, 'No tree-based model available\nfor feature importance', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance - N/A')
        
        # 5. Model Comparison Radar Chart
        ax5 = axes[1, 1]
        try:
            # Create a simplified radar chart using regular plot
            metrics_for_radar = ['Accuracy', 'Macro F1', 'Weighted F1']
            top_models = enhanced_comparison.head(3)
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
            
            for idx, (_, model_row) in enumerate(top_models.iterrows()):
                values = []
                for metric in metrics_for_radar:
                    val = model_row[metric]
                    if isinstance(val, str) or pd.isna(val):
                        val = 0
                    values.append(val)
                
                values += values[:1]  # Complete the circle
                
                ax5.plot(angles, values, 'o-', linewidth=2, label=model_row['Model'])
                ax5.fill(angles, values, alpha=0.25)
            
            ax5.set_xticks(angles[:-1])
            ax5.set_xticklabels(metrics_for_radar)
            ax5.set_ylim(0, 1)
            ax5.set_title('Top 3 Models Comparison')
            ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            ax5.grid(True)
            
        except Exception as e:
            ax5.text(0.5, 0.5, f'Radar chart failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Model Comparison (Failed)')
        
        # 6. ROC Curves (for binary classification or best class)
        ax6 = axes[1, 2]
        try:
            best_result = self.analysis_results[best_model_name]
            if best_result['probabilities'] is not None:
                y_test = self.trainer_state['y_test']
                y_proba = best_result['probabilities']
                
                if self.trainer_state['n_classes'] == 2:
                    # Binary classification ROC
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                    roc_auc = auc(fpr, tpr)
                    ax6.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax6.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax6.set_xlim([0.0, 1.0])
                    ax6.set_ylim([0.0, 1.05])
                    ax6.set_xlabel('False Positive Rate')
                    ax6.set_ylabel('True Positive Rate')
                    ax6.set_title(f'ROC Curve - {best_model_name}')
                    ax6.legend(loc="lower right")
                else:
                    # Multi-class: show ROC for most frequent class
                    most_frequent_class = np.bincount(y_test).argmax()
                    y_test_binary = (y_test == most_frequent_class).astype(int)
                    y_proba_class = y_proba[:, most_frequent_class] if y_proba.ndim > 1 else y_proba
                    
                    fpr, tpr, _ = roc_curve(y_test_binary, y_proba_class)
                    roc_auc = auc(fpr, tpr)
                    
                    class_name = self.trainer_state['class_names'][most_frequent_class] if most_frequent_class < len(self.trainer_state['class_names']) else f'Class {most_frequent_class}'
                    
                    ax6.plot(fpr, tpr, color='darkorange', lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
                    ax6.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax6.set_xlim([0.0, 1.0])
                    ax6.set_ylim([0.0, 1.05])
                    ax6.set_xlabel('False Positive Rate')
                    ax6.set_ylabel('True Positive Rate')
                    ax6.set_title(f'ROC Curve - {class_name}')
                    ax6.legend(loc="lower right")
            else:
                ax6.text(0.5, 0.5, 'No probability predictions\navailable for ROC curve', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('ROC Curve - N/A')
                
        except Exception as e:
            ax6.text(0.5, 0.5, f'ROC curve failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('ROC Curve (Failed)')
        
        # 7. Class Distribution
        ax7 = axes[2, 0]
        try:
            class_dist = pd.Series(self.trainer_state['y_test']).value_counts()
            class_labels = [self.trainer_state['class_names'][i] if i < len(self.trainer_state['class_names']) else f'Class {i}' 
                           for i in class_dist.index]
            
            wedges, texts, autotexts = ax7.pie(class_dist.values, labels=class_labels, autopct='%1.1f%%', startangle=90)
            ax7.set_title('Test Set Class Distribution')
            
            # Make text smaller if there are many classes
            if len(class_labels) > 5:
                for text in texts + autotexts:
                    text.set_fontsize(8)
                    
        except Exception as e:
            ax7.text(0.5, 0.5, f'Class distribution failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Class Distribution (Failed)')
        
        # 8. Precision-Recall Curve
        ax8 = axes[2, 1]
        try:
            best_result = self.analysis_results[best_model_name]
            if best_result['probabilities'] is not None:
                y_test = self.trainer_state['y_test']
                y_proba = best_result['probabilities']
                
                if self.trainer_state['n_classes'] == 2:
                    # Binary classification PR curve
                    precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                    pr_auc = auc(recall, precision)
                    ax8.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
                    ax8.set_xlabel('Recall')
                    ax8.set_ylabel('Precision')
                    ax8.set_title(f'Precision-Recall Curve - {best_model_name}')
                    ax8.legend()
                else:
                    # Multi-class: show PR for most frequent class
                    most_frequent_class = np.bincount(y_test).argmax()
                    y_test_binary = (y_test == most_frequent_class).astype(int)
                    y_proba_class = y_proba[:, most_frequent_class] if y_proba.ndim > 1 else y_proba
                    
                    precision, recall, _ = precision_recall_curve(y_test_binary, y_proba_class)
                    pr_auc = auc(recall, precision)
                    
                    class_name = self.trainer_state['class_names'][most_frequent_class] if most_frequent_class < len(self.trainer_state['class_names']) else f'Class {most_frequent_class}'
                    
                    ax8.plot(recall, precision, color='blue', lw=2, label=f'{class_name} (AUC = {pr_auc:.2f})')
                    ax8.set_xlabel('Recall')
                    ax8.set_ylabel('Precision')
                    ax8.set_title(f'Precision-Recall Curve - {class_name}')
                    ax8.legend()
            else:
                ax8.text(0.5, 0.5, 'No probability predictions\navailable for PR curve', 
                        ha='center', va='center', transform=ax8.transAxes)
                ax8.set_title('Precision-Recall Curve - N/A')
                
        except Exception as e:
            ax8.text(0.5, 0.5, f'PR curve failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Precision-Recall Curve (Failed)')
        
        # 9. Error Analysis
        ax9 = axes[2, 2]
        try:
            best_result = self.analysis_results[best_model_name]
            y_test = self.trainer_state['y_test']
            y_pred = best_result['predictions']
            
            # Calculate error rates per class
            error_rates = []
            class_names_for_error = []
            
            for class_idx, class_name in enumerate(self.trainer_state['class_names']):
                class_mask = (y_test == class_idx)
                if class_mask.sum() > 0:
                    class_predictions = y_pred[class_mask]
                    error_rate = (class_predictions != class_idx).mean()
                    error_rates.append(error_rate * 100)  # Convert to percentage
                    class_names_for_error.append(class_name)
            
            if error_rates:
                bars = ax9.bar(range(len(error_rates)), error_rates)
                ax9.set_xticks(range(len(error_rates)))
                ax9.set_xticklabels(class_names_for_error, rotation=45)
                ax9.set_ylabel('Error Rate (%)')
                ax9.set_title('Error Rate by Attack Type')
                
                # Color bars based on error rate
                for i, bar in enumerate(bars):
                    if error_rates[i] < 5:
                        bar.set_color('#2ecc71')  # Green for low error
                    elif error_rates[i] < 15:
                        bar.set_color('#f39c12')  # Orange for medium error
                    else:
                        bar.set_color('#e74c3c')  # Red for high error
            else:
                ax9.text(0.5, 0.5, 'No error data available', 
                        ha='center', va='center', transform=ax9.transAxes)
                ax9.set_title('Error Analysis - N/A')
                
        except Exception as e:
            ax9.text(0.5, 0.5, f'Error analysis failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Error Analysis (Failed)')
        
        plt.tight_layout()
        plt.savefig('part2_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return True

def run_part2():
    """Run Part 2: Model Analysis and Evaluation"""
    log_checkpoint("Starting Part 2: Model Analysis and Evaluation")
    
    analyzer = ModelAnalyzer()
    
    # Check for Part 1 results
    if not os.path.exists("part1_results.pkl"):
        log_checkpoint("ERROR: Part 1 results not found!")
        log_checkpoint("Required file: part1_results.pkl")
        log_checkpoint("Please run Part 1 first.")
        return False
    
    try:
        # Load Part 1 results
        if not analyzer.load_part1_results():
            return False
        
        # Perform detailed model evaluation
        detailed_results = analyzer.detailed_model_evaluation()
        
        if not detailed_results:
            log_checkpoint("No models were successfully evaluated")
            return False
        
        # Generate performance comparison
        enhanced_comparison = analyzer.generate_performance_comparison()
        
        # Analyze per-class performance
        class_df, best_model_name = analyzer.analyze_per_class_performance()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Prepare results for saving
        save_data = {
            'analysis_results': analyzer.analysis_results,
            'trainer_state': analyzer.trainer_state,
            'enhanced_comparison': enhanced_comparison,
            'class_performance': class_df,
            'best_model_name': best_model_name,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models_analyzed': len(detailed_results),
                'best_model_accuracy': enhanced_comparison.iloc[0]['Accuracy'] if enhanced_comparison is not None else None,
                'best_model_f1': enhanced_comparison.iloc[0]['Weighted F1'] if enhanced_comparison is not None else None
            }
        }
        
        # Save results
        with open('part2_results.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        
        # Generate summary report
        print("\n" + "="*80)
        print("PART 2 ANALYSIS SUMMARY")
        print("="*80)
        
        if enhanced_comparison is not None:
            print(f"Models Analyzed: {len(detailed_results)}")
            print(f"Best Model: {best_model_name}")
            print(f"Best Accuracy: {enhanced_comparison.iloc[0]['Accuracy']:.4f}")
            print(f"Best Weighted F1: {enhanced_comparison.iloc[0]['Weighted F1']:.4f}")
            
            # Additional insights
            avg_accuracy = enhanced_comparison['Accuracy'].mean()
            accuracy_std = enhanced_comparison['Accuracy'].std()
            print(f"Average Model Accuracy: {avg_accuracy:.4f} ± {accuracy_std:.4f}")
            
            if class_df is not None and len(class_df) > 0:
                best_attack_detection = class_df.iloc[0]['Attack Type']
                worst_attack_detection = class_df.iloc[-1]['Attack Type']
                print(f"Best Detected Attack Type: {best_attack_detection}")
                print(f"Most Challenging Attack Type: {worst_attack_detection}")
        
        print("\nGenerated Files:")
        print("  - part2_results.pkl (Analysis results)")
        print("  - part2_analysis_results.png (Visualization)")
        
        log_checkpoint("Part 2 completed successfully!")
        return True
        
    except Exception as e:
        log_checkpoint(f"Part 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("XAI Network IDS - Part 2: Model Analysis and Evaluation")
    print("=" * 60)
    
    # Display system information
    print(f"TensorFlow: {'Available' if TF_AVAILABLE else 'Not Available'}")
    print(f"Plotly: {'Available' if PLOTLY_AVAILABLE else 'Not Available'}")
    print()
    
    success = run_part2()
    
    if success:
        print("\n✅ Part 2 completed successfully!")
        print("Ready to proceed to Part 3: XAI Analysis")
    else:
        print("\n❌ Part 2 failed!")
        print("Please check the error messages above.")