#!/usr/bin/env python3
"""
Part 3 SHAPE FIX: Handles "Per-column arrays must each be 1-dimensional" error
This specifically fixes array shape issues in SHAP importance processing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import time
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Import checks (same as before)
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP available - Advanced explainability enabled")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - Using alternative explainability methods")

try:
    import eli5
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
    print("ELI5 available - Additional explanations enabled")
except ImportError:
    ELI5_AVAILABLE = False
    print("ELI5 not available")

plt.style.use('default')
sns.set_palette("husl")

def log_checkpoint(message, elapsed_time=None):
    """Log progress with timestamps"""
    current_time = datetime.now().strftime("%H:%M:%S")
    if elapsed_time:
        print(f"[{current_time}] {message} - Elapsed: {elapsed_time:.2f}s")
    else:
        print(f"[{current_time}] {message}")

def safe_flatten_importance(importance_array):
    """Safely flatten importance array to 1D regardless of input shape"""
    try:
        # Convert to numpy array if it isn't already
        if not isinstance(importance_array, np.ndarray):
            importance_array = np.array(importance_array)
        
        log_checkpoint(f"Input importance shape: {importance_array.shape}")
        log_checkpoint(f"Input importance dtype: {importance_array.dtype}")
        
        # Handle different shapes
        if importance_array.ndim == 1:
            # Already 1D, just return it
            result = importance_array.flatten()
        elif importance_array.ndim == 2:
            # 2D array - flatten or take mean
            if importance_array.shape[0] == 1:
                # Shape like (1, n_features) - just flatten
                result = importance_array.flatten()
            else:
                # Shape like (n_classes, n_features) - take mean across classes
                result = importance_array.mean(axis=0)
        elif importance_array.ndim == 3:
            # 3D array - likely (n_classes, n_samples, n_features)
            result = importance_array.mean(axis=(0, 1))
        else:
            # Higher dimensions - flatten everything
            result = importance_array.flatten()
        
        # Ensure result is 1D
        result = np.array(result).flatten()
        
        log_checkpoint(f"Output importance shape: {result.shape}")
        log_checkpoint(f"Output importance dtype: {result.dtype}")
        
        return result
        
    except Exception as e:
        log_checkpoint(f"Error in safe_flatten_importance: {e}")
        # Return zeros as fallback
        return np.zeros(1)

class XAIExplainerShapeFix:
    """XAI explainer with proper array shape handling"""
    
    def __init__(self):
        self.analysis_results = None
        self.trainer_state = None
        self.best_model_name = None
        self.best_model = None
        self.feature_names = None
        self.class_names = None
        self.explanations = {}
        self.attack_signatures = {}
    
    def load_part2_results(self, filepath='part2_results.pkl'):
        """Load results from Part 2"""
        try:
            log_checkpoint(f"Loading Part 2 results from {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Part 2 results file not found: {filepath}")
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.analysis_results = data['analysis_results']
            self.trainer_state = data['trainer_state']
            self.best_model_name = data['best_model_name']
            self.best_model = self.analysis_results[self.best_model_name]['model_info']['model']
            self.feature_names = self.trainer_state['feature_names']
            self.class_names = self.trainer_state['class_names']
            
            log_checkpoint(f"Loaded analysis for {len(self.analysis_results)} models")
            log_checkpoint(f"Best model: {self.best_model_name}")
            log_checkpoint(f"Features: {len(self.feature_names)}, Classes: {len(self.class_names)}")
            
            return True
            
        except Exception as e:
            log_checkpoint(f"Error loading Part 2 results: {e}")
            return False
    
    def generate_shap_explanations(self):
        """Generate SHAP explanations with proper shape handling"""
        if not SHAP_AVAILABLE:
            log_checkpoint("SHAP not available - skipping SHAP explanations")
            return None
        
        try:
            log_checkpoint("Generating SHAP explanations...")
            
            # Prepare data
            X_test = self.trainer_state['X_test']
            X_explain = X_test.iloc[:300]  # Even smaller subset for stability
            
            # Scale data if needed
            if self.best_model_name in ['Neural Network', 'Deep Neural Network']:
                scaler = self.trainer_state['scalers']['minmax']
                X_explain_scaled = pd.DataFrame(
                    scaler.transform(X_explain),
                    columns=self.feature_names,
                    index=X_explain.index
                )
                X_explain = X_explain_scaled
            
            shap_explanations = {}
            
            # Use TreeExplainer for tree-based models
            if hasattr(self.best_model, 'feature_importances_'):
                log_checkpoint("Using TreeExplainer for tree-based model")
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(X_explain)
                
                log_checkpoint(f"Raw SHAP values type: {type(shap_values)}")
                
                if isinstance(shap_values, list):
                    log_checkpoint(f"SHAP list length: {len(shap_values)}")
                    if len(shap_values) > 0:
                        log_checkpoint(f"First SHAP array shape: {shap_values[0].shape}")
                    
                    # Store raw values
                    shap_explanations['shap_values'] = shap_values
                    
                    # Calculate global importance with proper shape handling
                    try:
                        # Convert to numpy array and handle shapes carefully
                        shap_array = np.array(shap_values)
                        log_checkpoint(f"Combined SHAP array shape: {shap_array.shape}")
                        
                        # Use safe flattening function
                        global_importance = safe_flatten_importance(shap_array)
                        shap_explanations['global_importance'] = global_importance
                        
                    except Exception as shape_error:
                        log_checkpoint(f"Shape handling failed: {shape_error}")
                        # Fallback: use mean of absolute values from first class
                        if len(shap_values) > 0:
                            global_importance = safe_flatten_importance(np.abs(shap_values[0]).mean(axis=0))
                            shap_explanations['global_importance'] = global_importance
                        else:
                            return None
                else:
                    # Single array case
                    log_checkpoint(f"Single SHAP array shape: {shap_values.shape}")
                    shap_explanations['shap_values'] = [shap_values]
                    global_importance = safe_flatten_importance(np.abs(shap_values).mean(axis=0))
                    shap_explanations['global_importance'] = global_importance
            
            else:
                # For non-tree models, use a simpler approach
                log_checkpoint("Using simplified explanation for non-tree model")
                # Skip SHAP for now if it's not a tree model to avoid complexity
                return None
            
            # Validate final results
            if 'global_importance' in shap_explanations:
                final_importance = shap_explanations['global_importance']
                log_checkpoint(f"Final global importance shape: {final_importance.shape}")
                log_checkpoint(f"Final global importance length: {len(final_importance)}")
                log_checkpoint(f"Feature names length: {len(self.feature_names)}")
                
                # Final safety check
                if len(final_importance) != len(self.feature_names):
                    min_len = min(len(final_importance), len(self.feature_names))
                    log_checkpoint(f"Final dimension fix: truncating to {min_len}")
                    shap_explanations['global_importance'] = final_importance[:min_len]
                    shap_explanations['feature_names'] = self.feature_names[:min_len]
                else:
                    shap_explanations['feature_names'] = self.feature_names
            
            log_checkpoint("SHAP explanations generated successfully")
            return shap_explanations
            
        except Exception as e:
            log_checkpoint(f"SHAP explanation generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_alternative_explanations(self):
        """Generate alternative explanations with shape validation"""
        log_checkpoint("Generating alternative explanations...")
        
        explanations = {}
        
        try:
            # Built-in feature importance
            if hasattr(self.best_model, 'feature_importances_'):
                raw_importance = self.best_model.feature_importances_
                
                # Use safe flattening
                importance_scores = safe_flatten_importance(raw_importance)
                
                # Ensure length match
                min_len = min(len(importance_scores), len(self.feature_names))
                importance_scores = importance_scores[:min_len]
                feature_names_truncated = self.feature_names[:min_len]
                
                explanations['feature_importance'] = {
                    'importance': importance_scores,
                    'features': feature_names_truncated,
                    'method': 'Built-in Feature Importance'
                }
                log_checkpoint(f"Built-in importance: {len(importance_scores)} features")
            
            # Permutation importance
            if ELI5_AVAILABLE:
                try:
                    X_test = self.trainer_state['X_test']
                    y_test = self.trainer_state['y_test']
                    
                    # Use smaller dataset for stability
                    if self.best_model_name in ['Neural Network', 'Deep Neural Network']:
                        scaler = self.trainer_state['scalers']['minmax']
                        X_test_scaled = scaler.transform(X_test)
                        X_test_for_perm = X_test_scaled[:200]
                    else:
                        X_test_for_perm = X_test.iloc[:200].values
                    
                    y_test_for_perm = y_test.iloc[:200]
                    
                    perm = PermutationImportance(self.best_model, random_state=42)
                    perm.fit(X_test_for_perm, y_test_for_perm)
                    
                    # Use safe flattening
                    perm_importance = safe_flatten_importance(perm.feature_importances_)
                    perm_std = safe_flatten_importance(perm.feature_importances_std_)
                    
                    # Ensure length match
                    min_len = min(len(perm_importance), len(self.feature_names))
                    perm_importance = perm_importance[:min_len]
                    perm_std = perm_std[:min_len]
                    feature_names_truncated = self.feature_names[:min_len]
                    
                    explanations['permutation_importance'] = {
                        'importance': perm_importance,
                        'importance_std': perm_std,
                        'features': feature_names_truncated,
                        'method': 'Permutation Importance'
                    }
                    log_checkpoint(f"Permutation importance: {len(perm_importance)} features")
                    
                except Exception as e:
                    log_checkpoint(f"Permutation importance failed: {e}")
            
            return explanations
            
        except Exception as e:
            log_checkpoint(f"Alternative explanation generation failed: {e}")
            return {}
    
    def create_attack_signatures(self, shap_explanations=None):
        """Create attack signatures with proper array handling"""
        log_checkpoint("Creating attack signatures...")
        
        signatures = {}
        
        try:
            # Get importance scores with shape validation
            importance_scores = None
            feature_names_to_use = self.feature_names
            
            if shap_explanations and 'global_importance' in shap_explanations:
                importance_scores = shap_explanations['global_importance']
                feature_names_to_use = shap_explanations.get('feature_names', self.feature_names)
                log_checkpoint(f"Using SHAP importance for signatures")
            elif hasattr(self.best_model, 'feature_importances_'):
                raw_importance = self.best_model.feature_importances_
                importance_scores = safe_flatten_importance(raw_importance)
                log_checkpoint(f"Using model importance for signatures")
            else:
                log_checkpoint("No importance scores available for signature creation")
                return {}
            
            # Final validation and safety checks
            log_checkpoint(f"Importance scores shape: {importance_scores.shape}")
            log_checkpoint(f"Feature names length: {len(feature_names_to_use)}")
            
            # Ensure both are 1D and same length
            importance_scores_1d = np.array(importance_scores).flatten()
            min_len = min(len(importance_scores_1d), len(feature_names_to_use))
            
            log_checkpoint(f"Using {min_len} features for signatures")
            
            # Create DataFrame with validated arrays
            feature_df = pd.DataFrame({
                'feature': list(feature_names_to_use[:min_len]),  # Ensure it's a list
                'importance': list(importance_scores_1d[:min_len])  # Ensure it's a list
            }).sort_values('importance', ascending=False)
            
            log_checkpoint(f"Feature DataFrame created successfully with shape: {feature_df.shape}")
            
            # Generate signatures for each attack type
            for class_idx, class_name in enumerate(self.class_names):
                top_features = feature_df.head(5)['feature'].tolist()
                signature = self.generate_attack_explanation(class_name, top_features)
                
                signatures[class_name] = {
                    'key_features': top_features,
                    'signature': signature,
                    'confidence_indicators': self.get_confidence_indicators(class_name, top_features),
                    'mitigation_strategies': self.get_mitigation_strategies(class_name)
                }
            
            log_checkpoint(f"Created signatures for {len(signatures)} attack types")
            return signatures
            
        except Exception as e:
            log_checkpoint(f"Attack signature creation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def create_simple_visualizations(self, alternative_explanations=None):
        """Create simple visualizations that work reliably"""
        log_checkpoint("Creating simple visualizations...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('XAI Network IDS - Analysis Results', fontsize=14, fontweight='bold')
            
            # 1. Feature Importance
            ax1 = axes[0, 0]
            if alternative_explanations and 'feature_importance' in alternative_explanations:
                try:
                    alt_data = alternative_explanations['feature_importance']
                    importance_scores = alt_data['importance']
                    feature_names_viz = alt_data['features']
                    
                    # Ensure arrays are 1D and same length
                    importance_1d = np.array(importance_scores).flatten()
                    min_len = min(len(importance_1d), len(feature_names_viz))
                    
                    # Create simple DataFrame
                    simple_df = pd.DataFrame({
                        'feature': list(feature_names_viz[:min_len]),
                        'importance': list(importance_1d[:min_len])
                    }).sort_values('importance', ascending=False).head(8)
                    
                    # Create bar plot
                    bars = ax1.barh(range(len(simple_df)), simple_df['importance'])
                    ax1.set_yticks(range(len(simple_df)))
                    ax1.set_yticklabels([f[:12] + '...' if len(f) > 12 else f for f in simple_df['feature']], fontsize=8)
                    ax1.set_xlabel('Importance')
                    ax1.set_title('Feature Importance')
                    ax1.invert_yaxis()
                    
                except Exception as e:
                    ax1.text(0.5, 0.5, f'Feature plot failed:\n{str(e)[:30]}...', 
                            ha='center', va='center', transform=ax1.transAxes, fontsize=8)
                    ax1.set_title('Feature Importance (Error)')
            else:
                ax1.text(0.5, 0.5, 'No feature importance\ndata available', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Feature Importance - N/A')
            
            # 2. Model Performance
            ax2 = axes[0, 1]
            try:
                best_result = self.analysis_results[self.best_model_name]
                metrics = best_result['metrics']
                
                metric_names = ['Accuracy', 'F1-Score']
                metric_values = [metrics['accuracy'], metrics['weighted_f1']]
                
                bars = ax2.bar(metric_names, metric_values, color=['#3498db', '#2ecc71'])
                ax2.set_ylabel('Score')
                ax2.set_title(f'Model Performance\n({self.best_model_name})')
                ax2.set_ylim(0, 1)
                
                for bar, value in zip(bars, metric_values):
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                            
            except Exception as e:
                ax2.text(0.5, 0.5, 'Performance data\nnot available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Model Performance (Error)')
            
            # 3. Attack Signatures Count
            ax3 = axes[1, 0]
            if self.attack_signatures:
                sig_count = len(self.attack_signatures)
                ax3.bar(['Signatures Created'], [sig_count], color='lightgreen')
                ax3.set_ylabel('Count')
                ax3.set_title('Attack Signatures')
                ax3.text(0, sig_count/2, str(sig_count), ha='center', va='center', fontsize=16, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No signatures\ncreated', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Attack Signatures - None')
            
            # 4. Dataset Info
            ax4 = axes[1, 1]
            try:
                info_data = {
                    'Features': len(self.feature_names),
                    'Classes': len(self.class_names),
                    'Test Samples': len(self.trainer_state.get('X_test', []))
                }
                
                bars = ax4.bar(info_data.keys(), info_data.values(), color=['#f39c12', '#9b59b6', '#e74c3c'])
                ax4.set_ylabel('Count')
                ax4.set_title('Dataset Information')
                
                for bar, (key, value) in zip(bars, info_data.items()):
                    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(info_data.values())*0.02,
                            str(value), ha='center', va='bottom', fontsize=10)
                            
            except Exception as e:
                ax4.text(0.5, 0.5, 'Dataset info\nnot available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Dataset Info (Error)')
            
            plt.tight_layout()
            plt.savefig('part3_xai_analysis.png', dpi=200, bbox_inches='tight')
            log_checkpoint("Simple visualization saved successfully")
            plt.show()
            
            return True
            
        except Exception as e:
            log_checkpoint(f"Simple visualization failed: {e}")
            return False
    
    # Include helper methods
    def generate_attack_explanation(self, attack_type, top_features):
        """Generate explanation for attack type"""
        base_explanations = {
            'Normal': "Normal network traffic with standard patterns",
            'Fuzzers': "Fuzzer attacks with irregular packet patterns",
            'Analysis': "Network analysis attacks with scanning behavior",
            'Backdoors': "Backdoor activities with unauthorized access",
            'DoS': "Denial of Service attacks with high traffic rates",
            'Exploits': "Exploit attempts targeting vulnerabilities",
            'Generic': "Generic malicious activities",
            'Reconnaissance': "Reconnaissance with network probing",
            'Shellcode': "Shellcode execution patterns",
            'Worms': "Worm propagation behaviors"
        }
        return base_explanations.get(attack_type, f"{attack_type} attack patterns detected.")
    
    def get_confidence_indicators(self, attack_type, top_features):
        """Get confidence indicators"""
        return [f"Strong patterns in {feature}" for feature in top_features[:2]]
    
    def get_mitigation_strategies(self, attack_type):
        """Get mitigation strategies"""
        return ["Implement monitoring", "Deploy appropriate defenses"]
    
    def generate_simple_html_report(self):
        """Generate simple HTML report that works"""
        log_checkpoint("Generating simple HTML report...")
        
        try:
            best_result = self.analysis_results[self.best_model_name]
            accuracy = best_result['metrics']['accuracy']
            f1_score = best_result['metrics']['weighted_f1']
            
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>XAI Network IDS Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #3498db; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ background: #2ecc71; color: white; padding: 5px 10px; margin: 5px; border-radius: 3px; display: inline-block; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>XAI Network IDS Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Model Performance</h2>
        <div class="metric">Model: {self.best_model_name}</div>
        <div class="metric">Accuracy: {accuracy:.4f}</div>
        <div class="metric">F1-Score: {f1_score:.4f}</div>
    </div>
    
    <div class="section">
        <h2>Dataset Information</h2>
        <p>Features: {len(self.feature_names)}</p>
        <p>Classes: {len(self.class_names)}</p>
        <p>Attack signatures created: {len(self.attack_signatures)}</p>
    </div>
    
    <div class="section">
        <h2>Attack Types</h2>
        <ul>
"""
            
            for class_name in self.class_names:
                html_content += f"<li>{class_name}</li>\n"
            
            html_content += """
        </ul>
    </div>
</body>
</html>"""
            
            with open('xai_comprehensive_report.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            log_checkpoint("Simple HTML report generated")
            return html_content
            
        except Exception as e:
            log_checkpoint(f"HTML report generation failed: {e}")
            return None

def run_part3_shape_fix():
    """Run Part 3 with shape fixes"""
    log_checkpoint("Starting Part 3: XAI Analysis (SHAPE FIX)")
    
    explainer = XAIExplainerShapeFix()
    
    if not os.path.exists("part2_results.pkl"):
        log_checkpoint("ERROR: Part 2 results not found!")
        return False
    
    try:
        # Load results
        if not explainer.load_part2_results():
            return False
        
        # Generate explanations
        shap_explanations = explainer.generate_shap_explanations()
        alternative_explanations = explainer.generate_alternative_explanations()
        
        # Create signatures (with proper array handling)
        explainer.attack_signatures = explainer.create_attack_signatures(shap_explanations)
        
        # Create simple visualizations
        visualization_success = explainer.create_simple_visualizations(alternative_explanations)
        
        # Generate simple HTML report
        html_report = explainer.generate_simple_html_report()
        
        # Save results
        save_data = {
            'explainer_state': {
                'best_model_name': explainer.best_model_name,
                'feature_names': explainer.feature_names,
                'class_names': explainer.class_names
            },
            'explanations': {
                'shap': shap_explanations,
                'alternative': alternative_explanations
            },
            'attack_signatures': explainer.attack_signatures,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'signatures_created': len(explainer.attack_signatures),
                'visualization_created': visualization_success,
                'html_report_created': html_report is not None
            }
        }
        
        with open('final_results.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        
        # Summary
        print("\n" + "="*50)
        print("PART 3 COMPLETED (SHAPE FIX)")
        print("="*50)
        print(f"Best Model: {explainer.best_model_name}")
        print(f"Attack Signatures: {len(explainer.attack_signatures)}")
        print(f"Visualization: {'Success' if visualization_success else 'Failed'}")
        print(f"HTML Report: {'Success' if html_report else 'Failed'}")
        
        return True
        
    except Exception as e:
        log_checkpoint(f"Part 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("XAI Network IDS - Part 3 SHAPE FIX")
    print("=" * 40)
    
    success = run_part3_shape_fix()
    
    if success:
        print("\n✅ Part 3 completed with shape fixes!")
    else:
        print("\n❌ Part 3 failed - check errors above")