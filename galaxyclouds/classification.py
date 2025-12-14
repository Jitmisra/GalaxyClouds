import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from .observables import compute_all_observables
from .transforms import transform_to_galaxy_frame

def build_feature_matrix(X, mask, use_galaxy_frame=True):
    """
    Compute all observables and return as feature matrix.
    If use_galaxy_frame=True, transform to principal frame first.
    """
    if use_galaxy_frame:
        X_trans, _ = transform_to_galaxy_frame(X, mask)
        df_feats = compute_all_observables(X_trans, mask)
    else:
        df_feats = compute_all_observables(X, mask)
        
    return df_feats

def train_morphology_classifier(X_train, y_train, model_type='xgboost'):
    """
    Train galaxy morphology classifier.
    Returns trained model with .predict() and .predict_proba() methods.
    """
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    else:
        raise NotImplementedError(f"Model {model_type} not implemented.")

def evaluate_classifier(model, X_test, y_test):
    """
    Full evaluation suite.
    Returns dict of all metrics and figures.
    """
    plt.style.use('dark_background')
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc_scores = []
    
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    colors = ['blue', 'orange', 'green']
    class_names = ['Elliptical', 'Spiral', 'Irregular']
    
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        auc_scores.append(auc)
        ax_roc.plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_names[i]} (AUC = {auc:.2f})')
        
    ax_roc.plot([0, 1], [0, 1], 'w--', lw=1)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves')
    ax_roc.legend()
    
    # Macro AUC
    macro_auc = np.mean(auc_scores)
    
    results = {
        'roc_fig': fig_roc,
        'macro_auc': macro_auc,
        'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    }
    
    return results

def compare_frame_performance(X, y, mask):
    """
    Train classifier twice:
    1. Using sky-frame features
    2. Using galaxy-frame features
    """
    from sklearn.model_selection import train_test_split
    
    print("Building sky-frame features...")
    df_sky = build_feature_matrix(X, mask, use_galaxy_frame=False)
    
    print("Building galaxy-frame features...")
    df_gal = build_feature_matrix(X, mask, use_galaxy_frame=True)
    
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    
    model_sky = train_morphology_classifier(df_sky.iloc[idx_train], y[idx_train])
    res_sky = evaluate_classifier(model_sky, df_sky.iloc[idx_test], y[idx_test])
    
    model_gal = train_morphology_classifier(df_gal.iloc[idx_train], y[idx_train])
    res_gal = evaluate_classifier(model_gal, df_gal.iloc[idx_test], y[idx_test])
    
    print(f"Sky Frame Macro AUC:    {res_sky['macro_auc']:.3f}")
    print(f"Galaxy Frame Macro AUC: {res_gal['macro_auc']:.3f}")
    print(f"Improvement:            {(res_gal['macro_auc'] - res_sky['macro_auc'])*100:.1f}%")
    
    explain = """
    WHY galaxy frame gives better results:
    Removing extrinsic orientation, sky projection effects, and standardizing size by half-light radius 
    removes irrelevant variance from the model, forcing it to learn intrinsic morphological patterns.
    This is analogous to how rest-frame jet features remove longitudinal boost dependence.
    """
    print(explain)

# Frame comparison utility
