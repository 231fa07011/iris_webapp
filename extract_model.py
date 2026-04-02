import joblib
import json
import os
import numpy as np

def to_standard(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [to_standard(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_standard(value) for key, value in obj.items()}
    else:
        return obj

def extract():
    model = joblib.load('iris_model.joblib')
    try:
        info = joblib.load('model_info.joblib')
    except:
        info = {}
    
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        # Tree attributes
        trees.append({
            'children_left': tree.children_left.tolist(),
            'children_right': tree.children_right.tolist(),
            'feature': tree.feature.tolist(),
            'threshold': tree.threshold.tolist(),
            'value': tree.value.tolist()
        })
        
    data = {
        'trees': trees,
        'target_names': info.get('target_names', ['setosa', 'versicolor', 'virginica']),
        'feature_names': info.get('feature_names', ['sepal length', 'sepal width', 'petal length', 'petal width'])
    }
    
    final_data = to_standard(data)
    
    with open('model_data.json', 'w') as f:
        json.dump(final_data, f)
    print("✅ Model data extracted successfully.")

if __name__ == "__main__":
    extract()
