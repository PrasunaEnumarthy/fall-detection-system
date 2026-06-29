import onnxruntime as ort
import numpy as np
import pandas as pd

# Load model
session = ort.InferenceSession('models/model.onnx')
print('Model loaded successfully')

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Get 5 fall windows
fall_windows = df[df['fall_label'] == 1].head(5)
print('\n=== TESTING ON 5 FALL WINDOWS ===')

for i, (_, row) in enumerate(fall_windows.iterrows()):
    features = row[[f'f_{j}' for j in range(1200)]].values.astype('float32')
    window = features.reshape(1, 200, 6)
    
    outputs = session.run(None, {'imu_window': window})
    fall_prob = float(outputs[0][0][0])
    fall_type_idx = int(np.argmax(outputs[1][0]))
    pre_act_idx = int(np.argmax(outputs[2][0]))
    
    fall_types = ['slip', 'trip', 'faint']
    pre_acts = ['walking', 'standing', 'bending', 'sitting']
    
    print(f'Window {i+1}: fall_prob={fall_prob:.3f} | type={fall_types[fall_type_idx]} | activity={pre_acts[pre_act_idx]} | actual_type={row["fall_type"]}')

print('\n=== TESTING ON 5 ADL WINDOWS (should NOT detect fall) ===')
adl_windows = df[df['fall_label'] == 0].head(5)

for i, (_, row) in enumerate(adl_windows.iterrows()):
    features = row[[f'f_{j}' for j in range(1200)]].values.astype('float32')
    window = features.reshape(1, 200, 6)
    outputs = session.run(None, {'imu_window': window})
    fall_prob = float(outputs[0][0][0])
    result = 'FALSE ALARM ⚠' if fall_prob > 0.5 else 'CORRECT ✓'
    print(f'Window {i+1}: fall_prob={fall_prob:.3f} | {result}')