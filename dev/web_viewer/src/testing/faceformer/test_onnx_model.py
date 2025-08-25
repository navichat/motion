import onnxruntime as ort
import numpy as np
import json

# Load the model
print("Loading ONNX model...")
session = ort.InferenceSession('faceformer_core_step.onnx')

# Print model info
print("\nModel inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")

print("\nModel outputs:")
for out in session.get_outputs():
    print(f"  {out.name}: {out.shape} ({out.type})")

# Load sample data
with open('faceformer_sample_data.json', 'r') as f:
    sample_data = json.load(f)

core_data = sample_data['core_step']

# Prepare inputs
audio_features = np.array(core_data['audio_features'], dtype=np.float32)
vertice_emb = np.array(core_data['vertice_emb'], dtype=np.float32)
one_hot = np.array(core_data['one_hot'], dtype=np.float32)
template = np.array(core_data['template'], dtype=np.float32)

print(f"\nInput shapes:")
print(f"  audio_features: {audio_features.shape}")
print(f"  vertice_emb: {vertice_emb.shape}")
print(f"  one_hot: {one_hot.shape}")
print(f"  template: {template.shape}")

# Run inference
print("\nRunning inference...")
try:
    outputs = session.run(None, {
        'audio_features': audio_features,
        'vertice_emb': vertice_emb,
        'one_hot': one_hot,
        'template': template
    })
    
    print("Inference successful!")
    print(f"Output shapes:")
    for i, out in enumerate(session.get_outputs()):
        print(f"  {out.name}: {outputs[i].shape}")
    
    # Save successful test data for JavaScript
    test_data = {
        'inputs': {
            'audio_features': audio_features.tolist(),
            'vertice_emb': vertice_emb.tolist(),
            'one_hot': one_hot.tolist(),
            'template': template.tolist()
        },
        'outputs': {
            session.get_outputs()[0].name: outputs[0].tolist(),
            session.get_outputs()[1].name: outputs[1].tolist()
        }
    }
    
    with open('onnx_test_data.json', 'w') as f:
        json.dump(test_data, f)
    print("Test data saved to onnx_test_data.json")
        
except Exception as e:
    print(f"Inference failed: {e}")
    import traceback
    traceback.print_exc()
