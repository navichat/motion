import onnxruntime as ort
import numpy as np
import json

try:
    print("Loading session...")
    session = ort.InferenceSession('faceformer_core_step.onnx')
    print("Session loaded successfully")
    
    print("Loading test data...")
    with open('onnx_test_data.json', 'r') as f:
        test_data = json.load(f)
    print("Test data loaded successfully")

    inputs = test_data['inputs']
    audio_features = np.array(inputs['audio_features'], dtype=np.float32)
    template = np.array(inputs['template'], dtype=np.float32)
    one_hot = np.array(inputs['one_hot'], dtype=np.float32)

    # Start with initial embedding
    current_emb = np.array(inputs['vertice_emb'], dtype=np.float32)
    print('Initial embedding shape:', current_emb.shape)

    generated_steps = []
    for step in range(3):
        print(f'\nStep {step + 1}:')
        print(f'  Current embedding shape: {current_emb.shape}')
        
        try:
            result = session.run(None, {
                'audio_features': audio_features,
                'vertice_emb': current_emb,
                'one_hot': one_hot,
                'template': template
            })
            
            new_out = result[0]
            updated_emb = result[1]
            
            print(f'  New output shape: {new_out.shape}')
            print(f'  Updated embedding shape: {updated_emb.shape}')
            print(f'  Output range: [{new_out.min():.4f}, {new_out.max():.4f}]')
            
            generated_steps.append(new_out)
            current_emb = updated_emb
            
        except Exception as e:
            print(f'  Error in step {step + 1}: {e}')
            break
        
    print(f'\nSuccessfully generated {len(generated_steps)} steps!')
    
    # Save multi-step test data
    multi_step_data = {
        'initial_inputs': {
            'audio_features': audio_features.tolist(),
            'template': template.tolist(),
            'one_hot': one_hot.tolist(),
            'initial_vertice_emb': inputs['vertice_emb']
        },
        'steps': []
    }
    
    # Re-run to capture all steps
    current_emb = np.array(inputs['vertice_emb'], dtype=np.float32)
    for step in range(len(generated_steps)):
        result = session.run(None, {
            'audio_features': audio_features,
            'vertice_emb': current_emb,
            'one_hot': one_hot,
            'template': template
        })
        
        multi_step_data['steps'].append({
            'step': step + 1,
            'input_vertice_emb': current_emb.tolist(),
            'output_new_vertice': result[0].tolist(),
            'output_updated_emb': result[1].tolist(),
            'input_shape': list(current_emb.shape),
            'output_new_shape': list(result[0].shape),
            'output_updated_shape': list(result[1].shape)
        })
        
        current_emb = result[1]
    
    with open('multi_step_test_data.json', 'w') as f:
        json.dump(multi_step_data, f)
    print("Multi-step test data saved to multi_step_test_data.json")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
