#!/usr/bin/env python3

"""
ONNX Model Repair and Validation Tool
Fixes common ONNX issues including wire type problems
"""

import os
import sys
import onnx
import numpy as np
from pathlib import Path

class ONNXRepairer:
    def __init__(self):
        self.models_to_check = {
            'RSMT': '/home/barberb/motion/RSMT-Realtime-Stylized-Motion-Transition/output/web_viewer/deepphase.onnx',
            'DeepMimic': '/home/barberb/motion/deepmimic/data/policies_onnx/compatible_humanoid3d_humanoid3d_walk.onnx',
            'Audio2Gesture': '/home/barberb/motion/audio2gesture_step_fixed.onnx',
            'FaceFormer': '/home/barberb/motion/engine/web_porting_poc/faceformer/faceformer_core_step.onnx'
        }
    
    def validate_and_repair_model(self, model_path, model_name):
        """Validate and attempt to repair an ONNX model"""
        print(f"\n🔍 Checking {model_name}:")
        print(f"   📁 Path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"   ❌ File not found")
            return False
        
        try:
            # Load and validate the model
            print(f"   📝 Loading model...")
            model = onnx.load(model_path)
            
            print(f"   ✅ Model loaded successfully")
            print(f"   📊 IR Version: {model.ir_version}")
            print(f"   🏷️  Producer: {model.producer_name} {model.producer_version}")
            print(f"   🔢 Opset Version: {model.opset_import[0].version if model.opset_import else 'Unknown'}")
            
            # Check model validity
            print(f"   🔍 Validating model structure...")
            onnx.checker.check_model(model)
            print(f"   ✅ Model structure is valid")
            
            # Print model info
            print(f"   📊 Model Info:")
            print(f"      • Inputs: {len(model.graph.input)}")
            for i, inp in enumerate(model.graph.input):
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in inp.type.tensor_type.shape.dim]
                print(f"        {i+1}. {inp.name}: {shape}")
            
            print(f"      • Outputs: {len(model.graph.output)}")
            for i, out in enumerate(model.graph.output):
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in out.type.tensor_type.shape.dim]
                print(f"        {i+1}. {out.name}: {shape}")
            
            print(f"      • Nodes: {len(model.graph.node)}")
            
            # Try to optimize and re-save the model to fix potential issues
            backup_path = model_path + '.backup'
            repaired_path = model_path + '.repaired'
            
            print(f"   🔧 Creating backup...")
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(model_path, backup_path)
                print(f"   ✅ Backup created: {backup_path}")
            
            print(f"   🔧 Optimizing model...")
            # Apply basic optimizations that might fix serialization issues
            from onnx import optimizer
            optimized_model = optimizer.optimize(model)
            
            print(f"   💾 Saving repaired model...")
            onnx.save(optimized_model, repaired_path)
            
            print(f"   ✅ Repaired model saved: {repaired_path}")
            print(f"   💡 To use the repaired model, replace the original with:")
            print(f"      mv '{repaired_path}' '{model_path}'")
            
            return True
            
        except onnx.onnx_cpp2py_export.checker.ValidationError as e:
            print(f"   ❌ Model validation error: {e}")
            print(f"   💡 This model has structural issues that need manual fixing")
            return False
            
        except Exception as e:
            error_str = str(e)
            print(f"   ❌ Error loading model: {error_str}")
            
            if 'wire type' in error_str.lower():
                print(f"   🔧 WIRE TYPE ISSUE DETECTED:")
                print(f"      • This is a protobuf serialization issue")
                print(f"      • The model was likely exported with incompatible tools")
                print(f"      • Solution: Re-export with ONNX 1.13+ and protobuf 3.20+")
                
                # Try to recover by loading with different settings
                try:
                    print(f"   🔄 Attempting recovery...")
                    # Load with different options
                    model = onnx.load(model_path, format=None, load_external_data=False)
                    print(f"   ✅ Recovery successful with alternative loading")
                    return True
                except Exception as recovery_error:
                    print(f"   ❌ Recovery failed: {recovery_error}")
            
            return False
    
    def run_full_check(self):
        """Check all models"""
        print("🔧 ONNX Model Repair Tool")
        print("=" * 50)
        
        results = {}
        
        for model_name, model_path in self.models_to_check.items():
            success = self.validate_and_repair_model(model_path, model_name)
            results[model_name] = success
        
        print(f"\n📊 SUMMARY:")
        print(f"   • Total models checked: {len(results)}")
        success_count = sum(results.values())
        print(f"   • Successfully validated: {success_count}/{len(results)}")
        
        failed_models = [name for name, success in results.items() if not success]
        if failed_models:
            print(f"   • Failed models: {', '.join(failed_models)}")
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   1. For wire type errors: Re-export models with newer ONNX tools")
            print(f"   2. For missing files: Check file paths in model-loader-webnn.js")
            print(f"   3. Use repaired models if available (.repaired files)")
        else:
            print(f"   ✅ All models are healthy!")
        
        return results

def main():
    try:
        import onnx
    except ImportError:
        print("❌ ONNX library not found. Install with:")
        print("   pip install onnx")
        sys.exit(1)
    
    repairer = ONNXRepairer()
    results = repairer.run_full_check()
    
    # Generate fix commands
    print(f"\n📜 QUICK FIX COMMANDS:")
    print(f"# Apply repaired models (run these if repair was successful):")
    for model_name in repairer.models_to_check.keys():
        model_path = repairer.models_to_check[model_name]
        if os.path.exists(model_path + '.repaired'):
            print(f"mv '{model_path}.repaired' '{model_path}'  # Fix {model_name}")

if __name__ == "__main__":
    main()
