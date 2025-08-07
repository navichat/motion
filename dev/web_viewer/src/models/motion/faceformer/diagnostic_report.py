#!/usr/bin/env python3
"""
FaceFormer Model Comparison Summary and Diagnosis
Analyzes the comparison results and provides recommendations
"""

import json
import os
import numpy as np

def load_comparison_results():
    """Load all comparison result files"""
    results = {}
    
    # Load Python model outputs
    python_file = "python_model_outputs.json"
    if os.path.exists(python_file):
        with open(python_file, 'r') as f:
            results['python'] = json.load(f)
        print("✅ Python model outputs loaded")
    else:
        print("❌ Python model outputs not found")
        return None
    
    # Load Node.js comparison results
    node_file = "node_comparison_results.json"
    if os.path.exists(node_file):
        with open(node_file, 'r') as f:
            results['node'] = json.load(f)
        print("✅ Node.js comparison results loaded")
    else:
        print("⚠️ Node.js comparison results not found")
    
    return results

def analyze_differences(python_outputs, js_outputs):
    """Analyze the differences between Python and JS outputs"""
    analysis = {}
    
    # Analyze new_vertice_out
    if 'new_vertice_out' in python_outputs and 'new_vertice_out' in js_outputs:
        py_vertices = np.array(python_outputs['new_vertice_out'])
        js_vertices = np.array(js_outputs['new_vertice_out'])
        
        diff = np.abs(py_vertices - js_vertices)
        analysis['vertices'] = {
            'max_diff': float(np.max(diff)),
            'mean_diff': float(np.mean(diff)),
            'std_diff': float(np.std(diff)),
            'num_large_diffs': int(np.sum(diff > 0.01)),
            'percent_large_diffs': float(np.sum(diff > 0.01) / len(diff) * 100)
        }
    
    # Analyze updated_vertice_emb
    if 'updated_vertice_emb' in python_outputs and 'updated_vertice_emb' in js_outputs:
        py_emb = np.array(python_outputs['updated_vertice_emb'])
        js_emb = np.array(js_outputs['updated_vertice_emb'])
        
        diff = np.abs(py_emb - js_emb)
        analysis['embedding'] = {
            'max_diff': float(np.max(diff)),
            'mean_diff': float(np.mean(diff)),
            'std_diff': float(np.std(diff)),
            'num_large_diffs': int(np.sum(diff > 0.01)),
            'percent_large_diffs': float(np.sum(diff > 0.01) / len(diff) * 100)
        }
    
    return analysis

def diagnose_issues(analysis):
    """Diagnose potential issues and provide recommendations"""
    diagnosis = {
        'issues': [],
        'recommendations': [],
        'severity': 'low'
    }
    
    # Check vertices
    if 'vertices' in analysis:
        v_analysis = analysis['vertices']
        if v_analysis['max_diff'] > 0.1:
            diagnosis['issues'].append("Large differences in vertex outputs (max > 0.1)")
            diagnosis['severity'] = 'high'
        elif v_analysis['max_diff'] > 0.01:
            diagnosis['issues'].append("Moderate differences in vertex outputs (max > 0.01)")
            diagnosis['severity'] = 'medium'
        
        if v_analysis['percent_large_diffs'] > 10:
            diagnosis['issues'].append(f"{v_analysis['percent_large_diffs']:.1f}% of vertices have large differences")
    
    # Check embedding
    if 'embedding' in analysis:
        e_analysis = analysis['embedding']
        if e_analysis['max_diff'] > 0.1:
            diagnosis['issues'].append("Large differences in embedding outputs (max > 0.1)")
            diagnosis['severity'] = 'high'
        elif e_analysis['max_diff'] > 0.01:
            diagnosis['issues'].append("Moderate differences in embedding outputs (max > 0.01)")
            if diagnosis['severity'] == 'low':
                diagnosis['severity'] = 'medium'
    
    # Generate recommendations
    if diagnosis['severity'] == 'high':
        diagnosis['recommendations'].extend([
            "🔧 Check model architecture consistency between Python and ONNX",
            "🔧 Verify weight initialization and loading",
            "🔧 Compare layer-by-layer outputs for debugging",
            "🔧 Check input preprocessing differences"
        ])
    elif diagnosis['severity'] == 'medium':
        diagnosis['recommendations'].extend([
            "⚠️ Small numerical differences are normal but should be minimized",
            "⚠️ Consider using same random seeds for weight initialization",
            "⚠️ Verify ONNX export settings and precision",
            "⚠️ Check for different activation functions or numerical precision"
        ])
    else:
        diagnosis['recommendations'].append("✅ Models are well aligned!")
    
    return diagnosis

def generate_report(results):
    """Generate a comprehensive comparison report"""
    print("\n" + "=" * 80)
    print("🔍 FACEFORMER MODEL COMPARISON DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # Extract data
    python_outputs = results['python']['simplified_results']['simplified_python_outputs']
    
    if 'node' in results:
        js_outputs = results['node']['js_results']['outputs']
        node_comparison = results['node']['comparison']
        
        print(f"\n📊 QUICK STATS:")
        print(f"  ✅ Shapes match: {node_comparison['shapes_match']}")
        print(f"  {'✅' if node_comparison['values_close'] else '⚠️'} Values close: {node_comparison['values_close']}")
        print(f"  📏 Max vertex difference: {node_comparison['max_diff_vertices']:.6f}")
        print(f"  📏 Max embedding difference: {node_comparison['max_diff_embedding']:.6f}")
        print(f"  🎯 Tolerance: {node_comparison['tolerance']}")
        
        # Detailed analysis
        analysis = analyze_differences(python_outputs, js_outputs)
        diagnosis = diagnose_issues(analysis)
        
        print(f"\n📈 DETAILED ANALYSIS:")
        if 'vertices' in analysis:
            v = analysis['vertices']
            print(f"  Vertices:")
            print(f"    Max difference: {v['max_diff']:.6f}")
            print(f"    Mean difference: {v['mean_diff']:.6f}")
            print(f"    Std difference: {v['std_diff']:.6f}")
            print(f"    Large diffs (>0.01): {v['num_large_diffs']} ({v['percent_large_diffs']:.1f}%)")
        
        if 'embedding' in analysis:
            e = analysis['embedding']
            print(f"  Embedding:")
            print(f"    Max difference: {e['max_diff']:.6f}")
            print(f"    Mean difference: {e['mean_diff']:.6f}")
            print(f"    Std difference: {e['std_diff']:.6f}")
            print(f"    Large diffs (>0.01): {e['num_large_diffs']} ({e['percent_large_diffs']:.1f}%)")
        
        print(f"\n🩺 DIAGNOSIS:")
        print(f"  Severity: {diagnosis['severity'].upper()}")
        
        if diagnosis['issues']:
            print(f"  Issues found:")
            for issue in diagnosis['issues']:
                print(f"    ⚠️ {issue}")
        else:
            print(f"  ✅ No significant issues detected")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in diagnosis['recommendations']:
            print(f"    {rec}")
    
    else:
        print("\n⚠️ JavaScript comparison results not available")
        print("   Run the Node.js comparison script to get detailed analysis")
    
    # Model statistics
    print(f"\n📋 MODEL INFORMATION:")
    python_info = results['python']['simplified_results']['model_params']
    print(f"  Audio dimensions: {python_info['audio_dim']}")
    print(f"  Embedding dimensions: {python_info['embedding_dim']}")
    print(f"  Vertex dimensions: {python_info['vertex_dim']}")
    print(f"  Number of subjects: {python_info['num_subjects']}")
    
    # Sample values comparison
    print(f"\n🔢 SAMPLE VALUES COMPARISON:")
    print(f"  Python vertices (first 3): {python_outputs['new_vertice_out'][:3]}")
    print(f"  Python embedding (first 3): {python_outputs['updated_vertice_emb'][:3]}")
    
    if 'node' in results:
        print(f"  JS vertices (first 3): {js_outputs['new_vertice_out'][:3]}")
        print(f"  JS embedding (first 3): {js_outputs['updated_vertice_emb'][:3]}")
    
    print(f"\n🎯 CONCLUSIONS:")
    if 'node' in results:
        if diagnosis['severity'] == 'low':
            print("  ✅ Models are well-aligned and producing consistent outputs")
            print("  ✅ JavaScript implementation is correctly matching Python model")
            print("  ✅ Ready for production use")
        elif diagnosis['severity'] == 'medium':
            print("  ⚠️ Models show acceptable differences but could be improved")
            print("  ⚠️ JavaScript implementation is mostly correct")
            print("  ⚠️ Consider fine-tuning for better precision")
        else:
            print("  ❌ Models show significant differences")
            print("  ❌ JavaScript implementation needs investigation")
            print("  ❌ Not ready for production use")
    else:
        print("  📝 Python reference model has been generated")
        print("  📝 Run JavaScript comparison to complete analysis")
    
    print("\n" + "=" * 80)
    print("📄 Report generated successfully!")
    print("=" * 80)

def main():
    print("📊 FaceFormer Model Comparison Diagnostic Tool")
    
    # Load results
    results = load_comparison_results()
    if not results:
        print("❌ Could not load comparison results")
        return
    
    # Generate report
    generate_report(results)
    
    # Save detailed report
    try:
        report_data = {
            'timestamp': '2025-07-13',
            'analysis_version': '1.0',
            'results': results
        }
        
        with open('diagnostic_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"💾 Detailed report saved to: diagnostic_report.json")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")

if __name__ == "__main__":
    main()
