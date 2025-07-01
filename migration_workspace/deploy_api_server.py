#!/usr/bin/env python3
"""
Motion Inference API Server

Production-ready Flask API server for the Mojo Motion Bridge.
Provides REST endpoints for real-time motion inference.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

try:
    from mojo_bridge import MojoMotionBridge
    BRIDGE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Mojo Bridge import failed: {e}")
    BRIDGE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global bridge instance
bridge = None
startup_time = time.time()

def initialize_bridge():
    """Initialize the motion inference bridge."""
    global bridge
    if not BRIDGE_AVAILABLE:
        logger.error("Mojo Bridge not available")
        return False
    
    try:
        bridge = MojoMotionBridge()
        logger.info("✅ Motion inference bridge initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize bridge: {e}")
        return False

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time
    return jsonify({
        'status': 'healthy' if bridge else 'degraded',
        'bridge_available': bridge is not None,
        'uptime_seconds': uptime,
        'timestamp': time.time()
    })

# Model info endpoint
@app.route('/models', methods=['GET'])
def get_models():
    """Get information about loaded models."""
    if not bridge:
        return jsonify({'error': 'Bridge not initialized'}), 500
    
    try:
        models_info = {
            'loaded_models': list(bridge.onnx_sessions.keys()),
            'performance_stats': bridge.get_performance_report(),
            'total_models': len(bridge.onnx_sessions)
        }
        return jsonify(models_info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

# Motion phase encoding endpoint
@app.route('/api/motion/phase', methods=['POST'])
def encode_motion_phase():
    """Encode motion features to phase coordinates."""
    if not bridge:
        return jsonify({'error': 'Bridge not initialized'}), 500
    
    try:
        data = request.json
        motion_features = np.array(data['motion_features'], dtype=np.float32)
        
        # Validate input shape
        if motion_features.shape[-1] != 132:
            return jsonify({'error': 'Motion features must have 132 dimensions'}), 400
        
        # Ensure batch dimension
        if len(motion_features.shape) == 1:
            motion_features = motion_features.reshape(1, -1)
        
        # Perform inference
        start_time = time.time()
        phase_coords = bridge.encode_motion_phase(motion_features)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return jsonify({
            'phase_coordinates': phase_coords.tolist(),
            'input_shape': motion_features.shape,
            'output_shape': phase_coords.shape,
            'inference_time_ms': inference_time
        })
        
    except Exception as e:
        logger.error(f"Error in motion phase encoding: {e}")
        return jsonify({'error': str(e)}), 500

# Style extraction endpoint
@app.route('/api/motion/style', methods=['POST'])
def extract_motion_style():
    """Extract style vectors from motion sequence."""
    if not bridge:
        return jsonify({'error': 'Bridge not initialized'}), 500
    
    try:
        data = request.json
        motion_sequence = np.array(data['motion_sequence'], dtype=np.float32)
        
        # Validate input shape
        expected_size = 60 * 73  # 4380
        if motion_sequence.shape[-1] != expected_size:
            return jsonify({'error': f'Motion sequence must have {expected_size} dimensions'}), 400
        
        # Ensure batch dimension
        if len(motion_sequence.shape) == 1:
            motion_sequence = motion_sequence.reshape(1, -1)
        
        # Perform inference
        start_time = time.time()
        mu, logvar = bridge.extract_motion_style(motion_sequence)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return jsonify({
            'style_mean': mu.tolist(),
            'style_logvar': logvar.tolist(),
            'input_shape': motion_sequence.shape,
            'output_shapes': {'mu': mu.shape, 'logvar': logvar.shape},
            'inference_time_ms': inference_time
        })
        
    except Exception as e:
        logger.error(f"Error in style extraction: {e}")
        return jsonify({'error': str(e)}), 500

# Action generation endpoint
@app.route('/api/deepmimic/actions', methods=['POST'])
def generate_actions():
    """Generate character control actions from state."""
    if not bridge:
        return jsonify({'error': 'Bridge not initialized'}), 500
    
    try:
        data = request.json
        state = np.array(data['state'], dtype=np.float32)
        
        # Validate input shape
        if state.shape[-1] != 197:
            return jsonify({'error': 'State must have 197 dimensions'}), 400
        
        # Ensure batch dimension
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Perform inference
        start_time = time.time()
        actions = bridge.generate_actions(state)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return jsonify({
            'actions': actions.tolist(),
            'input_shape': state.shape,
            'output_shape': actions.shape,
            'inference_time_ms': inference_time
        })
        
    except Exception as e:
        logger.error(f"Error in action generation: {e}")
        return jsonify({'error': str(e)}), 500

# State value estimation endpoint
@app.route('/api/deepmimic/value', methods=['POST'])
def estimate_state_value():
    """Estimate state value using critic network."""
    if not bridge:
        return jsonify({'error': 'Bridge not initialized'}), 500
    
    try:
        data = request.json
        state = np.array(data['state'], dtype=np.float32)
        
        # Validate input shape
        if state.shape[-1] != 197:
            return jsonify({'error': 'State must have 197 dimensions'}), 400
        
        # Ensure batch dimension
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Perform inference
        start_time = time.time()
        value = bridge.estimate_state_value(state)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return jsonify({
            'state_value': value.tolist(),
            'input_shape': state.shape,
            'output_shape': value.shape,
            'inference_time_ms': inference_time
        })
        
    except Exception as e:
        logger.error(f"Error in state value estimation: {e}")
        return jsonify({'error': str(e)}), 500

# Full pipeline endpoint
@app.route('/api/motion/pipeline', methods=['POST'])
def process_motion_pipeline():
    """Process full motion inference pipeline."""
    if not bridge:
        return jsonify({'error': 'Bridge not initialized'}), 500
    
    try:
        data = request.json
        motion_features = np.array(data['motion_features'], dtype=np.float32)
        character_state = np.array(data['character_state'], dtype=np.float32)
        
        # Validate input shapes
        if motion_features.shape[-1] != 132:
            return jsonify({'error': 'Motion features must have 132 dimensions'}), 400
        if character_state.shape[-1] != 197:
            return jsonify({'error': 'Character state must have 197 dimensions'}), 400
        
        # Ensure batch dimensions
        if len(motion_features.shape) == 1:
            motion_features = motion_features.reshape(1, -1)
        if len(character_state.shape) == 1:
            character_state = character_state.reshape(1, -1)
        
        # Perform inference
        start_time = time.time()
        results = bridge.process_motion_pipeline(motion_features, character_state)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        json_results['inference_time_ms'] = inference_time
        
        return jsonify(json_results)
        
    except Exception as e:
        logger.error(f"Error in motion pipeline: {e}")
        return jsonify({'error': str(e)}), 500

# Batch processing endpoint
@app.route('/api/motion/batch', methods=['POST'])
def batch_process_motions():
    """Process batch of motion data."""
    if not bridge:
        return jsonify({'error': 'Bridge not initialized'}), 500
    
    try:
        data = request.json
        motion_batch = np.array(data['motion_batch'], dtype=np.float32)
        state_batch = np.array(data['state_batch'], dtype=np.float32)
        
        # Validate input shapes
        if motion_batch.shape[-1] != 132:
            return jsonify({'error': 'Motion batch must have 132 dimensions in last axis'}), 400
        if state_batch.shape[-1] != 197:
            return jsonify({'error': 'State batch must have 197 dimensions in last axis'}), 400
        
        # Perform batch inference
        start_time = time.time()
        results = bridge.batch_process_motions(motion_batch, state_batch)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        json_results['batch_size'] = motion_batch.shape[0]
        json_results['total_inference_time_ms'] = inference_time
        json_results['avg_inference_time_ms'] = inference_time / motion_batch.shape[0]
        
        return jsonify(json_results)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return jsonify({'error': str(e)}), 500

# Performance benchmarking endpoint
@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """Run performance benchmark."""
    if not bridge:
        return jsonify({'error': 'Bridge not initialized'}), 500
    
    try:
        data = request.json
        iterations = data.get('iterations', 100)
        
        # Run benchmark
        start_time = time.time()
        benchmark_results = bridge.benchmark_inference(iterations)
        total_time = (time.time() - start_time) * 1000  # ms
        
        benchmark_results['total_benchmark_time_ms'] = total_time
        benchmark_results['iterations'] = iterations
        
        return jsonify(benchmark_results)
        
    except Exception as e:
        logger.error(f"Error in benchmark: {e}")
        return jsonify({'error': str(e)}), 500

# Web interface
@app.route('/', methods=['GET'])
def web_interface():
    """Simple web interface for testing the API."""
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Motion Inference API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .endpoint { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .method { background: #007bff; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        pre { background: #282c34; color: #abb2bf; padding: 15px; border-radius: 4px; overflow-x: auto; }
        h1 { color: #333; text-align: center; }
        h2 { color: #555; }
        .status { text-align: center; padding: 20px; }
        .healthy { color: #28a745; }
        .degraded { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Motion Inference API Server</h1>
        
        <div class="status">
            <p>Bridge Status: <span class="{{ 'healthy' if bridge_available else 'degraded' }}">
                {{ '✅ Operational' if bridge_available else '❌ Not Available' }}
            </span></p>
        </div>

        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Health check and system status</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /models</h3>
            <p>Get information about loaded models</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/motion/phase</h3>
            <p>Encode motion features to phase coordinates</p>
            <pre>{"motion_features": [132 float values]}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/motion/style</h3>
            <p>Extract style vectors from motion sequence</p>
            <pre>{"motion_sequence": [4380 float values]}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/deepmimic/actions</h3>
            <p>Generate character control actions</p>
            <pre>{"state": [197 float values]}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/deepmimic/value</h3>
            <p>Estimate state value</p>
            <pre>{"state": [197 float values]}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/motion/pipeline</h3>
            <p>Full motion processing pipeline</p>
            <pre>{"motion_features": [132 floats], "character_state": [197 floats]}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/motion/batch</h3>
            <p>Batch processing</p>
            <pre>{"motion_batch": [[132 floats], ...], "state_batch": [[197 floats], ...]}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/benchmark</h3>
            <p>Performance benchmarking</p>
            <pre>{"iterations": 100}</pre>
        </div>
        
        <h2>Example Usage</h2>
        <pre>
# Test motion phase encoding
curl -X POST http://localhost:5000/api/motion/phase \\
  -H "Content-Type: application/json" \\
  -d '{"motion_features": [0.1, 0.2, ..., (132 values total)]}'

# Test full pipeline
curl -X POST http://localhost:5000/api/motion/pipeline \\
  -H "Content-Type: application/json" \\
  -d '{
    "motion_features": [0.1, 0.2, ..., (132 values)],
    "character_state": [0.1, 0.2, ..., (197 values)]
  }'
        </pre>
    </div>
</body>
</html>
    """
    return render_template_string(html_template, bridge_available=bridge is not None)

if __name__ == '__main__':
    print("🚀 Starting Motion Inference API Server...")
    print("=" * 50)
    
    # Initialize the bridge
    if initialize_bridge():
        print("✅ Bridge initialized successfully")
        print(f"📊 Loaded models: {list(bridge.onnx_sessions.keys())}")
    else:
        print("❌ Bridge initialization failed")
    
    # Get configuration
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🌐 Starting server on {host}:{port}")
    print(f"🔗 Web interface: http://localhost:{port}")
    print("🚀 Server starting...")
    
    # Start the Flask app
    app.run(host=host, port=port, debug=debug)
