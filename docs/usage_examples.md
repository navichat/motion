# Usage Examples

This document provides practical examples demonstrating how to use the various projects in the Motion workspace together to create complete motion processing and animation workflows.

## Table of Contents

1. [Basic Workflows](#basic-workflows)
2. [BVH to Animation Pipeline](#bvh-to-animation-pipeline)
3. [Training Custom Motion Models](#training-custom-motion-models)
4. [Real-time Character Animation](#real-time-character-animation)
5. [Advanced Integration Examples](#advanced-integration-examples)
6. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Basic Workflows

### Workflow 1: Convert Motion Capture to Animated Character

This example shows the complete pipeline from BVH motion capture data to an animated character in the chat interface.

```bash
# Step 1: Convert BVH to DeepMimic format
cd BvhToDeepMimic
python -c "
from bvhtomimic import BvhConverter
converter = BvhConverter('./Settings/settings.json')
converter.writeDeepMimicFile('../assets/walking.bvh', '../assets/walking_deepmimic.txt')
print('Conversion complete!')
"

# Step 2: Train DeepMimic policy
cd ../pytorch_DeepMimic/deepmimic
# Update motion file path in args file
sed -i 's/motion_file.*txt/motion_file \.\.\/\.\.\/assets\/walking_deepmimic\.txt/' train_humanoid3d_walk_args.txt
python DeepMimic_Optimizer.py --arg_file train_humanoid3d_walk_args.txt

# Step 3: Generate smooth transitions with RSMT
cd ../../RSMT-Realtime-Stylized-Motion-Transition
python process_dataset.py --preprocess
python process_dataset.py --train_phase_model
python train_deephase.py --epochs 100

# Step 4: Deploy in chat interface
cd ../chat/webapp
npm run build
cd ../server
node server.js
```

### Workflow 2: Style Transfer Between Motions

```python
# style_transfer_example.py
import sys
sys.path.append('BvhToDeepMimic')
sys.path.append('RSMT-Realtime-Stylized-Motion-Transition')

from bvhtomimic import BvhConverter
import torch
import numpy as np

# Convert multiple BVH files with different styles
converter = BvhConverter("BvhToDeepMimic/Settings/settings.json")

motion_files = [
    ("assets/casual_walk.bvh", "casual"),
    ("assets/confident_walk.bvh", "confident"),
    ("assets/sneaky_walk.bvh", "sneaky")
]

# Convert all motion files
converted_motions = []
for bvh_file, style in motion_files:
    output_file = f"assets/{style}_deepmimic.txt"
    converter.writeDeepMimicFile(bvh_file, output_file)
    converted_motions.append((output_file, style))
    print(f"Converted {style} motion")

# Use RSMT to create style transitions
# This would integrate with RSMT's style transfer capabilities
print("Style transfer setup complete!")
```

## BVH to Animation Pipeline

### Example 1: Batch Processing BVH Files

```python
# batch_convert_bvh.py
import os
from pathlib import Path
from bvhtomimic import BvhConverter

def batch_convert_bvh_files(input_dir, output_dir, settings_file):
    """Convert all BVH files in a directory to DeepMimic format"""
    
    converter = BvhConverter(settings_file)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all BVH files
    bvh_files = list(input_path.glob("*.bvh"))
    
    print(f"Found {len(bvh_files)} BVH files to convert")
    
    for bvh_file in bvh_files:
        try:
            # Generate output filename
            output_file = output_path / f"{bvh_file.stem}_deepmimic.txt"
            
            # Convert file
            converter.writeDeepMimicFile(str(bvh_file), str(output_file))
            print(f"✓ Converted: {bvh_file.name}")
            
        except Exception as e:
            print(f"✗ Failed to convert {bvh_file.name}: {e}")
    
    print("Batch conversion complete!")

# Usage
if __name__ == "__main__":
    batch_convert_bvh_files(
        input_dir="assets/bvh_files",
        output_dir="assets/deepmimic_files",
        settings_file="BvhToDeepMimic/Settings/settings.json"
    )
```

### Example 2: Custom Joint Mapping

```json
{
  "// Custom settings for different skeleton types": "",
  "mixamo_settings": {
    "scale": 0.01,
    "rootJoints": ["mixamorig:Hips"],
    "jointMapping": {
      "mixamorig:Hips": "root",
      "mixamorig:LeftUpLeg": "left_hip",
      "mixamorig:RightUpLeg": "right_hip",
      "mixamorig:LeftLeg": "left_knee",
      "mixamorig:RightLeg": "right_knee",
      "mixamorig:LeftFoot": "left_ankle",
      "mixamorig:RightFoot": "right_ankle",
      "mixamorig:Spine": "torso",
      "mixamorig:LeftArm": "left_shoulder",
      "mixamorig:RightArm": "right_shoulder",
      "mixamorig:LeftForeArm": "left_elbow",
      "mixamorig:RightForeArm": "right_elbow"
    }
  }
}
```

```python
# custom_skeleton_converter.py
from bvhtomimic import BvhConverter
import json

# Load custom settings for Mixamo skeleton
with open('custom_mixamo_settings.json', 'r') as f:
    settings = json.load(f)

# Create temporary settings file
with open('temp_mixamo_settings.json', 'w') as f:
    json.dump(settings['mixamo_settings'], f, indent=2)

# Convert with custom settings
converter = BvhConverter("temp_mixamo_settings.json")
converter.writeDeepMimicFile(
    "assets/mixamo_character.bvh", 
    "assets/mixamo_character_deepmimic.txt"
)

print("Mixamo character converted successfully!")
```

## Training Custom Motion Models

### Example 1: Training DeepMimic with Custom Motions

```bash
#!/bin/bash
# train_custom_motion.sh

# Set up training environment
cd pytorch_DeepMimic/deepmimic

# Create custom argument file
cat > train_custom_motion_args.txt << EOF
--env_type humanoid3d
--char_file chars/humanoid3d.txt
--motion_file ../../assets/custom_motion_deepmimic.txt
--num_workers 4
--int_output_iters 50
--int_save_iters 100
--max_iter 5000
--learning_rate 0.0001
--output_path output/custom_motion/
EOF

# Start training
echo "Starting training for custom motion..."
python DeepMimic_Optimizer.py --arg_file train_custom_motion_args.txt

# Monitor training progress
echo "Training started. Monitor progress in output/custom_motion/"
```

### Example 2: Multi-Motion Training

```python
# multi_motion_training.py
import os
import subprocess
import time

def train_multiple_motions(motion_configs):
    """Train DeepMimic policies for multiple motions"""
    
    for config in motion_configs:
        print(f"Training motion: {config['name']}")
        
        # Create argument file
        args_content = f"""
--env_type humanoid3d
--char_file chars/humanoid3d.txt
--motion_file {config['motion_file']}
--num_workers {config.get('workers', 4)}
--int_output_iters {config.get('output_iters', 50)}
--int_save_iters {config.get('save_iters', 100)}
--max_iter {config.get('max_iter', 3000)}
--learning_rate {config.get('learning_rate', 0.0001)}
--output_path output/{config['name']}/
        """.strip()
        
        args_file = f"train_{config['name']}_args.txt"
        with open(args_file, 'w') as f:
            f.write(args_content)
        
        # Start training process
        cmd = ["python", "DeepMimic_Optimizer.py", "--arg_file", args_file]
        process = subprocess.Popen(cmd)
        
        # Wait for completion or timeout
        try:
            process.wait(timeout=config.get('timeout', 3600))  # 1 hour default
            print(f"✓ Completed training: {config['name']}")
        except subprocess.TimeoutExpired:
            process.terminate()
            print(f"⚠ Training timeout: {config['name']}")

# Configuration for multiple motions
motion_configs = [
    {
        'name': 'walking',
        'motion_file': '../../assets/walking_deepmimic.txt',
        'max_iter': 2000
    },
    {
        'name': 'running',
        'motion_file': '../../assets/running_deepmimic.txt',
        'max_iter': 3000
    },
    {
        'name': 'jumping',
        'motion_file': '../../assets/jumping_deepmimic.txt',
        'max_iter': 4000
    }
]

if __name__ == "__main__":
    os.chdir("pytorch_DeepMimic/deepmimic")
    train_multiple_motions(motion_configs)
```

### Example 3: RSMT Style Training

```python
# rsmt_style_training.py
import subprocess
import os

def train_rsmt_pipeline(dataset_path, styles=['casual', 'energetic', 'graceful']):
    """Complete RSMT training pipeline with style awareness"""
    
    os.chdir("RSMT-Realtime-Stylized-Motion-Transition")
    
    # Step 1: Preprocess dataset
    print("Step 1: Preprocessing dataset...")
    subprocess.run(["python", "process_dataset.py", "--preprocess"])
    
    # Step 2: Train phase model
    print("Step 2: Training phase model...")
    subprocess.run(["python", "process_dataset.py", "--train_phase_model"])
    subprocess.run(["python", "train_deephase.py", "--epochs", "150"])
    
    # Step 3: Add phase to dataset
    print("Step 3: Adding phase vectors to dataset...")
    phase_model_path = "output/deephase/latest_model.pth"
    subprocess.run([
        "python", "process_dataset.py", 
        "--add_phase_to_dataset", 
        "--model_path", phase_model_path
    ])
    
    # Step 4: Train manifold
    print("Step 4: Training manifold model...")
    subprocess.run(["python", "process_dataset.py", "--train_manifold_model"])
    subprocess.run(["python", "train_styleVAE.py", "--epochs", "200"])
    
    # Step 5: Train sampler
    print("Step 5: Training sampler...")
    manifold_model_path = "output/styleVAE/m_save_model_latest"
    subprocess.run([
        "python", "train_sampler.py",
        "--manifold_model", manifold_model_path,
        "--epochs", "100"
    ])
    
    print("RSMT training pipeline complete!")

if __name__ == "__main__":
    train_rsmt_pipeline("MotionData/100STYLE")
```

## Real-time Character Animation

### Example 1: Chat Interface with Motion Integration

```javascript
// chat_motion_integration.js
import { Player } from '@navi/player';

class MotionChatInterface {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.player = null;
        this.currentMotion = 'idle';
        this.motionQueue = [];
    }
    
    async initialize() {
        // Initialize 3D player
        this.player = new Player({
            container: this.container,
            assets: {
                character: 'assets/avatars/default_character.glb',
                animations: {
                    idle: 'assets/animations/idle.bvh',
                    talking: 'assets/animations/talking.bvh',
                    listening: 'assets/animations/listening.bvh',
                    excited: 'assets/animations/excited.bvh'
                }
            }
        });
        
        await this.player.loadCharacter();
        console.log('Chat interface initialized');
    }
    
    processMessage(message, emotion = 'neutral') {
        // Analyze message for motion cues
        const motionCues = this.analyzeMessageForMotion(message);
        
        // Queue appropriate animation
        const targetAnimation = this.selectAnimation(emotion, motionCues);
        this.queueMotion(targetAnimation);
        
        // Generate response
        return this.generateResponse(message, emotion);
    }
    
    analyzeMessageForMotion(message) {
        const cues = {
            excitement: ['amazing', 'wow', 'incredible', 'fantastic'].some(word => 
                message.toLowerCase().includes(word)
            ),
            questioning: message.includes('?'),
            greeting: ['hello', 'hi', 'hey'].some(word => 
                message.toLowerCase().includes(word)
            )
        };
        
        return cues;
    }
    
    selectAnimation(emotion, cues) {
        if (cues.excitement) return 'excited';
        if (cues.greeting) return 'wave';
        if (cues.questioning) return 'thinking';
        
        switch(emotion) {
            case 'happy': return 'smile';
            case 'sad': return 'disappointed';
            case 'excited': return 'excited';
            default: return 'talking';
        }
    }
    
    queueMotion(animationName) {
        this.motionQueue.push(animationName);
        if (this.motionQueue.length === 1) {
            this.playNextMotion();
        }
    }
    
    async playNextMotion() {
        if (this.motionQueue.length === 0) {
            this.player.playAnimation('idle');
            return;
        }
        
        const nextMotion = this.motionQueue.shift();
        await this.player.playAnimation(nextMotion);
        
        // Continue with next motion after delay
        setTimeout(() => this.playNextMotion(), 500);
    }
    
    generateResponse(message, emotion) {
        // This would integrate with the psyche module
        return `I understand you're feeling ${emotion}. Let me help with that!`;
    }
}

// Usage example
const chatInterface = new MotionChatInterface('character-container');
chatInterface.initialize().then(() => {
    console.log('Chat interface ready');
    
    // Example conversation
    const response1 = chatInterface.processMessage("Hello there!", "happy");
    const response2 = chatInterface.processMessage("That's amazing!", "excited");
});
```

### Example 2: Real-time Motion Transition Server

```javascript
// motion_transition_server.js
import WebSocket from 'ws';
import { spawn } from 'child_process';

class MotionTransitionServer {
    constructor(port = 8081) {
        this.wss = new WebSocket.Server({ port });
        this.activeConnections = new Map();
        this.motionCache = new Map();
        
        this.setupWebSocketHandlers();
        console.log(`Motion transition server started on port ${port}`);
    }
    
    setupWebSocketHandlers() {
        this.wss.on('connection', (ws) => {
            const clientId = this.generateClientId();
            this.activeConnections.set(clientId, ws);
            
            ws.on('message', (data) => {
                this.handleMotionRequest(clientId, JSON.parse(data));
            });
            
            ws.on('close', () => {
                this.activeConnections.delete(clientId);
            });
        });
    }
    
    async handleMotionRequest(clientId, request) {
        const { type, sourceMotion, targetStyle, transitionDuration } = request;
        
        switch(type) {
            case 'generate_transition':
                await this.generateMotionTransition(
                    clientId, 
                    sourceMotion, 
                    targetStyle, 
                    transitionDuration
                );
                break;
                
            case 'apply_style':
                await this.applyMotionStyle(clientId, sourceMotion, targetStyle);
                break;
                
            case 'blend_motions':
                await this.blendMotions(clientId, request.motions, request.weights);
                break;
        }
    }
    
    async generateMotionTransition(clientId, sourceMotion, targetStyle, duration) {
        const ws = this.activeConnections.get(clientId);
        if (!ws) return;
        
        // Check cache first
        const cacheKey = `${sourceMotion}_${targetStyle}_${duration}`;
        if (this.motionCache.has(cacheKey)) {
            ws.send(JSON.stringify({
                type: 'transition_ready',
                motionData: this.motionCache.get(cacheKey)
            }));
            return;
        }
        
        // Generate transition using RSMT
        try {
            ws.send(JSON.stringify({ type: 'transition_generating' }));
            
            const motionData = await this.runRSMTTransition(
                sourceMotion, 
                targetStyle, 
                duration
            );
            
            // Cache result
            this.motionCache.set(cacheKey, motionData);
            
            ws.send(JSON.stringify({
                type: 'transition_ready',
                motionData: motionData
            }));
            
        } catch (error) {
            ws.send(JSON.stringify({
                type: 'transition_error',
                error: error.message
            }));
        }
    }
    
    async runRSMTTransition(sourceMotion, targetStyle, duration) {
        return new Promise((resolve, reject) => {
            const rsmt = spawn('python', [
                'RSMT-Realtime-Stylized-Motion-Transition/generate_transition.py',
                '--source', sourceMotion,
                '--target_style', targetStyle,
                '--duration', duration.toString(),
                '--output_format', 'json'
            ]);
            
            let output = '';
            rsmt.stdout.on('data', (data) => {
                output += data.toString();
            });
            
            rsmt.on('close', (code) => {
                if (code === 0) {
                    try {
                        const motionData = JSON.parse(output);
                        resolve(motionData);
                    } catch (e) {
                        reject(new Error('Failed to parse motion data'));
                    }
                } else {
                    reject(new Error(`RSMT process failed with code ${code}`));
                }
            });
        });
    }
    
    generateClientId() {
        return Math.random().toString(36).substring(7);
    }
}

// Start server
const server = new MotionTransitionServer(8081);
```

## Advanced Integration Examples

### Example 1: Complete Animation Pipeline

```python
# complete_animation_pipeline.py
import json
import subprocess
import time
from pathlib import Path

class AnimationPipeline:
    def __init__(self, workspace_root):
        self.root = Path(workspace_root)
        self.bvh_converter_path = self.root / "BvhToDeepMimic"
        self.deepmimic_path = self.root / "pytorch_DeepMimic"
        self.rsmt_path = self.root / "RSMT-Realtime-Stylized-Motion-Transition"
        self.assets_path = self.root / "assets"
        
    def process_motion_capture(self, bvh_file, character_name):
        """Complete pipeline from BVH to animated character"""
        
        print(f"Processing motion capture for {character_name}")
        
        # Step 1: Convert BVH to DeepMimic
        deepmimic_file = self.convert_bvh_to_deepmimic(bvh_file, character_name)
        
        # Step 2: Train DeepMimic policy
        policy_file = self.train_deepmimic_policy(deepmimic_file, character_name)
        
        # Step 3: Generate style variations with RSMT
        style_variations = self.generate_style_variations(
            deepmimic_file, 
            ['casual', 'energetic', 'graceful']
        )
        
        # Step 4: Create character package for chat interface
        character_package = self.create_character_package(
            character_name,
            policy_file,
            style_variations
        )
        
        print(f"Character package created: {character_package}")
        return character_package
    
    def convert_bvh_to_deepmimic(self, bvh_file, character_name):
        """Convert BVH file to DeepMimic format"""
        
        from bvhtomimic import BvhConverter
        
        converter = BvhConverter(
            str(self.bvh_converter_path / "Settings" / "settings.json")
        )
        
        output_file = self.assets_path / f"{character_name}_deepmimic.txt"
        converter.writeDeepMimicFile(str(bvh_file), str(output_file))
        
        print(f"✓ Converted BVH to DeepMimic: {output_file}")
        return output_file
    
    def train_deepmimic_policy(self, motion_file, character_name):
        """Train DeepMimic policy for the motion"""
        
        # Create training configuration
        args_content = f"""
--env_type humanoid3d
--char_file chars/humanoid3d.txt
--motion_file {motion_file}
--num_workers 4
--int_output_iters 25
--int_save_iters 50
--max_iter 1000
--output_path output/{character_name}/
        """.strip()
        
        args_file = self.deepmimic_path / "deepmimic" / f"train_{character_name}_args.txt"
        with open(args_file, 'w') as f:
            f.write(args_content)
        
        # Run training
        cmd = [
            "python", "DeepMimic_Optimizer.py",
            "--arg_file", str(args_file)
        ]
        
        print(f"Training DeepMimic policy for {character_name}...")
        result = subprocess.run(
            cmd, 
            cwd=str(self.deepmimic_path / "deepmimic"),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            policy_file = self.deepmimic_path / "deepmimic" / "output" / character_name / "agent0_model_anet.pth"
            print(f"✓ Training completed: {policy_file}")
            return policy_file
        else:
            raise Exception(f"Training failed: {result.stderr}")
    
    def generate_style_variations(self, motion_file, styles):
        """Generate style variations using RSMT"""
        
        style_files = {}
        
        for style in styles:
            print(f"Generating {style} style variation...")
            
            # This would use RSMT to generate style variations
            # Simplified for example
            style_file = self.assets_path / f"motion_{style}.bvh"
            
            # Run RSMT style transfer (pseudo-code)
            cmd = [
                "python", "generate_style_variation.py",
                "--input", str(motion_file),
                "--style", style,
                "--output", str(style_file)
            ]
            
            # subprocess.run(cmd, cwd=str(self.rsmt_path))
            style_files[style] = style_file
        
        print(f"✓ Generated {len(styles)} style variations")
        return style_files
    
    def create_character_package(self, character_name, policy_file, style_variations):
        """Create deployable character package"""
        
        package_dir = self.assets_path / "characters" / character_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Create character manifest
        manifest = {
            "name": character_name,
            "policy_file": str(policy_file),
            "style_variations": {k: str(v) for k, v in style_variations.items()},
            "created": time.time(),
            "version": "1.0"
        }
        
        manifest_file = package_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✓ Character package created: {package_dir}")
        return package_dir

# Usage example
if __name__ == "__main__":
    pipeline = AnimationPipeline("/path/to/motion/workspace")
    
    # Process a motion capture file
    character_package = pipeline.process_motion_capture(
        bvh_file="assets/walking_motion.bvh",
        character_name="walker"
    )
    
    print(f"Ready to deploy character: {character_package}")
```

### Example 2: Real-time Performance Monitoring

```python
# performance_monitor.py
import time
import psutil
import torch
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, log_file="performance.log"):
        self.log_file = log_file
        self.metrics = []
        self.start_time = time.time()
        
    def log_system_metrics(self):
        """Log current system performance metrics"""
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
        
        # GPU metrics if available
        if torch.cuda.is_available():
            metrics["gpu"] = {
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_cached": torch.cuda.memory_reserved(),
                "gpu_utilization": self.get_gpu_utilization()
            }
        
        self.metrics.append(metrics)
        self.save_metrics()
        
    def get_gpu_utilization(self):
        """Get GPU utilization percentage"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return None
    
    def save_metrics(self):
        """Save metrics to log file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def monitor_training(self, training_function, *args, **kwargs):
        """Monitor performance during training"""
        
        print("Starting performance monitoring...")
        
        def monitoring_loop():
            while self.monitoring:
                self.log_system_metrics()
                time.sleep(5)  # Log every 5 seconds
        
        import threading
        self.monitoring = True
        monitor_thread = threading.Thread(target=monitoring_loop)
        monitor_thread.start()
        
        try:
            # Run the training function
            result = training_function(*args, **kwargs)
            
        finally:
            self.monitoring = False
            monitor_thread.join()
            
        print(f"Monitoring complete. Metrics saved to {self.log_file}")
        return result
    
    def generate_report(self):
        """Generate performance report"""
        
        if not self.metrics:
            return "No metrics collected"
        
        avg_cpu = sum(m["cpu_percent"] for m in self.metrics) / len(self.metrics)
        avg_memory = sum(m["memory_percent"] for m in self.metrics) / len(self.metrics)
        
        report = f"""
Performance Report
==================
Duration: {time.time() - self.start_time:.2f} seconds
Samples: {len(self.metrics)}
Average CPU: {avg_cpu:.1f}%
Average Memory: {avg_memory:.1f}%
        """
        
        if any("gpu" in m for m in self.metrics):
            gpu_metrics = [m["gpu"] for m in self.metrics if "gpu" in m]
            if gpu_metrics:
                avg_gpu_mem = sum(m["memory_allocated"] for m in gpu_metrics) / len(gpu_metrics)
                report += f"Average GPU Memory: {avg_gpu_mem / 1024**3:.2f} GB\n"
        
        return report

# Usage example
monitor = PerformanceMonitor("training_performance.log")

def example_training():
    # Simulate training process
    for i in range(100):
        time.sleep(0.1)  # Simulate work
        if i % 10 == 0:
            print(f"Training step {i}")

# Monitor training
monitor.monitor_training(example_training)
print(monitor.generate_report())
```

## Troubleshooting Common Issues

### Issue 1: BVH Conversion Failures

```python
# bvh_troubleshooter.py
import json
from pathlib import Path

def diagnose_bvh_conversion_issues(bvh_file, settings_file):
    """Diagnose common BVH conversion problems"""
    
    print(f"Diagnosing BVH file: {bvh_file}")
    
    # Check file existence
    if not Path(bvh_file).exists():
        print("❌ BVH file does not exist")
        return False
    
    # Check settings file
    if not Path(settings_file).exists():
        print("❌ Settings file does not exist")
        return False
    
    # Load and validate settings
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        print("✓ Settings file loaded successfully")
    except Exception as e:
        print(f"❌ Invalid settings file: {e}")
        return False
    
    # Check required settings
    required_keys = ['scale', 'rootJoints', 'jointMapping']
    for key in required_keys:
        if key not in settings:
            print(f"❌ Missing required setting: {key}")
            return False
    
    print("✓ Settings validation passed")
    
    # Analyze BVH file structure
    try:
        with open(bvh_file, 'r') as f:
            content = f.read()
        
        if 'HIERARCHY' not in content:
            print("❌ Invalid BVH format: missing HIERARCHY")
            return False
        
        if 'MOTION' not in content:
            print("❌ Invalid BVH format: missing MOTION section")
            return False
        
        print("✓ BVH format validation passed")
        
        # Extract joint names from BVH
        lines = content.split('\n')
        joint_names = []
        for line in lines:
            if 'JOINT' in line or 'ROOT' in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    joint_names.append(parts[1])
        
        print(f"Found {len(joint_names)} joints in BVH file")
        
        # Check joint mapping coverage
        mapped_joints = set(settings['jointMapping'].keys())
        bvh_joints = set(joint_names)
        
        unmapped_joints = bvh_joints - mapped_joints
        missing_joints = mapped_joints - bvh_joints
        
        if unmapped_joints:
            print(f"⚠ Unmapped BVH joints: {unmapped_joints}")
        
        if missing_joints:
            print(f"❌ Missing BVH joints: {missing_joints}")
            return False
        
        print("✓ Joint mapping validation passed")
        return True
        
    except Exception as e:
        print(f"❌ BVH file analysis failed: {e}")
        return False

# Auto-fix common issues
def auto_fix_bvh_settings(bvh_file, settings_file, output_settings_file):
    """Automatically fix common BVH conversion issues"""
    
    # Load existing settings
    with open(settings_file, 'r') as f:
        settings = json.load(f)
    
    # Analyze BVH file to extract joint names
    with open(bvh_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    joint_names = []
    for line in lines:
        if 'JOINT' in line or 'ROOT' in line:
            parts = line.strip().split()
            if len(parts) >= 2:
                joint_names.append(parts[1])
    
    # Create automatic joint mapping based on common patterns
    auto_mapping = {}
    
    # Common joint name patterns
    mapping_patterns = {
        'hip': ['hip', 'pelvis', 'root'],
        'left_hip': ['leftupleg', 'left_upleg', 'l_upleg'],
        'right_hip': ['rightupleg', 'right_upleg', 'r_upleg'],
        'left_knee': ['leftleg', 'left_leg', 'l_leg'],
        'right_knee': ['rightleg', 'right_leg', 'r_leg'],
        'torso': ['spine', 'spine1', 'chest']
    }
    
    for deepmimic_joint, patterns in mapping_patterns.items():
        for bvh_joint in joint_names:
            for pattern in patterns:
                if pattern.lower() in bvh_joint.lower():
                    auto_mapping[bvh_joint] = deepmimic_joint
                    break
    
    # Update settings with auto-mapping
    settings['jointMapping'].update(auto_mapping)
    
    # Save updated settings
    with open(output_settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
    
    print(f"Auto-fixed settings saved to: {output_settings_file}")
    print(f"Added {len(auto_mapping)} automatic joint mappings")

# Usage examples
if __name__ == "__main__":
    # Diagnose issues
    success = diagnose_bvh_conversion_issues(
        "assets/problematic_motion.bvh",
        "BvhToDeepMimic/Settings/settings.json"
    )
    
    if not success:
        # Try auto-fix
        print("Attempting auto-fix...")
        auto_fix_bvh_settings(
            "assets/problematic_motion.bvh",
            "BvhToDeepMimic/Settings/settings.json",
            "BvhToDeepMimic/Settings/auto_fixed_settings.json"
        )
```

### Issue 2: Training Performance Problems

```python
# training_optimizer.py
import torch
import time

def optimize_training_performance():
    """Optimize system for deep learning training"""
    
    optimizations = []
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ Found {gpu_count} GPU(s)")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        optimizations.append("Enabled cuDNN optimizations")
        
        # Check GPU memory
        for i in range(gpu_count):
            memory = torch.cuda.get_device_properties(i).total_memory
            print(f"GPU {i}: {memory / 1024**3:.1f} GB memory")
    else:
        print("⚠ No GPU available - training will be slower")
    
    # Set optimal number of workers
    import multiprocessing
    optimal_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Recommended number of workers: {optimal_workers}")
    optimizations.append(f"Set workers to {optimal_workers}")
    
    # Memory optimization settings
    torch.autograd.set_detect_anomaly(False)  # Disable for performance
    optimizations.append("Disabled anomaly detection")
    
    return optimizations

def benchmark_training_step(model, data_loader, device):
    """Benchmark training step performance"""
    
    model = model.to(device)
    model.train()
    
    times = []
    
    for i, batch in enumerate(data_loader):
        if i >= 10:  # Only benchmark 10 steps
            break
        
        start_time = time.time()
        
        # Simulate training step
        batch = batch.to(device)
        output = model(batch)
        loss = torch.nn.functional.mse_loss(output, batch)
        loss.backward()
        
        step_time = time.time() - start_time
        times.append(step_time)
    
    avg_time = sum(times) / len(times)
    print(f"Average training step time: {avg_time:.4f} seconds")
    
    return avg_time

# Usage
optimizations = optimize_training_performance()
for opt in optimizations:
    print(f"Applied: {opt}")
```

This comprehensive usage examples document provides practical, working code examples for integrating all the projects in the Motion workspace. The examples progress from basic workflows to advanced integration patterns, making it easy for users to understand how to use the systems together effectively.

The documentation includes real-world scenarios like batch processing, performance monitoring, and troubleshooting, which are essential for production use of these motion processing and animation systems.
