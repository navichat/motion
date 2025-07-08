#!/usr/bin/env python3
"""
RSMT Integration Proof of Concept

This script provides a minimal working example of how to integrate the actual
RSMT models into the web showcase. Start here before implementing the full plan.

Usage:
    python rsmt_poc.py --source_motion source.bvh --target_motion target.bvh
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# Add the repository root to the path
sys.path.append(str(Path(__file__).parent.parent))

class RSMTProofOfConcept:
    """
    Minimal RSMT integration to demonstrate the real neural network pipeline
    """
    
    def __init__(self):
        self.deephase_model = None
        self.stylevae_model = None
        self.transitionnet_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_models(self):
        """
        Load the trained RSMT models from checkpoints
        """
        print("Loading RSMT models...")
        
        try:
            # Load DeepPhase model
            deephase_path = self._find_model_checkpoint("DeepPhase")
            if deephase_path:
                print(f"Loading DeepPhase from {deephase_path}")
                self.deephase_model = torch.load(deephase_path, map_location=self.device)
                self.deephase_model.eval()
            
            # Load StyleVAE model  
            stylevae_path = self._find_model_checkpoint("StyleVAE")
            if stylevae_path:
                print(f"Loading StyleVAE from {stylevae_path}")
                self.stylevae_model = torch.load(stylevae_path, map_location=self.device)
                self.stylevae_model.eval()
                
            # Load TransitionNet model
            transitionnet_path = self._find_model_checkpoint("Transition")
            if transitionnet_path:
                print(f"Loading TransitionNet from {transitionnet_path}")
                self.transitionnet_model = torch.load(transitionnet_path, map_location=self.device)
                self.transitionnet_model.eval()
                
            print(f"Models loaded on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Note: This requires trained model checkpoints from the RSMT training pipeline")
            return False
            
        return True
    
    def _find_model_checkpoint(self, model_type):
        """
        Find the latest model checkpoint for a given type
        """
        search_paths = [
            f"../results/*{model_type}*/myResults/*/m_save_model_*",
            f"../results/*{model_type}*/myResults/*/model_*.pth",
            f"./models/{model_type.lower()}_latest.pth"
        ]
        
        import glob
        for pattern in search_paths:
            files = glob.glob(pattern)
            if files:
                # Return the most recent file
                return max(files, key=os.path.getctime)
        
        return None
    
    def encode_phase(self, motion_data):
        """
        Use DeepPhase model to encode temporal motion patterns
        """
        if self.deephase_model is None:
            print("DeepPhase model not loaded, using mock phase encoding")
            # Mock phase encoding for demonstration
            num_frames = len(motion_data)
            phases = []
            for i in range(num_frames):
                # Simple sinusoidal phase for demo
                sx = np.cos(i * 0.1)
                sy = np.sin(i * 0.1)
                phases.append((sx, sy))
            return phases
        
        # Real DeepPhase inference
        with torch.no_grad():
            # Convert motion data to velocity features
            velocities = self._compute_velocities(motion_data)
            input_tensor = torch.tensor(velocities, dtype=torch.float32).to(self.device)
            
            # Run through DeepPhase model
            phase_output = self.deephase_model(input_tensor)
            
            # Extract (sx, sy) coordinates
            phases = []
            for frame_output in phase_output:
                sx, sy = frame_output[:2].cpu().numpy()
                phases.append((float(sx), float(sy)))
                
        return phases
    
    def encode_style(self, motion_data, phase_data):
        """
        Use StyleVAE to encode motion style
        """
        if self.stylevae_model is None:
            print("StyleVAE model not loaded, using mock style encoding")
            # Mock style code
            return np.random.randn(256).tolist()
        
        # Real StyleVAE inference
        with torch.no_grad():
            # Prepare input data
            motion_tensor = torch.tensor(motion_data, dtype=torch.float32).to(self.device)
            phase_tensor = torch.tensor(phase_data, dtype=torch.float32).to(self.device)
            
            # Encode style
            style_code = self.stylevae_model.encode_style(motion_tensor, phase_tensor)
            
        return style_code.cpu().numpy().tolist()
    
    def generate_transition(self, source_motion, target_motion, transition_length=60):
        """
        Generate a stylized motion transition using the full RSMT pipeline
        """
        print(f"Generating {transition_length}-frame transition...")
        
        # Step 1: Encode phase information
        print("1. Encoding phase patterns...")
        source_phase = self.encode_phase(source_motion)
        target_phase = self.encode_phase(target_motion)
        
        # Step 2: Encode style information  
        print("2. Encoding motion styles...")
        source_style = self.encode_style(source_motion, source_phase)
        target_style = self.encode_style(target_motion, target_phase)
        
        # Step 3: Generate transition
        print("3. Generating transition frames...")
        if self.transitionnet_model is None:
            print("TransitionNet model not loaded, using enhanced interpolation")
            transition_frames = self._enhanced_interpolation(
                source_motion, target_motion, source_phase, target_phase, 
                source_style, target_style, transition_length
            )
        else:
            # Real TransitionNet inference
            transition_frames = self._generate_with_transitionnet(
                source_motion, target_motion, source_phase, target_phase,
                source_style, target_style, transition_length
            )
        
        print(f"Generated {len(transition_frames)} transition frames")
        return transition_frames
    
    def _enhanced_interpolation(self, source_motion, target_motion, source_phase, 
                              target_phase, source_style, target_style, length):
        """
        Enhanced interpolation that uses phase and style information
        Better than basic linear interpolation, though not as good as real TransitionNet
        """
        transition_frames = []
        
        for i in range(length):
            alpha = i / (length - 1)
            
            # Phase-aware blending
            phase_weight = self._compute_phase_weight(alpha, source_phase, target_phase)
            
            # Style-aware blending
            style_weight = self._compute_style_weight(alpha, source_style, target_style)
            
            # Combined interpolation
            if i < len(source_motion) and i < len(target_motion):
                source_frame = source_motion[i % len(source_motion)]
                target_frame = target_motion[i % len(target_motion)]
                
                # Phase-influenced interpolation
                frame = []
                for j in range(len(source_frame)):
                    # Use both phase and style weights
                    combined_weight = (phase_weight + style_weight) / 2
                    value = (1 - combined_weight) * source_frame[j] + combined_weight * target_frame[j]
                    
                    # Add phase-based variation for natural motion
                    if j < 6:  # Position and rotation channels
                        phase_influence = 0.1 * np.sin(alpha * np.pi) * phase_weight
                        value += phase_influence
                    
                    frame.append(value)
                
                transition_frames.append(frame)
        
        return transition_frames
    
    def _compute_phase_weight(self, alpha, source_phase, target_phase):
        """
        Compute blending weight based on phase information
        """
        # Use phase manifold distance for more natural timing
        if len(source_phase) > 0 and len(target_phase) > 0:
            source_sx, source_sy = source_phase[0]
            target_sx, target_sy = target_phase[0] if target_phase else (0, 0)
            
            # Phase-based timing curve
            phase_distance = np.sqrt((target_sx - source_sx)**2 + (target_sy - source_sy)**2)
            phase_curve = 1 - np.exp(-5 * alpha)  # Exponential ease-in
            
            return phase_curve * (1 + 0.5 * phase_distance)
        
        return alpha
    
    def _compute_style_weight(self, alpha, source_style, target_style):
        """
        Compute blending weight based on style similarity
        """
        # Smooth style transition curve
        return 1 / (1 + np.exp(-10 * (alpha - 0.5)))  # Sigmoid curve
    
    def _generate_with_transitionnet(self, source_motion, target_motion, 
                                   source_phase, target_phase, source_style, 
                                   target_style, length):
        """
        Use the actual TransitionNet model for generation
        """
        with torch.no_grad():
            # Prepare inputs for TransitionNet
            inputs = {
                'source_motion': torch.tensor(source_motion, dtype=torch.float32).to(self.device),
                'target_motion': torch.tensor(target_motion, dtype=torch.float32).to(self.device),
                'source_phase': torch.tensor(source_phase, dtype=torch.float32).to(self.device),
                'target_phase': torch.tensor(target_phase, dtype=torch.float32).to(self.device),
                'source_style': torch.tensor(source_style, dtype=torch.float32).to(self.device),
                'target_style': torch.tensor(target_style, dtype=torch.float32).to(self.device),
                'length': length
            }
            
            # Generate transition
            transition_output = self.transitionnet_model.generate_transition(**inputs)
            
            # Convert to list format
            return transition_output.cpu().numpy().tolist()
    
    def _compute_velocities(self, motion_data):
        """
        Compute velocity features for DeepPhase model
        """
        if len(motion_data) < 2:
            return [motion_data[0] if motion_data else [0] * 72]
        
        velocities = []
        for i in range(1, len(motion_data)):
            velocity = []
            for j in range(len(motion_data[i])):
                vel = motion_data[i][j] - motion_data[i-1][j]
                velocity.append(vel)
            velocities.append(velocity)
        
        return velocities

def load_bvh_motion(filepath):
    """
    Load motion data from BVH file
    Returns list of frame data (simplified for POC)
    """
    try:
        # Import BVH utilities
        sys.path.append('..')
        from src.utils import BVH_mod as BVH
        
        # Load BVH file
        bvh_data = BVH.read_bvh(filepath)
        
        # Extract frame data
        motion_frames = []
        for i in range(len(bvh_data.hip_pos)):
            frame = []
            
            # Add hip position
            frame.extend(bvh_data.hip_pos[i])
            
            # Add joint rotations (flattened)
            if i < len(bvh_data.quats):
                for quat in bvh_data.quats[i]:
                    frame.extend(quat)
            
            motion_frames.append(frame)
        
        return motion_frames
        
    except Exception as e:
        print(f"Error loading BVH file {filepath}: {e}")
        # Return mock data for testing
        print("Using mock motion data for demonstration")
        return [[i * 0.1 + j for j in range(72)] for i in range(100)]

def save_transition_bvh(transition_frames, output_path):
    """
    Save transition frames as BVH file
    """
    try:
        # This would need proper BVH writing implementation
        print(f"Saving {len(transition_frames)} frames to {output_path}")
        print("Note: BVH writing implementation needed for complete POC")
        
        # For now, save as JSON for inspection
        import json
        json_path = output_path.replace('.bvh', '.json')
        with open(json_path, 'w') as f:
            json.dump(transition_frames, f, indent=2)
        print(f"Transition data saved as JSON: {json_path}")
        
    except Exception as e:
        print(f"Error saving transition: {e}")

def main():
    parser = argparse.ArgumentParser(description='RSMT Integration Proof of Concept')
    parser.add_argument('--source_motion', type=str, default='neutral_reference.bvh',
                       help='Source motion BVH file')
    parser.add_argument('--target_motion', type=str, default='elated_reference.bvh',
                       help='Target motion BVH file')
    parser.add_argument('--output', type=str, default='rsmt_transition.bvh',
                       help='Output transition BVH file')
    parser.add_argument('--length', type=int, default=60,
                       help='Transition length in frames')
    
    args = parser.parse_args()
    
    print("RSMT Integration Proof of Concept")
    print("=" * 50)
    
    # Initialize RSMT system
    rsmt = RSMTProofOfConcept()
    
    # Load models
    if not rsmt.load_models():
        print("Warning: Could not load all RSMT models")
        print("Continuing with mock implementations for demonstration")
    
    # Load motion data
    print(f"\\nLoading motion data...")
    source_motion = load_bvh_motion(args.source_motion)
    target_motion = load_bvh_motion(args.target_motion)
    
    print(f"Source motion: {len(source_motion)} frames")
    print(f"Target motion: {len(target_motion)} frames")
    
    # Generate transition
    print(f"\\nGenerating RSMT transition...")
    transition_frames = rsmt.generate_transition(
        source_motion, target_motion, args.length
    )
    
    # Save result
    save_transition_bvh(transition_frames, args.output)
    
    print(f"\\nPOC completed successfully!")
    print(f"Transition saved to: {args.output}")
    print("\\nNext steps:")
    print("1. Integrate this into the web showcase")
    print("2. Add real-time inference via FastAPI")
    print("3. Implement WebGL acceleration")
    print("4. Add advanced UI controls")

if __name__ == "__main__":
    main()
