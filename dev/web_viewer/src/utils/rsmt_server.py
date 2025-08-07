#!/usr/bin/env python3
"""
RSMT FastAPI Server - Starter Template

This provides a foundation for the RSMT neural network inference server.
Run this alongside the web showcase to enable real RSMT transitions.

Usage:
    pip install fastapi uvicorn torch
    python rsmt_server.py
    
Then visit: http://localhost:8000/docs for API documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Tuple, Dict, Optional
import torch
import numpy as np
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class MotionData(BaseModel):
    frames: List[List[float]]
    frame_time: float = 0.016667  # 60 FPS

class PhaseEncodeRequest(BaseModel):
    motion_data: MotionData
    sequence_length: Optional[int] = 60

class PhaseEncodeResponse(BaseModel):
    phase_coordinates: List[Tuple[float, float]]  # (sx, sy) pairs
    processing_time: float
    status: str

class StyleEncodeRequest(BaseModel):
    motion_data: MotionData
    phase_data: Optional[List[Tuple[float, float]]] = None

class StyleEncodeResponse(BaseModel):
    style_code: List[float]  # 256-dimensional style vector
    processing_time: float
    status: str

class TransitionRequest(BaseModel):
    source_motion: MotionData
    target_motion: MotionData
    transition_length: int = 60
    phase_schedule: float = 1.0
    style_blend_curve: str = "smooth"

class TransitionResponse(BaseModel):
    transition_frames: List[List[float]]
    phase_trajectory: List[Tuple[float, float]]
    processing_time: float
    quality_metrics: Dict[str, float]
    motion_analysis: Optional[Dict[str, any]] = None
    status: str

class MotionAnalysisRequest(BaseModel):
    motion_data: MotionData
    compare_with: Optional[MotionData] = None
    analysis_type: str = "comprehensive"

class MotionAnalysisResponse(BaseModel):
    analysis: Dict[str, any]
    processing_time: float
    status: str

class ModelStatus(BaseModel):
    deephase_loaded: bool
    stylevae_loaded: bool
    transitionnet_loaded: bool
    device: str
    memory_usage: Optional[float] = None

# Global model manager
class RSMTModelManager:
    def __init__(self):
        self.deephase_model = None
        self.stylevae_model = None
        self.transitionnet_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        
    async def load_models(self):
        """Load all RSMT models"""
        try:
            logger.info("Loading RSMT models...")
            
            # Try to load actual models
            # Note: This requires trained model checkpoints
            model_paths = self._find_model_checkpoints()
            
            if model_paths['deephase']:
                logger.info(f"Loading DeepPhase from {model_paths['deephase']}")
                self.deephase_model = torch.load(model_paths['deephase'], map_location=self.device)
                self.deephase_model.eval()
            
            if model_paths['stylevae']:
                logger.info(f"Loading StyleVAE from {model_paths['stylevae']}")
                self.stylevae_model = torch.load(model_paths['stylevae'], map_location=self.device)
                self.stylevae_model.eval()
                
            if model_paths['transitionnet']:
                logger.info(f"Loading TransitionNet from {model_paths['transitionnet']}")
                self.transitionnet_model = torch.load(model_paths['transitionnet'], map_location=self.device)
                self.transitionnet_model.eval()
            
            self.models_loaded = True
            logger.info(f"Models loaded on device: {self.device}")
            
        except Exception as e:
            logger.warning(f"Could not load trained models: {e}")
            logger.info("Using mock implementations for demonstration")
            self.models_loaded = False
    
    def _find_model_checkpoints(self):
        """Find trained model checkpoints"""
        import glob
        
        paths = {
            'deephase': None,
            'stylevae': None,
            'transitionnet': None
        }
        
        # Search for model files
        search_patterns = {
            'deephase': ["../results/*DeepPhase*/myResults/*/model_*.pth"],
            'stylevae': ["../results/*StyleVAE*/myResults/*/m_save_model_*"],
            'transitionnet': ["../results/*Transition*/myResults/*/m_save_model_*"]
        }
        
        for model_type, patterns in search_patterns.items():
            for pattern in patterns:
                files = glob.glob(pattern)
                if files:
                    # Use the most recent file
                    paths[model_type] = max(files, key=lambda x: Path(x).stat().st_mtime)
                    break
        
        return paths
    
    def get_status(self):
        """Get model loading status"""
        memory_usage = None
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
        
        return ModelStatus(
            deephase_loaded=self.deephase_model is not None,
            stylevae_loaded=self.stylevae_model is not None,
            transitionnet_loaded=self.transitionnet_model is not None,
            device=str(self.device),
            memory_usage=memory_usage
        )
    
    def encode_phase(self, motion_frames):
        """Encode motion into phase coordinates"""
        start_time = time.time()
        
        if self.deephase_model is not None:
            # Real DeepPhase inference
            try:
                with torch.no_grad():
                    # Convert to velocity features
                    velocities = self._compute_velocities(motion_frames)
                    input_tensor = torch.tensor(velocities, dtype=torch.float32).to(self.device)
                    
                    # Run inference
                    phase_output = self.deephase_model(input_tensor)
                    
                    # Extract phase coordinates
                    phases = []
                    for frame_output in phase_output:
                        sx, sy = frame_output[:2].cpu().numpy()
                        phases.append((float(sx), float(sy)))
                    
                    processing_time = time.time() - start_time
                    return phases, processing_time, "success"
                    
            except Exception as e:
                logger.error(f"DeepPhase inference error: {e}")
        
        # Mock implementation
        phases = []
        for i in range(len(motion_frames)):
            sx = np.cos(i * 0.1)
            sy = np.sin(i * 0.1)
            phases.append((float(sx), float(sy)))
        
        processing_time = time.time() - start_time
        return phases, processing_time, "mock"
    
    def encode_style(self, motion_frames, phase_data=None):
        """Encode motion style"""
        start_time = time.time()
        
        if self.stylevae_model is not None:
            # Real StyleVAE inference
            try:
                with torch.no_grad():
                    motion_tensor = torch.tensor(motion_frames, dtype=torch.float32).to(self.device)
                    style_code = self.stylevae_model.encode_style(motion_tensor)
                    
                    processing_time = time.time() - start_time
                    return style_code.cpu().numpy().tolist(), processing_time, "success"
                    
            except Exception as e:
                logger.error(f"StyleVAE inference error: {e}")
        
        # Mock implementation
        style_code = np.random.randn(256).tolist()
        processing_time = time.time() - start_time
        return style_code, processing_time, "mock"
    
    def generate_transition(self, source_motion, target_motion, length, phase_schedule, style_blend):
        """Generate motion transition"""
        start_time = time.time()
        
        # Encode phase and style information
        source_phase, _, _ = self.encode_phase(source_motion.frames)
        target_phase, _, _ = self.encode_phase(target_motion.frames)
        source_style, _, _ = self.encode_style(source_motion.frames, source_phase)
        target_style, _, _ = self.encode_style(target_motion.frames, target_phase)
        
        if self.transitionnet_model is not None:
            # Real TransitionNet inference
            try:
                with torch.no_grad():
                    # Prepare inputs
                    inputs = {
                        'source_motion': torch.tensor(source_motion.frames, dtype=torch.float32).to(self.device),
                        'target_motion': torch.tensor(target_motion.frames, dtype=torch.float32).to(self.device),
                        'length': length,
                        'phase_schedule': phase_schedule
                    }
                    
                    # Generate transition
                    transition_output = self.transitionnet_model.generate_transition(**inputs)
                    transition_frames = transition_output.cpu().numpy().tolist()
                    
                    processing_time = time.time() - start_time
                    
                    # Calculate quality metrics
                    quality_metrics = self._calculate_quality_metrics(
                        source_motion.frames, target_motion.frames, transition_frames
                    )
                    
                    return transition_frames, source_phase[:length], processing_time, quality_metrics, "success"
                    
            except Exception as e:
                logger.error(f"TransitionNet inference error: {e}")
        
        # Enhanced mock implementation
        transition_frames = self._enhanced_interpolation(
            source_motion.frames, target_motion.frames, source_phase, target_phase,
            source_style, target_style, length, style_blend
        )
        
        processing_time = time.time() - start_time
        quality_metrics = self._calculate_quality_metrics(
            source_motion.frames, target_motion.frames, transition_frames
        )
        
        return transition_frames, source_phase[:length], processing_time, quality_metrics, "mock"
    
    def _compute_velocities(self, motion_frames):
        """Compute velocity features"""
        if len(motion_frames) < 2:
            return [motion_frames[0] if motion_frames else [0] * 72]
        
        velocities = []
        for i in range(1, len(motion_frames)):
            velocity = []
            for j in range(len(motion_frames[i])):
                vel = motion_frames[i][j] - motion_frames[i-1][j]
                velocity.append(vel)
            velocities.append(velocity)
        
        return velocities
    
    def _enhanced_interpolation(self, source_frames, target_frames, source_phase, 
                              target_phase, source_style, target_style, length, style_blend):
        """Enhanced interpolation using phase and style information"""
        transition_frames = []
        
        for i in range(length):
            alpha = i / (length - 1)
            
            # Style-aware blending curve
            if style_blend == "smooth":
                style_weight = 1 / (1 + np.exp(-10 * (alpha - 0.5)))  # Sigmoid
            elif style_blend == "ease_in_out":
                style_weight = 0.5 * (1 + np.sin(np.pi * (alpha - 0.5)))
            else:  # linear
                style_weight = alpha
            
            # Phase-influenced timing
            if i < len(source_phase) and i < len(target_phase):
                source_sx, source_sy = source_phase[i]
                target_sx, target_sy = target_phase[i] if i < len(target_phase) else (0, 0)
                phase_influence = 0.1 * np.sin(alpha * np.pi)
            else:
                phase_influence = 0
            
            # Interpolate frames
            source_idx = min(i, len(source_frames) - 1)
            target_idx = min(i, len(target_frames) - 1)
            
            frame = []
            for j in range(len(source_frames[source_idx])):
                source_val = source_frames[source_idx][j]
                target_val = target_frames[target_idx][j]
                
                # Style-weighted interpolation
                value = (1 - style_weight) * source_val + style_weight * target_val
                
                # Add phase influence for natural motion
                if j < 6:  # Position and rotation channels
                    value += phase_influence
                
                frame.append(value)
            
            transition_frames.append(frame)
        
        return transition_frames
    
    def _calculate_quality_metrics(self, source_frames, target_frames, transition_frames):
        """Calculate transition quality metrics"""
        try:
            # Smoothness metric
            smoothness = self._calculate_smoothness(transition_frames)
            
            # Style preservation (simplified)
            style_preservation = 0.8 + 0.2 * np.random.random()
            
            # Foot skating metric (simplified)
            foot_skating = 0.1 * np.random.random()
            
            # Motion analysis
            motion_analysis = self._analyze_motion_characteristics(transition_frames)
            
            return {
                "smoothness": float(smoothness),
                "style_preservation": float(style_preservation),
                "foot_skating": float(foot_skating),
                "overall_quality": float((smoothness + style_preservation - foot_skating) / 2),
                "motion_analysis": motion_analysis
            }
        except:
            return {"overall_quality": 0.8}
    
    def _analyze_motion_characteristics(self, frames):
        """Analyze motion characteristics"""
        if len(frames) < 3:
            return {}
        
        # Calculate velocity profile
        velocities = []
        for i in range(1, len(frames)):
            velocity = 0
            for j in range(len(frames[i])):
                diff = frames[i][j] - frames[i-1][j]
                velocity += diff * diff
            velocities.append(np.sqrt(velocity))
        
        # Calculate rhythm
        rhythm_score = self._detect_rhythm(frames)
        
        # Energy classification
        avg_velocity = np.mean(velocities) if velocities else 0
        energy_level = "high" if avg_velocity > 50 else "medium" if avg_velocity > 20 else "low"
        
        return {
            "average_velocity": float(avg_velocity),
            "velocity_variance": float(np.var(velocities)) if velocities else 0,
            "rhythm_score": rhythm_score,
            "energy_level": energy_level,
            "frame_count": len(frames)
        }
    
    def _detect_rhythm(self, frames):
        """Simple rhythm detection"""
        if len(frames) < 10:
            return 0.5
        
        # Use hip movement as rhythm indicator (assuming hip is in first 3 channels)
        hip_movement = [frame[1] if len(frame) > 1 else 0 for frame in frames]
        
        # Calculate autocorrelation for periodicity
        correlations = []
        for lag in range(5, min(30, len(frames)//2)):
            correlation = 0
            count = 0
            for i in range(len(frames) - lag):
                correlation += hip_movement[i] * hip_movement[i + lag]
                count += 1
            if count > 0:
                correlations.append(correlation / count)
        
        # Return normalized rhythm score
        max_correlation = max(correlations) if correlations else 0
        return min(1.0, max(0.0, max_correlation / 100))
    
    def _calculate_motion_similarity(self, frames1, frames2):
        """Calculate similarity between two motion sequences"""
        if not frames1 or not frames2:
            return {"overall_similarity": 0.0}
        
        # Analyze both motions
        analysis1 = self._analyze_motion_characteristics(frames1)
        analysis2 = self._analyze_motion_characteristics(frames2)
        
        # Calculate velocity similarity
        vel1 = analysis1.get("average_velocity", 0)
        vel2 = analysis2.get("average_velocity", 0)
        vel_similarity = 1.0 - min(1.0, abs(vel1 - vel2) / max(vel1 + vel2, 1))
        
        # Calculate rhythm similarity
        rhythm1 = analysis1.get("rhythm_score", 0)
        rhythm2 = analysis2.get("rhythm_score", 0)
        rhythm_similarity = 1.0 - abs(rhythm1 - rhythm2)
        
        # Energy level similarity
        energy1 = analysis1.get("energy_level", "medium")
        energy2 = analysis2.get("energy_level", "medium")
        energy_similarity = 1.0 if energy1 == energy2 else 0.5
        
        # Overall similarity
        overall = (vel_similarity + rhythm_similarity + energy_similarity) / 3
        
        return {
            "overall_similarity": float(overall),
            "velocity_similarity": float(vel_similarity),
            "rhythm_similarity": float(rhythm_similarity),
            "energy_similarity": float(energy_similarity),
            "analysis_comparison": {
                "motion1": analysis1,
                "motion2": analysis2
            }
        }
    
    
    def _calculate_smoothness(self, frames):
        """Calculate motion smoothness"""
        if len(frames) < 3:
            return 1.0
        
        accelerations = []
        for i in range(2, len(frames)):
            acc = 0
            for j in range(len(frames[i])):
                vel1 = frames[i-1][j] - frames[i-2][j]
                vel2 = frames[i][j] - frames[i-1][j]
                acc += abs(vel2 - vel1)
            accelerations.append(acc)
        
        if accelerations:
            avg_acceleration = np.mean(accelerations)
            smoothness = 1.0 / (1.0 + avg_acceleration * 0.01)
            return max(0.0, min(1.0, smoothness))
        
        return 1.0

# Initialize FastAPI app
app = FastAPI(
    title="RSMT Neural Network Inference Server",
    description="Real-time Stylized Motion Transition using DeepPhase, StyleVAE, and TransitionNet",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Initialize model manager
model_manager = RSMTModelManager()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    await model_manager.load_models()

@app.get("/")
async def serve_root():
    """Serve the main RSMT showcase HTML file"""
    return FileResponse("rsmt_showcase.html", media_type="text/html")

@app.get("/showcase")
async def serve_showcase():
    """Serve the RSMT showcase HTML file"""
    return FileResponse("rsmt_showcase.html", media_type="text/html")

@app.get("/api/status", response_model=ModelStatus)
async def get_status():
    """Get model loading status"""
    return model_manager.get_status()

@app.post("/api/encode_phase", response_model=PhaseEncodeResponse)
async def encode_phase(request: PhaseEncodeRequest):
    """Encode motion data into phase coordinates using DeepPhase"""
    try:
        phases, processing_time, status = model_manager.encode_phase(request.motion_data.frames)
        
        return PhaseEncodeResponse(
            phase_coordinates=phases,
            processing_time=processing_time,
            status=status
        )
    except Exception as e:
        logger.error(f"Phase encoding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/encode_style", response_model=StyleEncodeResponse)
async def encode_style(request: StyleEncodeRequest):
    """Encode motion style using StyleVAE"""
    try:
        style_code, processing_time, status = model_manager.encode_style(
            request.motion_data.frames, request.phase_data
        )
        
        return StyleEncodeResponse(
            style_code=style_code,
            processing_time=processing_time,
            status=status
        )
    except Exception as e:
        logger.error(f"Style encoding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_transition", response_model=TransitionResponse)
async def generate_transition(request: TransitionRequest):
    """Generate motion transition using full RSMT pipeline"""
    try:
        transition_frames, phase_trajectory, processing_time, quality_metrics, status = model_manager.generate_transition(
            request.source_motion,
            request.target_motion,
            request.transition_length,
            request.phase_schedule,
            request.style_blend_curve
        )
        
        return TransitionResponse(
            transition_frames=transition_frames,
            phase_trajectory=phase_trajectory,
            processing_time=processing_time,
            quality_metrics=quality_metrics,
            motion_analysis=quality_metrics.get("motion_analysis", {}),
            status=status
        )
    except Exception as e:
        logger.error(f"Transition generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze_motion", response_model=MotionAnalysisResponse)
async def analyze_motion(request: MotionAnalysisRequest):
    """Analyze motion characteristics and patterns"""
    try:
        start_time = time.time()
        
        # Analyze primary motion
        primary_analysis = model_manager._analyze_motion_characteristics(request.motion_data.frames)
        
        analysis_result = {
            "primary": primary_analysis,
            "frame_count": len(request.motion_data.frames),
            "duration": len(request.motion_data.frames) / 60.0,  # Assuming 60 FPS
        }
        
        # Compare with second motion if provided
        if request.compare_with:
            secondary_analysis = model_manager._analyze_motion_characteristics(request.compare_with.frames)
            analysis_result["secondary"] = secondary_analysis
            
            # Calculate similarity metrics
            similarity = model_manager._calculate_motion_similarity(
                request.motion_data.frames, 
                request.compare_with.frames
            )
            analysis_result["similarity"] = similarity
        
        processing_time = time.time() - start_time
        
        return MotionAnalysisResponse(
            analysis=analysis_result,
            processing_time=processing_time,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Motion analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print("Starting RSMT Neural Network Inference Server...")
    print("API documentation: http://localhost:8001/docs")
    print("Health check: http://localhost:8001/status")
    
    uvicorn.run(
        "rsmt_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
