#!/usr/bin/env python3
"""
Motion Viewer Development Server - Phase 1
Serves static files and provides basic API endpoints for avatar and animation management
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for the dev environment
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"
VIEWER_DIR = BASE_DIR / "viewer"

# Pydantic models for API
class AvatarInfo(BaseModel):
    id: str
    name: str
    file: str
    format: str
    description: Optional[str] = None
    previewImage: Optional[str] = None
    config: Dict[str, Any] = {}

class AnimationInfo(BaseModel):
    id: str
    name: str
    file: str
    format: str
    duration: Optional[float] = None
    description: Optional[str] = None
    source: str = "chat"
    tags: List[str] = []

class EnvironmentInfo(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str] = None
    assets: List[str] = []

# Create FastAPI app
app = FastAPI(
    title="Motion Viewer Development Server",
    description="Phase 1 server for 3D avatar viewer with chat interface integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving
if VIEWER_DIR.exists():
    app.mount("/viewer", StaticFiles(directory=str(VIEWER_DIR)), name="viewer")
    
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

@app.get("/")
async def root():
    """Serve the main viewer page"""
    index_path = VIEWER_DIR / "templates" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return {"message": "Motion Viewer Development Server - Phase 1", "status": "ready"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "phase": "1",
        "description": "Chat interface integration phase",
        "features": [
            "3D avatar loading",
            "Chat animation playback", 
            "Classroom environment",
            "Basic controls"
        ]
    }

@app.get("/api/avatars", response_model=List[AvatarInfo])
async def get_avatars():
    """Get list of available avatars"""
    avatars = []
    avatar_dir = ASSETS_DIR / "avatars"
    
    # Try to load avatar index
    index_file = avatar_dir / "index.json"
    if index_file.exists():
        try:
            with open(index_file, 'r') as f:
                avatar_data = json.load(f)
                for i, avatar in enumerate(avatar_data):
                    avatars.append(AvatarInfo(
                        id=avatar.get('id', f'avatar_{i}'),
                        name=avatar.get('name', f'Avatar {i+1}'),
                        file=avatar.get('file', ''),
                        format=avatar.get('format', 'vrm'),
                        description=avatar.get('description'),
                        previewImage=avatar.get('previewImage'),
                        config=avatar.get('config', {})
                    ))
        except Exception as e:
            logger.warning(f"Could not load avatar index: {e}")
    
    # Fallback: scan directory for VRM files
    if not avatars and avatar_dir.exists():
        for file_path in avatar_dir.glob("*.vrm"):
            avatars.append(AvatarInfo(
                id=file_path.stem,
                name=file_path.stem.replace('_', ' ').title(),
                file=file_path.name,
                format="vrm",
                description="Auto-discovered VRM avatar"
            ))
    
    # Add default avatar if none found
    if not avatars:
        avatars.append(AvatarInfo(
            id="default",
            name="Default Avatar",
            file="default.vrm",
            format="vrm",
            description="Default test avatar",
            config={
                "scale": 1.0,
                "position": [0, 0, 0]
            }
        ))
    
    return avatars

@app.get("/api/animations", response_model=List[AnimationInfo])
async def get_animations():
    """Get list of available animations"""
    animations = []
    
    # Load chat animations
    chat_anim_dir = ASSETS_DIR / "animations"
    chat_index = chat_anim_dir / "index.json"
    
    if chat_index.exists():
        try:
            with open(chat_index, 'r') as f:
                chat_data = json.load(f)
                for i, anim in enumerate(chat_data):
                    animations.append(AnimationInfo(
                        id=anim.get('id', f'chat_anim_{i}'),
                        name=anim.get('name', f'Chat Animation {i+1}'),
                        file=anim.get('file', ''),
                        format=anim.get('format', 'json'),
                        duration=anim.get('duration'),
                        description=anim.get('description'),
                        source="chat",
                        tags=anim.get('tags', ['chat', 'conversation'])
                    ))
        except Exception as e:
            logger.warning(f"Could not load chat animations: {e}")
    
    # Fallback: scan for JSON animation files
    if not animations and chat_anim_dir.exists():
        for file_path in chat_anim_dir.glob("*.json"):
            if file_path.name != "index.json":
                animations.append(AnimationInfo(
                    id=file_path.stem,
                    name=file_path.stem.replace('_', ' ').title(),
                    file=file_path.name,
                    format="json",
                    source="chat",
                    description="Auto-discovered chat animation"
                ))
    
    # Add default animations if none found
    if not animations:
        default_animations = [
            {
                "id": "idle",
                "name": "Idle",
                "file": "idle.json",
                "description": "Default idle animation"
            },
            {
                "id": "wave",
                "name": "Wave",
                "file": "wave.json", 
                "description": "Greeting wave animation"
            },
            {
                "id": "talk",
                "name": "Talk",
                "file": "talk.json",
                "description": "Talking gesture animation"
            }
        ]
        
        for anim_data in default_animations:
            animations.append(AnimationInfo(
                id=anim_data["id"],
                name=anim_data["name"],
                file=anim_data["file"],
                format="json",
                source="chat",
                description=anim_data["description"],
                tags=["default", "chat"]
            ))
    
    return animations

@app.get("/api/environments", response_model=List[EnvironmentInfo])
async def get_environments():
    """Get list of available environments"""
    environments = [
        EnvironmentInfo(
            id="classroom",
            name="Classroom",
            type="educational",
            description="Educational classroom environment with desks and whiteboard"
        ),
        EnvironmentInfo(
            id="stage",
            name="Stage",
            type="performance",
            description="Performance stage with spotlights and curtains"
        ),
        EnvironmentInfo(
            id="studio",
            name="Studio",
            type="neutral",
            description="Clean studio environment with grid floor"
        ),
        EnvironmentInfo(
            id="outdoor",
            name="Outdoor",
            type="natural",
            description="Outdoor environment with grass and sky"
        )
    ]
    
    return environments

@app.get("/api/avatar/{avatar_id}")
async def get_avatar(avatar_id: str):
    """Get specific avatar information"""
    avatars = await get_avatars()
    avatar = next((a for a in avatars if a.id == avatar_id), None)
    
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    
    return avatar

@app.get("/api/animation/{animation_id}")
async def get_animation(animation_id: str):
    """Get specific animation information"""
    animations = await get_animations()
    animation = next((a for a in animations if a.id == animation_id), None)
    
    if not animation:
        raise HTTPException(status_code=404, detail="Animation not found")
    
    return animation

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    avatars = await get_avatars()
    animations = await get_animations()
    environments = await get_environments()
    
    return {
        "phase": "1",
        "counts": {
            "avatars": len(avatars),
            "animations": len(animations),
            "environments": len(environments)
        },
        "features": {
            "3d_rendering": True,
            "avatar_loading": True,
            "animation_playback": True,
            "environment_switching": True,
            "neural_networks": False,  # Phase 2
            "style_transfer": False,   # Phase 2
            "transition_generation": False  # Phase 2
        },
        "assets_directory": str(ASSETS_DIR),
        "viewer_directory": str(VIEWER_DIR)
    }

# Phase 2 placeholders (will be implemented with RSMT integration)
@app.post("/api/style-transfer")
async def style_transfer():
    """Placeholder for Phase 2 style transfer"""
    raise HTTPException(
        status_code=501, 
        detail="Style transfer will be implemented in Phase 2 with RSMT integration"
    )

@app.post("/api/generate-transition")
async def generate_transition():
    """Placeholder for Phase 2 transition generation"""
    raise HTTPException(
        status_code=501,
        detail="Transition generation will be implemented in Phase 2 with RSMT integration"
    )

@app.get("/api/styles")
async def get_styles():
    """Placeholder for Phase 2 motion styles"""
    return {
        "message": "Motion styles will be available in Phase 2",
        "phase2_features": [
            "100STYLE dataset integration",
            "Neural network style transfer",
            "Real-time motion generation",
            "Style interpolation"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Resource not found",
            "path": str(request.url.path),
            "suggestion": "Check /api/avatars, /api/animations, or /api/environments for available resources"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": "Please check server logs for details"
        }
    )

def main():
    """Run the development server"""
    logger.info("Starting Motion Viewer Development Server - Phase 1")
    logger.info(f"Assets directory: {ASSETS_DIR}")
    logger.info(f"Viewer directory: {VIEWER_DIR}")
    
    # Create directories if they don't exist
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / "avatars").mkdir(exist_ok=True)
    (ASSETS_DIR / "animations").mkdir(exist_ok=True)
    (ASSETS_DIR / "scenes").mkdir(exist_ok=True)
    
    # Start server
    uvicorn.run(
        "enhanced_motion_server:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
