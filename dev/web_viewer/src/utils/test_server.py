#!/usr/bin/env python3
"""
Minimal FastAPI server to test connectivity
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI(title="RSMT Test Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    return {"message": "RSMT Test Server is running"}

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "server": "RSMT Test Server",
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
