#!/usr/bin/env python3
"""
RSMT WebSocket Server with Real-time Neural Network Warmup
"""

import json
import asyncio
import time
import logging
from pathlib import Path
from typing import Dict, Set
import threading

try:
    import websockets
    from websockets.server import serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
connected_clients: Set = set()
model_status = {
    "deephase": {"status": "Ready", "type": "Neural Network", "last_update": time.time()},
    "stylevae": {"status": "Standby", "type": "Placeholder", "last_update": time.time()},
    "transitionnet": {"status": "Idle", "type": "Placeholder", "last_update": time.time()}
}

class HTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for serving static files"""
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            # Serve the main HTML file
            try:
                with open('rsmt_showcase_modern.html', 'r') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content.encode())
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'File not found')
        else:
            # Serve static files
            try:
                filename = parsed_path.path[1:]  # Remove leading slash
                with open(filename, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                
                # Set appropriate content type
                if filename.endswith('.html'):
                    self.send_header('Content-type', 'text/html')
                elif filename.endswith('.js'):
                    self.send_header('Content-type', 'application/javascript')
                elif filename.endswith('.bvh'):
                    self.send_header('Content-type', 'text/plain')
                else:
                    self.send_header('Content-type', 'application/octet-stream')
                
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'404 - File not found')
    
    def log_message(self, format, *args):
        # Suppress default logging
        return

async def broadcast_status_update(model_name: str, status: str, model_type: str = None, details: Dict = None):
    """Broadcast model status update to all connected clients"""
    global model_status
    
    # Update global status
    model_status[model_name].update({
        "status": status,
        "type": model_type or model_status[model_name]["type"],
        "last_update": time.time()
    })
    
    if details:
        model_status[model_name].update(details)
    
    # Prepare WebSocket message
    message = {
        "type": "status_update",
        "model": model_name,
        "status": status,
        "model_type": model_type or model_status[model_name]["type"],
        "timestamp": time.time(),
        "details": details or {}
    }
    
    # Broadcast to all connected clients
    if connected_clients:
        logger.info(f"Broadcasting {model_name} status: {status}")
        disconnected = set()
        
        for client in connected_clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        connected_clients -= disconnected

async def simulate_model_warmup(model_name: str, duration: float = 2.0):
    """Simulate neural network model warmup with real-time updates"""
    
    # Starting warmup
    await broadcast_status_update(model_name, "Initializing...", "Neural Network", 
                                {"stage": "startup", "progress": 0})
    
    await asyncio.sleep(0.5)
    
    # Loading model
    await broadcast_status_update(model_name, "Loading Model...", "Neural Network",
                                {"stage": "loading", "progress": 25})
    
    await asyncio.sleep(duration * 0.3)
    
    # Warming up
    await broadcast_status_update(model_name, "Warming Up...", "Neural Network",
                                {"stage": "warmup", "progress": 60})
    
    await asyncio.sleep(duration * 0.4)
    
    # Testing
    await broadcast_status_update(model_name, "Testing...", "Neural Network",
                                {"stage": "testing", "progress": 85})
    
    await asyncio.sleep(duration * 0.3)
    
    # Ready
    processing_time = duration + (time.time() % 0.5)  # Add some randomness
    await broadcast_status_update(model_name, "Active", "Neural Network",
                                {"stage": "ready", "progress": 100, "processing_time": processing_time})

async def handle_warmup_command(model_name: str):
    """Handle warmup command for a specific model"""
    logger.info(f"Starting warmup for {model_name}")
    
    if model_name == "stylevae":
        await simulate_model_warmup("stylevae", 2.5)
    elif model_name == "transitionnet":
        await simulate_model_warmup("transitionnet", 2.0)
    elif model_name == "all":
        # Warm up all models in parallel
        await asyncio.gather(
            simulate_model_warmup("stylevae", 2.5),
            simulate_model_warmup("transitionnet", 2.0)
        )
    else:
        await broadcast_status_update(model_name, "Unknown Model", "Error")

async def websocket_handler(websocket, path):
    """Handle WebSocket connections"""
    logger.info(f"New WebSocket connection from {websocket.remote_address}")
    connected_clients.add(websocket)
    
    try:
        # Send initial status to new client
        initial_message = {
            "type": "initial_status",
            "models": model_status,
            "server": "RSMT WebSocket Server",
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(initial_message))
        
        # Handle incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                command_type = data.get("type")
                
                if command_type == "warmup":
                    model_name = data.get("model", "unknown")
                    logger.info(f"Received warmup command for: {model_name}")
                    
                    # Handle warmup command asynchronously
                    asyncio.create_task(handle_warmup_command(model_name))
                    
                    # Send immediate acknowledgment
                    ack_message = {
                        "type": "command_ack",
                        "command": "warmup",
                        "model": model_name,
                        "status": "started",
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(ack_message))
                
                elif command_type == "ping":
                    # Respond to ping
                    pong_message = {
                        "type": "pong",
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(pong_message))
                
                elif command_type == "get_status":
                    # Send current status
                    status_message = {
                        "type": "status_response",
                        "models": model_status,
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(status_message))
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {message}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket connection closed: {websocket.remote_address}")
    finally:
        connected_clients.discard(websocket)

def run_http_server():
    """Run HTTP server in a separate thread"""
    server_address = ('', 8080)  # Use port 8080 for HTTP
    httpd = HTTPServer(server_address, HTTPHandler)
    logger.info("HTTP server running on http://localhost:8080")
    httpd.serve_forever()

async def run_websocket_server():
    """Run WebSocket server"""
    if not WEBSOCKETS_AVAILABLE:
        logger.error("websockets module not available. Install with: pip install websockets")
        return
    
    logger.info("Starting WebSocket server on ws://localhost:8765")
    
    async with serve(websocket_handler, "localhost", 8765):
        logger.info("🚀 RSMT WebSocket Server running on ws://localhost:8765")
        logger.info("📁 HTTP Server running on http://localhost:8080")
        logger.info("🧠 Real-time neural network warmup ready!")
        logger.info("🔗 WebSocket commands: warmup, ping, get_status")
        
        # Keep the server running
        await asyncio.Future()  # Run forever

def main():
    """Main server startup"""
    
    # Start HTTP server in background thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    
    # Start WebSocket server
    try:
        asyncio.run(run_websocket_server())
    except KeyboardInterrupt:
        logger.info("\n🔥 Server shutting down...")

if __name__ == '__main__':
    if not WEBSOCKETS_AVAILABLE:
        print("❌ websockets module required. Install with:")
        print("pip install websockets")
        exit(1)
    
    main()
