#!/usr/bin/env python3
"""
Simple warmup server for RSMT neural models
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

class WarmupHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            status_data = {
                "status": "online",
                "server": "RSMT Warmup Server",
                "ai_status": "Partial AI",
                "models_using_ai": "1/3",
                "models": {
                    "deephase": {
                        "loaded": True,
                        "type": "Neural Network",
                        "status": "Active",
                        "capabilities": ["Phase Processing"],
                        "parameters": "2.1M"
                    },
                    "stylevae": {
                        "loaded": True,
                        "type": "Placeholder",
                        "status": "Standby",
                        "capabilities": ["Style Encoding"],
                        "parameters": "1.5M"
                    },
                    "transitionnet": {
                        "loaded": True,
                        "type": "Placeholder",
                        "status": "Idle",
                        "capabilities": ["Motion Transitions"],
                        "parameters": "3.2M"
                    }
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "models_initialized": True,
                "performance_mode": "Simulation Mode"
            }
            
            self.wfile.write(json.dumps(status_data, indent=2).encode())
            
        elif parsed_path.path == '/':
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
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        parsed_path = urlparse(self.path)
        
        # Common response headers
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        
        if parsed_path.path == '/api/encode_style':
            # Simulate StyleVAE warmup
            time.sleep(0.5)  # Simulate processing time
            
            response_data = {
                "style_code": [0.1] * 256,  # Dummy 256-dim style vector
                "processing_time": 0.521,
                "status": "success"
            }
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            
        elif parsed_path.path == '/api/generate_transition':
            # Simulate TransitionNet warmup
            time.sleep(0.3)  # Simulate processing time
            
            response_data = {
                "transition_frames": [[0] * 66 for _ in range(10)],  # Dummy transition frames
                "quality_metrics": {"smoothness": 0.85, "naturalness": 0.92},
                "processing_time": 0.312,
                "status": "success"
            }
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode())
    
    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        return

def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, WarmupHandler)
    print(f"🚀 RSMT Warmup Server running on http://localhost:8000")
    print(f"📊 API Status: http://localhost:8000/api/status")
    print(f"🎭 Main Interface: http://localhost:8000/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🔥 Server shutting down...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
