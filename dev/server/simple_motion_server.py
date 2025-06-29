#!/usr/bin/env python3
"""
Simple Motion Viewer Server
A lightweight server to serve the Motion Viewer dashboard on port 8081
"""

import os
import http.server
import socketserver
import json
from urllib.parse import urlparse, parse_qs
import mimetypes

class MotionViewerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Parse URL
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Serve the main dashboard at root
        if path == '/' or path == '':
            self.serve_dashboard()
        # API endpoints
        elif path.startswith('/api/'):
            self.handle_api(path)
        # Static files from viewer directory
        elif path.startswith('./') or path.endswith('.js') or path.endswith('.css'):
            self.serve_static_file(path)
        # Static files
        else:
            super().do_GET()
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        try:
            # Look for dashboard in the viewer directory
            server_dir = os.path.dirname(__file__)
            viewer_dir = os.path.join(os.path.dirname(server_dir), 'viewer')
            dashboard_path = os.path.join(viewer_dir, 'motion-dashboard.html')
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Content-Length', str(len(content.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except FileNotFoundError:
            self.send_error(404, "Dashboard not found")
    
    def serve_static_file(self, path):
        """Serve static files from the viewer directory"""
        try:
            # Remove leading ./ if present
            clean_path = path.lstrip('./')
            
            # Get file path in viewer directory
            server_dir = os.path.dirname(__file__)
            viewer_dir = os.path.join(os.path.dirname(server_dir), 'viewer')
            file_path = os.path.join(viewer_dir, clean_path)
            
            # Security check - ensure file is within viewer directory
            if not os.path.abspath(file_path).startswith(os.path.abspath(viewer_dir)):
                self.send_error(403, "Access denied")
                return
            
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            
            self.send_response(200)
            self.send_header('Content-type', mime_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            
        except FileNotFoundError:
            self.send_error(404, f"File not found: {path}")
        except Exception as e:
            self.send_error(500, f"Error serving file: {str(e)}")
    
    def handle_api(self, path):
        """Handle API requests"""
        if path == '/api/avatars':
            self.send_json({
                "avatars": [
                    {"id": "teacher", "name": "Teacher Avatar", "icon": "👩‍🏫"},
                    {"id": "student", "name": "Student Avatar", "icon": "👨‍🎓"},
                    {"id": "default", "name": "Default Avatar", "icon": "🧑"}
                ]
            })
        elif path == '/api/animations':
            self.send_json({
                "animations": [
                    {"id": "idle", "name": "Idle Pose", "icon": "🧘", "duration": 0},
                    {"id": "wave", "name": "Waving", "icon": "👋", "duration": 3},
                    {"id": "walk", "name": "Walking", "icon": "🚶", "duration": 5},
                    {"id": "point", "name": "Pointing", "icon": "👉", "duration": 2},
                    {"id": "present", "name": "Presenting", "icon": "📊", "duration": 3},
                    {"id": "think", "name": "Thinking", "icon": "🤔", "duration": 4}
                ]
            })
        elif path == '/api/environments':
            self.send_json({
                "environments": [
                    {"id": "classroom", "name": "Classroom", "description": "Traditional classroom setting"},
                    {"id": "office", "name": "Office", "description": "Modern office environment"},
                    {"id": "stage", "name": "Stage", "description": "Presentation stage"}
                ]
            })
        elif path == '/api/stats':
            self.send_json({
                "status": "online",
                "port": 8081,
                "avatars_count": 3,
                "animations_count": 28,
                "environments_count": 3,
                "server_info": {
                    "version": "1.0.0",
                    "name": "Motion Viewer",
                    "description": "3D Avatar Dashboard"
                }
            })
        else:
            self.send_error(404, "API endpoint not found")
    
    def send_json(self, data):
        """Send JSON response"""
        json_data = json.dumps(data, indent=2)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', str(len(json_data.encode('utf-8'))))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json_data.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[Motion Viewer] {self.address_string()} - {format % args}")

def start_server(port=8081):
    """Start the Motion Viewer server"""
    try:
        # Change to the viewer directory
        viewer_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(viewer_dir)
        
        # Start server
        with socketserver.TCPServer(("", port), MotionViewerHandler) as httpd:
            print(f"🚀 Motion Viewer Dashboard starting...")
            print(f"📍 Server running at: http://localhost:{port}")
            print(f"🎯 Dashboard: http://localhost:{port}")
            print(f"📊 API Documentation: http://localhost:{port}/api/stats")
            print(f"📁 Serving from: {viewer_dir}")
            print(f"⏹️  Press Ctrl+C to stop")
            print()
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 Server stopped by user")
                httpd.shutdown()
    
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Error: Port {port} is already in use")
            print(f"   Try stopping other servers or use a different port")
        else:
            print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    start_server()
