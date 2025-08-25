#!/usr/bin/env python3
"""
Simple HTTP server for the RSMT Viewer to avoid CORS issues
Run this script from the web_viewer directory to serve the application.
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from pathlib import Path

# Default port
PORT = 8000

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

def main():
    # Change to the web_viewer directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"🚀 Starting RSMT Viewer development server...")
    print(f"📁 Serving from: {script_dir}")
    print(f"🌐 Server will run on: http://localhost:{PORT}")
    print(f"📄 Main page: http://localhost:{PORT}/rsmt_showcase_modern.html")
    print(f"")
    print(f"🔧 This server solves CORS issues by serving files over HTTP instead of file://")
    print(f"⚠️  Press Ctrl+C to stop the server")
    print(f"")
    
    try:
        with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
            print(f"✅ Server started successfully!")
            print(f"🌐 Navigate to: http://localhost:{PORT}/rsmt_showcase_modern.html")
            print(f"")
            
            # Try to open browser automatically
            try:
                webbrowser.open(f'http://localhost:{PORT}/rsmt_showcase_modern.html')
                print(f"🌍 Browser opened automatically")
            except Exception as e:
                print(f"⚠️  Could not open browser automatically: {e}")
                print(f"📖 Please manually navigate to: http://localhost:{PORT}/rsmt_showcase_modern.html")
            
            print(f"")
            httpd.serve_forever()
            
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use!")
            print(f"💡 Try: killall python3 or use a different port")
            print(f"🔄 Or try: python3 start_server.py {PORT + 1}")
        else:
            print(f"❌ Failed to start server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
        print(f"✅ RSMT Viewer server shutdown complete")

if __name__ == "__main__":
    # Allow custom port via command line
    if len(sys.argv) > 1:
        try:
            PORT = int(sys.argv[1])
        except ValueError:
            print(f"❌ Invalid port number: {sys.argv[1]}")
            print(f"💡 Usage: python3 start_server.py [port]")
            sys.exit(1)
    
    main()
