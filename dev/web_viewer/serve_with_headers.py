#!/usr/bin/env python3
import http.server
import socket
import socketserver
import os
import time
import sys

PORT = int(os.environ.get('PORT', '8080'))

class Handler(http.server.SimpleHTTPRequestHandler):
	def end_headers(self):
		# CORS and COOP/COEP for WebGPU/WebNN/WASM
		self.send_header('Access-Control-Allow-Origin', '*')
		self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
		self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
		self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
		self.send_header('Pragma', 'no-cache')
		self.send_header('Expires', '0')
		super().end_headers()

def port_is_open(port: int) -> bool:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.settimeout(0.2)
		try:
			return s.connect_ex(('127.0.0.1', port)) == 0
		except Exception:
			return False

def main():
	# Serve from dev/web_viewer root so relative paths resolve
	script_dir = os.path.dirname(os.path.abspath(__file__))
	os.chdir(script_dir)
	# If something already listens on PORT, consider it good for reuse
	if port_is_open(PORT):
		print(f"Port {PORT} already in use; assuming server is running. Reusing existing server (keeping process alive).")
		try:
			while True:
				time.sleep(60)
				# If port becomes free, exit so runner can restart
				if not port_is_open(PORT):
					print(f"Port {PORT} freed; exiting reuse loop.")
					return 0
		except KeyboardInterrupt:
			print("\nExiting reuse loop...")
			return 0
	# Allow quick reuse of address
	socketserver.TCPServer.allow_reuse_address = True
	try:
		# Bind explicitly to IPv4 localhost to avoid IPv6-only localhost environments
		with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:
			print(f"Serving dev/web_viewer on http://localhost:{PORT}")
			try:
				httpd.serve_forever()
			except KeyboardInterrupt:
				print("\nShutting down server...")
			finally:
				httpd.server_close()
	except OSError as e:
		# If the port became busy between check and bind, treat as reusable
		if getattr(e, 'errno', None) == 98:
			print(f"Port {PORT} in use; reusing existing server.")
			return 0
		raise

if __name__ == '__main__':
	sys.exit(main())
