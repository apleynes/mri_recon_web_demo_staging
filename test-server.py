#!/usr/bin/env python3
"""
Test server for MRI Recon Web Demo with proper CORS headers for WebAssembly threading.
This simulates the conditions needed for multi-threading that GitHub Pages cannot provide.
"""

import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Set the required headers for WebAssembly SharedArrayBuffer
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Resource-Policy', 'same-site')
        
        # Set CSP headers - match the configuration in index.html and Trunk.toml
        csp = "default-src 'self'; script-src 'unsafe-inline' 'unsafe-eval' 'self' blob: data:; style-src 'unsafe-inline' 'self'; worker-src 'self' blob: data:; object-src 'none'; img-src 'self' data: blob:;"
        self.send_header('Content-Security-Policy', csp)
        
        # Cache control for development
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        
        super().end_headers()

    def log_message(self, format, *args):
        # Custom logging to show CORS status
        print(f"[CORS Server] {self.address_string()} - {format % args}")

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    
    # Change to the dist directory
    if os.path.exists('dist'):
        os.chdir('dist')
        print(f"Serving from ./dist directory")
    else:
        print(f"Warning: ./dist directory not found. Serving from current directory.")
    
    with socketserver.TCPServer(("", port), CORSHandler) as httpd:
        print(f"CORS-enabled server running at http://localhost:{port}")
        print(f"This server includes the headers required for WebAssembly multi-threading:")
        print(f"  - Cross-Origin-Embedder-Policy: require-corp")
        print(f"  - Cross-Origin-Opener-Policy: same-origin")
        print(f"  - Cross-Origin-Resource-Policy: same-site")
        print(f"\nPress Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\nServer stopped.")

if __name__ == "__main__":
    main() 