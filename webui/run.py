#!/usr/bin/env python3
"""
Kronos Web UI startup script
"""

import os
import sys
import subprocess

def check_dependencies() -> bool:
    """Check if dependencies are installed"""
    try:
        import flask
        import pandas
        import numpy
        import plotly
        print("[OK] All dependencies installed")
        return True
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def install_dependencies() -> bool:
    """Install dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Dependencies installation completed")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Dependencies installation failed")
        return False

def main() -> None:
    """Main function"""
    print("Starting Kronos Web UI...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\nAuto-install dependencies? (y/n): ", end="")
        if input().lower() == 'y':
            if not install_dependencies():
                return
        else:
            print("Please manually install dependencies and retry")
            return
    
    # Check model availability
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("[OK] Kronos model library available")
        model_available = True
    except ImportError:
        print("[WARNING] Kronos model library not available, will use simulated prediction")
        model_available = False
    
    # Start Flask application
    print("\nStarting Web server...")
    
    # Start server
    try:
from app import app
        print("[OK] Web server started successfully!")
        print("Access URL: http://127.0.0.1:7070")
        print("Tip: Press Ctrl+C to stop server")
        
        from decision.config import get_config

        web_config = get_config().webui
        app.run(
            debug=False,
            host=web_config.host,
            port=web_config.port,
            use_reloader=False,
        )
        
    except Exception as e:
        print(f"[ERROR] Startup failed: {e}")
        print("Please check if port 7070 is occupied")

if __name__ == "__main__":
    main()
