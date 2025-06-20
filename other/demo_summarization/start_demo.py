#!/usr/bin/env python3
"""
Startup script for the Audio Analysis Summarization Demo

This script helps start both the backend and frontend servers.
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'openai', 'pydub', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install dependencies:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_config():
    """Check if configuration files exist"""
    config_file = Path(__file__).parent / "config.json"
    api_config_file = Path(__file__).parent.parent.parent / "api.config"
    
    if not config_file.exists():
        print("❌ config.json not found")
        print("Please create config.json with your OpenAI API key")
        return False
    
    if not api_config_file.exists():
        print("❌ api.config not found in root directory")
        print("Please ensure Behavioral Signals API configuration exists")
        return False
    
    # Check if OpenAI key is configured
    try:
        import json
        with open(config_file) as f:
            config = json.load(f)
        
        if config.get('openai_api_key') == 'your_openai_api_key_here':
            print("⚠️  OpenAI API key not configured in config.json")
            print("Please update config.json with your actual API key")
            return False
    except Exception as e:
        print(f"❌ Error reading config.json: {e}")
        return False
    
    return True

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting backend server...")
    backend_process = subprocess.Popen([
        sys.executable, "backend.py"
    ], cwd=Path(__file__).parent)
    
    return backend_process

def start_frontend():
    """Start the Streamlit frontend"""
    print("🌐 Starting frontend application...")
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "frontend.py", 
        "--server.headless", "true",
        "--server.port", "8501"
    ], cwd=Path(__file__).parent)
    
    return frontend_process

def main():
    """Main function to start the demo application"""
    print("🎵 Audio Analysis Summarization Demo")
    print("=" * 50)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ Dependencies OK")
    
    # Check configuration
    print("⚙️  Checking configuration...")
    if not check_config():
        sys.exit(1)
    print("✅ Configuration OK")
    
    # Start servers
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = start_backend()
        time.sleep(3)  # Give backend time to start
        
        # Check if backend started successfully
        if backend_process.poll() is not None:
            print("❌ Backend failed to start")
            sys.exit(1)
        
        print("✅ Backend started on http://localhost:8000")
        
        # Start frontend
        frontend_process = start_frontend()
        time.sleep(3)  # Give frontend time to start
        
        # Check if frontend started successfully
        if frontend_process.poll() is not None:
            print("❌ Frontend failed to start")
            sys.exit(1)
        
        print("✅ Frontend started on http://localhost:8501")
        print("\n🎉 Demo application is ready!")
        print("📱 Open your browser and go to: http://localhost:8501")
        print("\n⏹️  Press Ctrl+C to stop both servers")
        
        # Wait for user to stop
        try:
            while True:
                time.sleep(1)
                # Check if processes are still running
                if backend_process.poll() is not None:
                    print("❌ Backend process stopped unexpectedly")
                    break
                if frontend_process.poll() is not None:
                    print("❌ Frontend process stopped unexpectedly")
                    break
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
    
    finally:
        # Clean up processes
        if backend_process and backend_process.poll() is None:
            print("🔄 Stopping backend...")
            backend_process.terminate()
            backend_process.wait()
        
        if frontend_process and frontend_process.poll() is None:
            print("🔄 Stopping frontend...")
            frontend_process.terminate()
            frontend_process.wait()
        
        print("✅ All servers stopped")

if __name__ == "__main__":
    main()
