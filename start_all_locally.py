import subprocess
import os
import sys
import time
import signal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Common Environment Variables for all services
env = os.environ.copy()
env["PYTHONPATH"] = os.getcwd()
env["CONFIG_PATH"] = os.path.join(os.getcwd(), "config", "settings.yaml")

# Use venv python if available
python_executable = sys.executable
if os.path.exists(".venv/bin/python"):
    python_executable = ".venv/bin/python"

# Check for OpenAI Key
if "OPENAI_API_KEY" not in env:
    print("WARNING: OPENAI_API_KEY not found in environment variables.")

processes = []

def start_service(command, name, port=None):
    print(f"Starting {name}..." + (f" (Port {port})" if port else ""))
    # Shell=True allows using the string command directly, but explicit list is safer/cleaner.
    # We use list format for subprocess.
    cmd_list = command.split()
    proc = subprocess.Popen(cmd_list, env=env)
    processes.append(proc)
    return proc

def check_dependencies():
    # Simple check to see if shared is installed or pythonpath works
    try:
        from shared.src.config import settings
        print("Shared library found.")
    except ImportError:
        print("Shared library not found in path. Ensure you ran 'pip install -e shared' or PYTHONPATH is set.")

def cleanup_ports():
    ports = [8000, 8001, 8002, 8003, 8004, 8501]
    print(f"Cleaning up ports: {ports}...")
    for port in ports:
        try:
            # Find PID using lsof
            cmd = f"lsof -ti:{port}"
            pid = subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL).decode().strip()
            if pid:
                print(f"Killing process {pid} on port {port}")
                subprocess.run(["kill", "-9", pid], check=True)
        except subprocess.CalledProcessError:
            # No process found on port
            pass
        except Exception as e:
            print(f"Error cleaning port {port}: {e}")

def main():
    cleanup_ports()
    check_dependencies()
    
    try:
        # 1. SQL Agent
        start_service(f"{python_executable} -m uvicorn services.sql_agent.src.app:app --host 0.0.0.0 --port 8001", "SQL Agent", 8001)
        
        # 2. Vector Agent
        start_service(f"{python_executable} -m uvicorn services.vector_agent.src.app:app --host 0.0.0.0 --port 8002", "Vector Agent", 8002)
        
        # 3. Summarization Agent
        start_service(f"{python_executable} -m uvicorn services.summarization_agent.src.app:app --host 0.0.0.0 --port 8003", "Summarizer Agent", 8003)
        
        # Give agents a moment to potentially fail fast if imports are wrong
        time.sleep(2)
        
        # 4. Orchestrator
        # Orchestrator needs to know where other agents are. 
        # In local non-docker run, localhost URLs in settings.yaml (default) usually point to docker service names if we didn't change them.
        # Wait! config/settings.yaml has:
        # sql_agent_url: "http://sql_agent:8001/query"
        # This WON'T work for local execution (outside docker) because "sql_agent" hostname doesn't exist.
        # We need to override these OR config loader needs to be smart.
        # We can override via ENV vars in this script.
        
        env["SQL_AGENT_URL"] = "http://localhost:8001/query"
        env["VECTOR_AGENT_URL"] = "http://localhost:8002/query"
        env["SUMMARIZER_AGENT_URL"] = "http://localhost:8003/summarize"
        env["ORCHESTRATOR_URL"] = "http://localhost:8000/query"
        env["AGENT_REGISTRY_URL"] = "http://localhost:8004"
        
        start_service(f"{python_executable} -m uvicorn services.orchestrator.src.app:app --host 0.0.0.0 --port 8000", "Orchestrator", 8000)
        
        time.sleep(2)
        
        # 5. Agent Registry
        start_service(f"{python_executable} -m uvicorn services.agent_registry.src.app:app --host 0.0.0.0 --port 8004", "Agent Registry", 8004)
        
        time.sleep(2)
        
        # 6. Frontend
        print("Starting Frontend...")
        start_service(f"{python_executable} -m streamlit run services/frontend/src/app.py --server.port 8501", "Frontend", 8501)
        
        print("\nAll services started! Press Ctrl+C to stop.")
        print("Frontend running at: http://localhost:8501")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping all services...")
        for p in processes:
            p.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
