import os
import sys
import time

# Ensure we can import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from services.sql_agent.src.agent import get_sql_agent

def main():
    agent = get_sql_agent(db_path="data/patients.db")
    
    start = time.time()
    result = agent.query("Give me the details about patient Abdul")
    end = time.time()
    
    print(f"LATENCY: {end - start:.2f} seconds")
    print("RESULT:", result)

if __name__ == "__main__":
    main()
