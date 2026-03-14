import sqlite3
import argparse
import sys
import os

def check_patient(name_query, db_path='patients.db'):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error connecting to database at {db_path}: {e}")
        return

    print(f"Searching for patient matching: '{name_query}'...")
    
    query = """
    SELECT id, first_name, last_name, birth_date, gender, address, city, state 
    FROM patients 
    WHERE first_name LIKE ? OR last_name LIKE ?
    """
    search_term = f"%{name_query}%"
    
    try:
        cursor.execute(query, (search_term, search_term))
        results = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Database error (tables might be missing): {e}")
        conn.close()
        return
    
    if not results:
        print("No patients found.")
        conn.close()
        return

    print(f"Found {len(results)} patient(s):")
    for idx, row in enumerate(results):
        print(f"{idx+1}. {row[1]} {row[2]} (ID: {row[0]}, DOB: {row[3]})")
    
    if len(results) > 1:
        print("\nMultiple matches found. Showing details for the first one.")
    
    # Show details for the first match
    pid = results[0][0]
    print(f"\n--- Clinical Data for {results[0][1]} {results[0][2]} ---")
    
    # Conditions
    print("\n[Conditions]")
    cursor.execute("SELECT start_date, description FROM conditions WHERE patient_id=? ORDER BY start_date DESC LIMIT 10", (pid,))
    conditions = cursor.fetchall()
    if conditions:
        for c in conditions:
            print(f"- {c[1]} (Onset: {c[0]})")
    else:
        print("No conditions recorded.")
        
    # Medications
    print("\n[Medications]")
    cursor.execute("SELECT start_date, description, status FROM medications WHERE patient_id=? ORDER BY start_date DESC LIMIT 10", (pid,))
    meds = cursor.fetchall()
    if meds:
        for m in meds:
            print(f"- {m[1]} (Status: {m[2]}, Date: {m[0]})")
    else:
        print("No medications found.")

    # Encounters (Optional)
    print("\n[Recent Encounters]")
    cursor.execute("SELECT start_date, description, provider FROM encounters WHERE patient_id=? ORDER BY start_date DESC LIMIT 5", (pid,))
    encs = cursor.fetchall()
    if encs:
        for e in encs:
             print(f"- {e[1]} with {e[2]} on {e[0]}")
    else:
         print("No encounters found.")
        
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check patient details in SQLite DB.")
    parser.add_argument("name", help="Name to search for (first or last)")
    parser.add_argument("--db", default="patients.db", help="Path to patients.db")
    args = parser.parse_args()
    
    check_patient(args.name, args.db)
