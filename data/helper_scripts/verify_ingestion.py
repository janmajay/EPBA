import sqlite3
import argparse

def verify(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    tables = ['patients', 'encounters', 'conditions', 'medications', 'observations', 'allergies', 'immunizations', 'procedures', 'careplans']
    
    print(f"--- Database Verification: {db_file} ---")
    
    total_patients = 0
    
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"Table '{table}': {count} rows")
            if table == 'patients':
                total_patients = count
        except Exception as e:
            print(f"Table '{table}': Error - {e}")
            
    if total_patients > 0:
        print("\n--- Sample Patient Data ---")
        cursor.execute("SELECT * FROM patients LIMIT 1")
        patient = cursor.fetchone()
        
        # Get column names
        cursor.execute("PRAGMA table_info(patients)")
        columns = [col[1] for col in cursor.fetchall()]
        
        pid_idx = columns.index('id')
        pid = patient[pid_idx]
        
        for col, val in zip(columns, patient):
            print(f"{col}: {val}")
        
        print(f"\n--- Clinical Data for Patient {pid} ---")
        print("Conditions:")
        cursor.execute(f"SELECT start_date, description FROM conditions WHERE patient_id=?", (pid,))
        for row in cursor.fetchall():
            print(f"- {row[1]} (Onset: {row[0]})")
        
        print("\nMedications:")
        cursor.execute(f"SELECT start_date, description, status FROM medications WHERE patient_id=?", (pid,))
        for row in cursor.fetchall():
            print(f"- {row[1]} (Status: {row[2]}, Date: {row[0]})")
    
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='patients.db', help='Database file path')
    args = parser.parse_args()
    verify(args.db)
