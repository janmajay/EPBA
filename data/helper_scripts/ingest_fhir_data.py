import json
import sqlite3
import os
import argparse
from datetime import datetime
import glob

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        print(e)
    return conn

def create_tables(conn):
    try:
        cursor = conn.cursor()
        
        # Patients
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            birth_date TEXT,
            death_date TEXT,
            ssn TEXT,
            drivers_license TEXT,
            passport TEXT,
            prefix TEXT,
            first_name TEXT,
            last_name TEXT,
            suffix TEXT,
            maiden_name TEXT,
            marital_status TEXT,
            race TEXT,
            ethnicity TEXT,
            gender TEXT,
            birthplace TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            zip TEXT
        )''')

        # Encounters
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS encounters (
            id TEXT PRIMARY KEY,
            patient_id TEXT,
            start_date TEXT,
            end_date TEXT,
            encounter_class TEXT,
            code TEXT,
            description TEXT,
            cost REAL,
            reason_code TEXT,
            reason_description TEXT,
            provider TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )''')

        # Conditions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conditions (
            id TEXT PRIMARY KEY,
            patient_id TEXT,
            start_date TEXT,
            stop_date TEXT,
            code TEXT,
            description TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )''')
        
        # Medications
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS medications (
            patient_id TEXT,
            start_date TEXT,
            stop_date TEXT,
            code TEXT,
            description TEXT,
            status TEXT,
            reason TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )''')

        # Observations
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            patient_id TEXT,
            date TEXT,
            category TEXT,
            code TEXT,
            description TEXT,
            value TEXT,
            units TEXT,
            type TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )''')

        # Allergies
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS allergies (
            patient_id TEXT,
            start_date TEXT,
            stop_date TEXT,
            code TEXT,
            description TEXT,
            criticality TEXT,
             FOREIGN KEY(patient_id) REFERENCES patients(id)
        )''')

        # CarePlans
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS careplans (
            id TEXT PRIMARY KEY,
            patient_id TEXT,
            start_date TEXT,
            stop_date TEXT,
            code TEXT,
            description TEXT,
            reason_code TEXT,
            reason_description TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )''')
        
        # Immunizations
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS immunizations (
            patient_id TEXT,
            date TEXT,
            code TEXT,
            description TEXT,
             FOREIGN KEY(patient_id) REFERENCES patients(id)
        )''')
        
        # Procedures
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS procedures (
            id TEXT PRIMARY KEY,
            patient_id TEXT,
            date TEXT,
            code TEXT,
            description TEXT,
            reason_code TEXT,
            reason_description TEXT,
             FOREIGN KEY(patient_id) REFERENCES patients(id)
        )''')


        conn.commit()
    except Exception as e:
        print(f"Error creating tables: {e}")

def get_identifier(identifiers, system_key):
    for entry in identifiers:
        if system_key in entry.get('system', ''):
            return entry.get('value')
    return None

def get_extension(extensions, url_key, value_key='valueString'):
    if not extensions: return None
    for entry in extensions:
        if url_key in entry.get('url', ''):
             # Deep check for valueCoding if needed but typically straightforward
             if 'valueCoding' in entry:
                 return entry['valueCoding'].get('display')
             return entry.get(value_key)
    return None

def ingest_file(conn, filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    entries = data.get('entry', [])
    if not entries:
        return
        
    patient_id = None
    
    # 1. First pass - find Patient
    for entry in entries:
        res = entry.get('resource', {})
        if res.get('resourceType') == 'Patient':
            patient_id = res.get('id')
            
            # Extract basic fields
            birth_date = res.get('birthDate')
            death_date = res.get('deceasedDateTime')
            
            identifiers = res.get('identifier', [])
            ssn = get_identifier(identifiers, 'ssn')
            drivers = get_identifier(identifiers, 'driver')
            passport = get_identifier(identifiers, 'passport')
            
            name_entry = res.get('name', [{}])[0]
            prefix = " ".join(name_entry.get('prefix', []))
            first = " ".join(name_entry.get('given', []))
            last = name_entry.get('family')
            suffix = " ".join(name_entry.get('suffix', []))
            maiden = get_extension(res.get('extension'), 'mothersMaidenName')
            
            marital = res.get('maritalStatus', {}).get('text')
            
            race = get_extension(res.get('extension'), 'race', 'valueString') # Or inspect complex extension
            ethnicity = get_extension(res.get('extension'), 'ethnicity', 'valueString')
            
            gender = res.get('gender')
            birthplace = get_extension(res.get('extension'), 'birthPlace', 'valueAddress') # returns dict handle carefully?
            # Actually birthPlace is usually Address, so text repr or specific fields
            if isinstance(birthplace, dict):
                 birthplace_str = f"{birthplace.get('city')}, {birthplace.get('state')}"
            else:
                 birthplace_str = str(birthplace)

            addr_entry = res.get('address', [{}])[0]
            address = " ".join(addr_entry.get('line', []))
            city = addr_entry.get('city')
            state = addr_entry.get('state')
            zip_code = addr_entry.get('postalCode')
            
            sql = '''INSERT OR REPLACE INTO patients (id, birth_date, death_date, ssn, drivers_license, passport, prefix, first_name, last_name, suffix, maiden_name, marital_status, race, ethnicity, gender, birthplace, address, city, state, zip)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            cur = conn.cursor()
            cur.execute(sql, (patient_id, birth_date, death_date, ssn, drivers, passport, prefix, first, last, suffix, maiden, marital, race, ethnicity, gender, birthplace_str, address, city, state, zip_code))
            break # Only one patient per bundle usually

    if not patient_id:
        print(f"No Patient resource found in {filepath}")
        return

    # 2. Second pass - other resources
    cur = conn.cursor()
    for entry in entries:
        res = entry.get('resource', {})
        rtype = res.get('resourceType')
        rid = res.get('id', str(datetime.now().timestamp())) # Fallback ID

        if rtype == 'Encounter':
            start = res.get('period', {}).get('start')
            end = res.get('period', {}).get('end')
            e_class = res.get('class', {}).get('code')
            
            type_coding = res.get('type', [{}])[0].get('coding', [{}])[0]
            code = type_coding.get('code')
            desc = type_coding.get('display')
            
            # Simple cost extraction could be from claim, but let's leave cost 0 for now as it's complex
            cost = 0 
            
            reason_entry = res.get('reason', [{}])[0].get('coding', [{}])[0]
            r_code = reason_entry.get('code')
            r_desc = reason_entry.get('display')
            
            provider = res.get('serviceProvider', {}).get('display')
            
            cur.execute("INSERT OR REPLACE INTO encounters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                        (rid, patient_id, start, end, e_class, code, desc, cost, r_code, r_desc, provider))
            
        elif rtype == 'Condition':
            start = res.get('onsetDateTime')
            stop = res.get('abatementDateTime')
            
            coding = res.get('code', {}).get('coding', [{}])[0]
            code = coding.get('code')
            desc = coding.get('display')
            
            cur.execute("INSERT OR REPLACE INTO conditions (id, patient_id, start_date, stop_date, code, description) VALUES (?, ?, ?, ?, ?, ?)", 
                       (rid, patient_id, start, stop, code, desc))

        elif rtype == 'MedicationRequest':
            # Note: mapping request to usage is loose, treat as medication history
            start = res.get('authoredOn')
            # Stop date might be in dosageInstruction or status
            stop = None # Logic can be complex
            
            coding = res.get('medicationCodeableConcept', {}).get('coding', [{}])[0]
            code = coding.get('code')
            desc = coding.get('display')
            
            status = res.get('status')
            
            cur.execute("INSERT INTO medications (patient_id, start_date, stop_date, code, description, status, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (patient_id, start, stop, code, desc, status, ''))

        elif rtype == 'Observation':
            date = res.get('effectiveDateTime')
            cat_coding = res.get('category', [{}])[0].get('coding', [{}])[0]
            category = cat_coding.get('display', cat_coding.get('code'))
            
            code_coding = res.get('code', {}).get('coding', [{}])[0]
            code = code_coding.get('code')
            desc = code_coding.get('display')
            
            val = res.get('valueQuantity', {}).get('value')
            units = res.get('valueQuantity', {}).get('unit')
            type_ = "Quantity"
            
            if val is None:
                val = res.get('valueString')
                type_ = "String"
                
            cur.execute("INSERT INTO observations (patient_id, date, category, code, description, value, units, type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (patient_id, date, category, code, desc, str(val), units, type_))

        elif rtype == 'AllergyIntolerance':
             start = res.get('assertedDate')
             coding = res.get('code', {}).get('coding', [{}])[0]
             code = coding.get('code')
             desc = coding.get('display')
             crit = res.get('criticality')
             
             cur.execute("INSERT INTO allergies (patient_id, start_date, stop_date, code, description, criticality) VALUES (?, ?, ?, ?, ?, ?)",
                         (patient_id, start, None, code, desc, crit))
                         
        elif rtype == 'Immunization':
             date = res.get('date')
             coding = res.get('vaccineCode', {}).get('coding', [{}])[0]
             code = coding.get('code')
             desc = coding.get('display')
             
             cur.execute("INSERT INTO immunizations (patient_id, date, code, description) VALUES (?, ?, ?, ?)",
                         (patient_id, date, code, desc))

        elif rtype == 'Procedure':
             # performedDateTime or performedPeriod
             date = res.get('performedDateTime')
             if not date:
                 date = res.get('performedPeriod', {}).get('start')
                 
             coding = res.get('code', {}).get('coding', [{}])[0]
             code = coding.get('code')
             desc = coding.get('display')
             
             cur.execute("INSERT OR REPLACE INTO procedures (id, patient_id, date, code, description, reason_code, reason_description) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (rid, patient_id, date, code, desc, None, None))
                         
        elif rtype == 'CarePlan':
             start = res.get('period', {}).get('start')
             stop = res.get('period', {}).get('end')
             
             cat_coding = res.get('category', [{}])[0].get('coding', [{}])[0]
             code = cat_coding.get('code')
             desc = cat_coding.get('display')
             
             cur.execute("INSERT OR REPLACE INTO careplans (id, patient_id, start_date, stop_date, code, description, reason_code, reason_description) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (rid, patient_id, start, stop, code, desc, None, None))

    conn.commit()


def main():
    parser = argparse.ArgumentParser(description='Ingest FHIR JSON to SQLite')
    parser.add_argument('--db', default='patients.db', help='Database file path')
    parser.add_argument('--data_dir', required=True, help='Directory containing JSON files')
    args = parser.parse_args()
    
    conn = create_connection(args.db)
    if conn is not None:
        create_tables(conn)
        
        files = glob.glob(os.path.join(args.data_dir, "*.json"))
        print(f"Found {len(files)} JSON files in {args.data_dir}")
        
        for filepath in files:
            # Skip non-patient files if any (e.g. metadata)
            if 'hospitalInformation' in filepath or 'practitionerInformation' in filepath:
                continue
                
            try:
                ingest_file(conn, filepath)
                # print(f"Processed {filepath}") # Verbose
            except Exception as e:
                print(f"Failed to process {filepath}: {e}")
                
        print("Ingestion complete.")
        conn.close()
    else:
        print("Error! cannot create the database connection.")

if __name__ == '__main__':
    main()
