#!/usr/bin/env python3
"""
Generate detailed medical PDF reports for all patients in data/fhir_stu3/
using OpenAI (gpt-4o-mini) and fpdf2.

Usage:
    python data/generate_reports.py                     # from project root
    python data/generate_reports.py --limit 5           # test with 5 patients
    python data/generate_reports.py --patient Abdul     # single patient
"""

import json
import os
import sys
import re
import glob
import time
import argparse
import collections
from datetime import datetime

# Add project root for shared config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from shared.src.config import settings, OPENAI_API_KEY
from openai import OpenAI
from fpdf import FPDF

# ── Constants ──────────────────────────────────────────────────────────────────
FHIR_DIR = os.path.join(PROJECT_ROOT, "data", "fhir_stu3")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "patient_reports")
LLM_MODEL = settings.LLM_MODEL_NAME  # gpt-4o-mini from settings.yaml

client = OpenAI(api_key=OPENAI_API_KEY)


# ── FHIR Data Extraction ──────────────────────────────────────────────────────

def get_display(codable_concept):
    """Extract display text from a FHIR CodeableConcept."""
    if not codable_concept:
        return "Unknown"
    if 'text' in codable_concept:
        return codable_concept['text']
    if 'coding' in codable_concept and len(codable_concept['coding']) > 0:
        return codable_concept['coding'][0].get('display', 'Unknown')
    return "Unknown"


def extract_patient_data(filepath):
    """Parse a FHIR STU3 Bundle JSON and extract clinical summary."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = data.get('entry', [])
    if not entries:
        return None

    patient_info = {}
    resources = collections.defaultdict(list)

    for entry in entries:
        res = entry.get('resource', {})
        rtype = res.get('resourceType')

        if rtype == 'Patient':
            name_entry = res.get('name', [{}])[0]
            first = " ".join(name_entry.get('given', []))
            last = name_entry.get('family', '')
            prefix = " ".join(name_entry.get('prefix', []))

            birth_date = res.get('birthDate')
            death_date = res.get('deceasedDateTime')
            gender = res.get('gender', 'unknown')

            # Calculate approximate age
            age = None
            if birth_date:
                birth_year = int(birth_date[:4])
                end_year = int(death_date[:4]) if death_date else datetime.now().year
                age = end_year - birth_year

            patient_info = {
                'first_name': first,
                'last_name': last,
                'prefix': prefix,
                'gender': gender,
                'birth_date': birth_date,
                'death_date': death_date,
                'age': age,
            }

        elif rtype == 'Condition':
            name = get_display(res.get('code'))
            onset = res.get('onsetDateTime', 'Unknown Date')
            status = res.get('clinicalStatus', 'unknown')
            resources['conditions'].append({
                'name': name, 'onset': onset, 'status': status
            })

        elif rtype == 'MedicationRequest':
            name = get_display(res.get('medicationCodeableConcept'))
            date = res.get('authoredOn', 'Unknown Date')
            status = res.get('status', 'unknown')
            resources['medications'].append({
                'name': name, 'date': date, 'status': status
            })

        elif rtype == 'Encounter':
            desc = get_display(res.get('type', [{}])[0]) if res.get('type') else 'Unknown'
            start = res.get('period', {}).get('start', 'Unknown Date')
            provider = res.get('serviceProvider', {}).get('display', 'Unknown')
            resources['encounters'].append({
                'description': desc, 'date': start, 'provider': provider
            })

        elif rtype == 'Procedure':
            name = get_display(res.get('code'))
            date = res.get('performedDateTime') or res.get('performedPeriod', {}).get('start', 'Unknown Date')
            resources['procedures'].append({'name': name, 'date': date})

        elif rtype == 'Observation':
            cat = "Uncategorized"
            if 'category' in res and len(res['category']) > 0:
                cat = get_display(res['category'][0])
            code_display = get_display(res.get('code'))
            val = res.get('valueQuantity', {}).get('value')
            units = res.get('valueQuantity', {}).get('unit', '')
            if val is None:
                val = res.get('valueString', '')
            date = res.get('effectiveDateTime', 'Unknown Date')
            resources['observations'].append({
                'category': cat, 'name': code_display,
                'value': val, 'units': units, 'date': date
            })

        elif rtype == 'AllergyIntolerance':
            name = get_display(res.get('code'))
            resources['allergies'].append(name)

        elif rtype == 'Immunization':
            name = get_display(res.get('vaccineCode'))
            date = res.get('date', 'Unknown Date')
            resources['immunizations'].append({'name': name, 'date': date})

        elif rtype == 'CarePlan':
            name = "CarePlan"
            if 'category' in res and len(res['category']) > 0:
                name = get_display(res['category'][0])
            resources['careplans'].append(name)

    if not patient_info:
        return None

    patient_info['resources'] = resources
    return patient_info


def build_clinical_summary(patient):
    """Build a text summary of patient data for the LLM prompt."""
    r = patient['resources']
    lines = []

    lines.append(f"Patient: {patient.get('prefix', '')} {patient['first_name']} {patient['last_name']}".strip())
    lines.append(f"Gender: {patient['gender']}")
    lines.append(f"Age: {patient['age']} years" if patient['age'] else "Age: Unknown")
    lines.append("")

    # Conditions (sorted by onset, most recent first, limit 15)
    conditions = sorted(r.get('conditions', []), key=lambda x: x['onset'], reverse=True)[:15]
    if conditions:
        lines.append("[Conditions]")
        for c in conditions:
            lines.append(f"- {c['name']} (Onset: {c['onset']}, Status: {c['status']})")
        lines.append("")

    # Medications (most recent 10)
    meds = sorted(r.get('medications', []), key=lambda x: x['date'], reverse=True)[:10]
    if meds:
        lines.append("[Medications]")
        for m in meds:
            lines.append(f"- {m['name']} (Status: {m['status']}, Date: {m['date']})")
        lines.append("")

    # Allergies
    allergies = list(set(r.get('allergies', [])))
    if allergies:
        lines.append("[Allergies]")
        for a in allergies:
            lines.append(f"- {a}")
        lines.append("")

    # Recent encounters (last 5)
    encounters = sorted(r.get('encounters', []), key=lambda x: x['date'], reverse=True)[:5]
    if encounters:
        lines.append("[Recent Encounters]")
        for e in encounters:
            lines.append(f"- {e['description']} with {e['provider']} on {e['date']}")
        lines.append("")

    # Recent observations (last 10 vital signs)
    vitals = [o for o in r.get('observations', []) if o['category'] == 'Vital Signs']
    vitals = sorted(vitals, key=lambda x: x['date'], reverse=True)[:10]
    if vitals:
        lines.append("[Recent Vital Signs]")
        for v in vitals:
            lines.append(f"- {v['name']}: {v['value']} {v['units']} (Date: {v['date']})")
        lines.append("")

    # Procedures (last 5)
    procedures = sorted(r.get('procedures', []), key=lambda x: x['date'], reverse=True)[:5]
    if procedures:
        lines.append("[Recent Procedures]")
        for p in procedures:
            lines.append(f"- {p['name']} (Date: {p['date']})")
        lines.append("")

    return "\n".join(lines)


def pick_primary_condition(patient):
    """Select the most clinically significant condition for the report focus."""
    conditions = patient['resources'].get('conditions', [])
    if not conditions:
        return "General Health Evaluation"

    # Filter out social/employment findings
    skip_keywords = [
        'employment', 'labor force', 'education', 'housing', 'social isolation',
        'stress', 'misuses drugs', 'risk activity', 'part-time', 'full-time',
        'received higher', 'only received primary', 'has a criminal record',
        'reports of violence', 'limited social contact'
    ]

    clinical_conditions = []
    for c in conditions:
        name_lower = c['name'].lower()
        if not any(kw in name_lower for kw in skip_keywords):
            clinical_conditions.append(c)

    if not clinical_conditions:
        return "General Health Evaluation"

    # Sort by onset (most recent first), prefer active conditions
    active = [c for c in clinical_conditions if c['status'] == 'active']
    if active:
        return active[0]['name']

    # Fall back to most recent non-social condition
    sorted_conds = sorted(clinical_conditions, key=lambda x: x['onset'], reverse=True)
    return sorted_conds[0]['name']


# ── LLM Report Generation ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a medical report writer. Generate a detailed, professional medical evaluation report in plain text format.

RULES:
1. The report MUST be detailed with final conclusion and parameters against each test conducted.
2. Use tabular format for test results (use | column separators for tables).
3. Include: Patient demographics, Clinical Presentation, Physical Examination Findings, relevant diagnostic tests with parameters/results/reference ranges, Final Conclusion & Impression.
4. End with: "Disclaimer: This report is a simulated medical document created for demonstration and documentation purposes only and should not be used for real clinical decision-making."
5. Choose the appropriate medical department and tests based on the primary condition.
6. Generate realistic but simulated test values with proper medical reference ranges.
7. Do NOT use markdown formatting like ** or #. Use plain text with clear section headers.
8. For tables, use this exact format:
   | Column1 | Column2 | Column3 |
   | Value1 | Value2 | Value3 |
"""


def generate_report_text(patient, primary_condition, clinical_summary):
    """Call OpenAI to generate the medical report text."""
    user_prompt = f"""Create a comprehensive medical evaluation report for {patient['first_name']} {patient['last_name']}.

Primary condition/reason for visit: {primary_condition}

The report should focus on evaluating the primary condition with appropriate diagnostic tests, examination findings, and lab results presented in tabular format with parameters, observed values, reference ranges, and interpretations.

IMPORTANT: The report needs to be detailed with final conclusion and parameters against each test that was conducted.

Note: The medical history below is for context/inference only. Generate appropriate test results and findings for the primary condition evaluation.

{clinical_summary}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=3000,
    )

    return response.choices[0].message.content


# ── PDF Generation ─────────────────────────────────────────────────────────────

class MedicalReportPDF(FPDF):
    """Custom PDF class for medical reports."""

    def header(self):
        pass  # No repeating header

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_report_content(self, text, patient_name):
        """Parse the LLM text output and render it into the PDF."""
        self.add_page()
        self.set_auto_page_break(auto=True, margin=20)

        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                self.ln(3)
                i += 1
                continue

            # Detect table blocks (lines containing |)
            if '|' in line and line.count('|') >= 3:
                table_lines = []
                while i < len(lines) and '|' in lines[i] and lines[i].strip().count('|') >= 3:
                    row = lines[i].strip()
                    # Skip separator rows like |---|---|
                    if not re.match(r'^[\s|:-]+$', row):
                        cells = [c.strip() for c in row.split('|') if c.strip()]
                        if cells:
                            table_lines.append(cells)
                    i += 1
                self._render_table(table_lines)
                continue

            # Detect section headers (ALL CAPS or title-like lines)
            if self._is_header(line):
                self.ln(4)
                self.set_font('Helvetica', 'B', 11)
                self.set_text_color(30, 60, 120)
                # Remove any leading/trailing special chars
                clean = line.strip('=-').strip()
                self.cell(0, 8, clean, new_x="LMARGIN", new_y="NEXT")
                self.set_draw_color(30, 60, 120)
                self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
                self.ln(2)
            elif line.startswith('Disclaimer:') or line.startswith('Disclaimer'):
                self.ln(4)
                self.set_draw_color(180, 180, 180)
                self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
                self.ln(3)
                self.set_font('Helvetica', 'I', 7)
                self.set_text_color(120, 120, 120)
                self.safe_multi_cell(0, 4, line)
            else:
                # Regular body text
                self.set_font('Helvetica', '', 9)
                self.set_text_color(40, 40, 40)
                # Handle bold-prefix lines like "Patient Name: Alonso Barton"
                if ':' in line and len(line.split(':')[0]) < 40:
                    parts = line.split(':', 1)
                    self.set_font('Helvetica', 'B', 9)
                    self.write(5, f"{parts[0]}: ")
                    self.set_font('Helvetica', '', 9)
                    self.write(5, parts[1].strip() + "\n")
                else:
                    self.safe_multi_cell(0, 5, line)

            i += 1

    def safe_multi_cell(self, w, h, txt, **kwargs):
        """Safely render text, catching FPDF2 strict spacing errors by pre-wrapping."""
        import textwrap
        
        # Estimate max characters per line at current font size (approximate)
        # width = self.w - self.l_margin - self.r_margin if w == 0 else w
        # At 9pt Helvetica, ~90 chars fit on A4 portrait
        max_chars = 85 if self.font_size_pt >= 9 else 110

        # Pre-wrap text to ensure no single word or line exceeds the page width
        try:
            wrapped_lines = textwrap.wrap(txt, width=max_chars, break_long_words=True)
            safe_txt = "\n".join(wrapped_lines) if wrapped_lines else txt
        except Exception:
            safe_txt = txt

        try:
            self.multi_cell(w, h, safe_txt, **kwargs)
        except Exception:
            # Absolute fallback if even that fails
            try:
                self.multi_cell(w, h, "[Text rendering error - content truncated]", **kwargs)
            except Exception:
                pass

    def _is_header(self, line):
        """Detect if a line is a section header."""
        clean = line.strip('=-').strip()
        if not clean:
            return False
        # All uppercase with at least 3 chars
        if clean.isupper() and len(clean) > 3:
            return True
        # Title case keywords
        header_keywords = [
            'Clinical Presentation', 'Physical Examination', 'Vital Parameters',
            'Laboratory', 'Diagnostic', 'Final Conclusion', 'Impression',
            'Electrocardiogram', 'ECG', 'MRI', 'X-Ray', 'CT Scan',
            'Blood Investigation', 'Examination Findings', 'Test Results',
            'Echocardiography', 'Stress Test', 'Pulmonary Function',
            'Comprehensive', 'Evaluation Report', 'Reason for Visit',
            'Treadmill', 'Cardiac Enzyme', 'Urinalysis', 'Complete Blood',
            'Imaging', 'Radiology', 'Hematology', 'Biochemistry',
        ]
        for kw in header_keywords:
            if kw.lower() in clean.lower() and len(clean) < 80:
                return True
        return False

    def _render_table(self, table_data):
        """Render a table from parsed rows, with adaptive sizing."""
        if not table_data:
            return

        num_cols = max(len(row) for row in table_data)
        if num_cols == 0:
            return
            
        usable_width = self.w - self.l_margin - self.r_margin

        # Adaptive font size based on column count
        if num_cols <= 3:
            font_size = 8
        elif num_cols <= 4:
            font_size = 7
        elif num_cols <= 5:
            font_size = 6
        else:
            font_size = 5

        col_width = usable_width / num_cols

        # If columns are too narrow even with small font, fall back to text
        if col_width < 15:
            self.set_font('Helvetica', '', 8)
            self.set_text_color(40, 40, 40)
            for row in table_data:
                # Use multi_cell for the entire row string
                text = " | ".join(self._sanitize(c) for c in row)
                # Fallback for multi_cell itself failing horizontally
                try:
                    self.multi_cell(0, 5, text)
                except Exception:
                    pass
            self.ln(3)
            return

        # Attempt table rendering, fallback to text if FPDF2 complains about horizontal space
        try:
            self._render_table_internal(table_data, num_cols, col_width, font_size)
        except Exception as e:
            self.set_font('Helvetica', '', 8)
            self.set_text_color(40, 40, 40)
            self.ln(2)
            self.multi_cell(0, 5, "[Table rendered as text due to formatting constraints]")
            for row in table_data:
                try:
                    self.multi_cell(0, 5, " | ".join(self._sanitize(c) for c in row))
                except Exception:
                    pass
            self.ln(3)

    def _render_table_internal(self, table_data, num_cols, col_width, font_size):
        """Internal method to draw the table grid."""
        # Multi_cell in fpdf2 handles text wrapping. We use it to calculate row height.
        row_min_height = 6
        self.set_font('Helvetica', '', font_size)
        
        for row_idx, row in enumerate(table_data):
            while len(row) < num_cols:
                row.append('')

            # Calculate max row height needed for wrapping
            max_lines = 1
            for cell in row:
                text = self._sanitize(cell)
                # Approximate lines by character count per line
                chars_per_line = max(1, int(col_width / (font_size * 0.2)))
                lines = max(1, len(text) // chars_per_line + (1 if len(text) % chars_per_line > 0 else 0))
                max_lines = max(max_lines, lines)
            
            row_height = max_lines * 4 + 2
            
            # Check if we need a new page
            if self.get_y() + row_height > self.h - 20:
                self.add_page()

            is_header = (row_idx == 0)
            if is_header:
                self.set_font('Helvetica', 'B', font_size)
                self.set_fill_color(230, 235, 245)
            else:
                self.set_font('Helvetica', '', font_size)
                if row_idx % 2 == 0:
                    self.set_fill_color(255, 255, 255)
                else:
                    self.set_fill_color(245, 247, 250)

            x_start = self.get_x()
            y_start = self.get_y()
            
            for cell in row:
                text = self._sanitize(cell)
                # Instead of cell(), use multi_cell to allow wrapping within the column width
                # Save current position
                x_curr = self.get_x()
                y_curr = self.get_y()
                
                # Draw the cell background and border with empty cell()
                self.cell(col_width, row_height, "", border=1, fill=True, align='L')
                
                # Move back to print text inside
                self.set_xy(x_curr, y_curr)
                # multi_cell moves Y to next line, so we need to constrain it
                self.multi_cell(col_width, 4, text, align='C' if is_header else 'L')
                
                # Move to the start of the next column
                self.set_xy(x_curr + col_width, y_start)
                
            self.set_xy(x_start, y_start + row_height)

        self.ln(3)

    def _sanitize(self, text):
        """Remove characters that fpdf can't handle."""
        if not isinstance(text, str):
            text = str(text)
        # Replace common problematic chars
        replacements = {
            '\u2264': '<=', '\u2265': '>=', '\u00b2': '2',
            '\u25a0': '', '\u2013': '-', '\u2014': '-',
            '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
            '\u2022': '-', '\u00b0': 'deg', '\u00b5': 'u',
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        # Encode to latin-1 and drop anything that fails
        return text.encode('latin-1', errors='replace').decode('latin-1')


def create_pdf(report_text, output_path, patient_name):
    """Generate a PDF from the report text."""
    pdf = MedicalReportPDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(15, 15, 15)
    pdf.add_report_content(report_text, patient_name)
    pdf.output(output_path)


# ── Batch Orchestration ────────────────────────────────────────────────────────

def sanitize_filename(name):
    """Create a safe filename from a string."""
    # Keep only alphanumeric, spaces, hyphens
    clean = re.sub(r'[^\w\s-]', '', name)
    # Convert spaces to underscores, collapse multiples
    clean = re.sub(r'[\s]+', '_', clean.strip())
    return clean[:50]  # Limit length


def process_patient(filepath, output_dir):
    """Process a single patient file: extract → LLM → PDF."""
    patient = extract_patient_data(filepath)
    if not patient or not patient.get('first_name'):
        return None, "No patient data found"

    first = patient['first_name']
    last = patient['last_name']
    patient_name = f"{first} {last}"

    # Pick primary condition for report focus
    primary_condition = pick_primary_condition(patient)
    condition_short = sanitize_filename(primary_condition.split('(')[0].strip())

    # Build output filename
    output_filename = f"{sanitize_filename(first)}_{sanitize_filename(last)}_Detailed_{condition_short}_Report.pdf"
    output_path = os.path.join(output_dir, output_filename)

    # Skip if already exists
    if os.path.exists(output_path):
        return output_path, "skipped (exists)"

    # Build clinical summary and generate report
    clinical_summary = build_clinical_summary(patient)
    report_text = generate_report_text(patient, primary_condition, clinical_summary)

    # Create PDF
    create_pdf(report_text, output_path, patient_name)

    return output_path, "created"


def main():
    parser = argparse.ArgumentParser(description='Generate medical PDF reports for all patients')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of patients to process (0 = all)')
    parser.add_argument('--patient', type=str, default='', help='Filter by patient name (partial match)')
    parser.add_argument('--data-dir', default=FHIR_DIR, help='Directory containing FHIR JSON files')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory for PDFs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Gather JSON files
    json_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    # Skip non-patient files
    json_files = [f for f in json_files
                  if 'hospitalInformation' not in f and 'practitionerInformation' not in f]

    # Optional name filter
    if args.patient:
        json_files = [f for f in json_files if args.patient.lower() in os.path.basename(f).lower()]

    # Optional limit
    if args.limit > 0:
        json_files = json_files[:args.limit]

    total = len(json_files)
    print(f"Processing {total} patient files...")
    print(f"Output directory: {args.output_dir}")
    print(f"LLM model: {LLM_MODEL}")
    print(f"{'='*60}")

    created = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for idx, filepath in enumerate(json_files):
        basename = os.path.basename(filepath)
        elapsed = time.time() - start_time
        rate = (idx / elapsed) if elapsed > 0 and idx > 0 else 0
        eta = ((total - idx) / rate) if rate > 0 else 0

        print(f"[{idx+1}/{total}] {basename[:50]}... ", end="", flush=True)

        try:
            outpath, status = process_patient(filepath, args.output_dir)
            if status == "skipped (exists)":
                skipped += 1
                print(f"⏭ skipped")
            elif outpath:
                created += 1
                print(f"✓ done ({elapsed:.0f}s elapsed, ~{eta/60:.0f}m remaining)")
            else:
                failed += 1
                print(f"⚠ {status}")
        except Exception as e:
            failed += 1
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done in {total_time/60:.1f} minutes")
    print(f"  Created: {created}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")


if __name__ == "__main__":
    main()
