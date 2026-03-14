import json
import collections

filepath = '/Users/jay/Downloads/fhir/Abdul218_Harris789_b0a06ead-cc42-aa48-dad6-841d4aa679fa.json'

with open(filepath, 'r') as f:
    data = json.load(f)

resources = collections.defaultdict(list)

def get_display(codable_concept):
    if not codable_concept: return "Unknown"
    if 'text' in codable_concept:
        return codable_concept['text']
    if 'coding' in codable_concept and len(codable_concept['coding']) > 0:
        return codable_concept['coding'][0].get('display', 'Unknown')
    return "Unknown"

for entry in data.get('entry', []):
    res = entries = entry.get('resource', {})
    rtype = res.get('resourceType')
    
    if rtype == 'Condition':
        name = get_display(res.get('code'))
        onset = res.get('onsetDateTime', 'Unknown Date')
        resources['Conditions'].append(f"{name} (Onset: {onset})")
        
    elif rtype == 'AllergyIntolerance':
        name = get_display(res.get('code'))
        resources['Allergies'].append(name)
        
    elif rtype == 'MedicationRequest':
        name = get_display(res.get('medicationCodeableConcept'))
        date = res.get('authoredOn', 'Unknown Date')
        status = res.get('status', 'unknown')
        resources['Medications'].append(f"{name} (Status: {status}, Date: {date})")
        
    elif rtype == 'Immunization':
        name = get_display(res.get('vaccineCode'))
        date = res.get('date', 'Unknown Date')
        resources['Immunizations'].append(f"{name} (Date: {date})")
        
    elif rtype == 'Procedure':
        name = get_display(res.get('code'))
        # Try different date fields
        date = res.get('performedPeriod', {}).get('start', res.get('performedDateTime', 'Unknown Date'))
        resources['Procedures'].append(f"{name} (Date: {date})")
        
    elif rtype == 'Observation':
        cat = "Uncategorized"
        if 'category' in res and len(res['category']) > 0:
             cat = get_display(res['category'][0])
        resources['Observations'].append(cat)

    elif rtype == 'CarePlan':
        name = "CarePlan"
        if 'category' in res and len(res['category']) > 0:
            name = get_display(res['category'][0])
        activity_count = len(res.get('activity', []))
        resources['CarePlans'].append(f"{name} with {activity_count} activities")
        
print(f"--- Medical Data Summary for {filepath.split('/')[-1]} ---\n")

print(f"Allergies ({len(resources['Allergies'])}):")
for item in sorted(set(resources['Allergies'])):
    print(f"- {item}")
print("")

print(f"Conditions/Diagnoses ({len(resources['Conditions'])}):")
for item in resources['Conditions']: # specific instances matter here primarily for timeline, but let's just list unique for concise reading if too many? No, list all for now but maybe truncated if huge.
    print(f"- {item}")
print("")

print(f"Medications ({len(resources['Medications'])}):")
# Deduplicate by name for summary, but finding chronic vs acute is hard. Just list occurrences or uniq. 
# Let's list unique Prescriptions to avoid noise
for item in sorted(set(resources['Medications'])):
    print(f"- {item}")
print("")

print(f"Immunizations ({len(resources['Immunizations'])}):")
for item in sorted(set(resources['Immunizations'])):
    print(f"- {item}")
print("")

print(f"Procedures ({len(resources['Procedures'])}):")
for item in sorted(set(resources['Procedures'])):
    print(f"- {item}")
print("")

print(f"Observations ({len(resources['Observations'])}):")
obs_counts = collections.Counter(resources['Observations'])
for cat, count in obs_counts.items():
    print(f"- {cat}: {count}")

print("")
print(f"CarePlans ({len(resources['CarePlans'])}):")
for item in sorted(set(resources['CarePlans'])):
    print(f"- {item}")
