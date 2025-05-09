import pandas as pd
import re

# Load the raw data
raw_data_file = 'calfire_updates_raw_data.csv'
data = pd.read_csv(raw_data_file)

# Initialize a DataFrame to store cleaned data
cleaned_data = pd.DataFrame(columns=['Update Timestamp', 'Acres Burned (Size)', 'Containment Percent', 'Total Personnel'])

# Function to extract data from raw content
def extract_data(content):
    # Extract Update Timestamp
    timestamp_match = re.search(r'Date:\s*(\d{2}/\d{2}/\d{4})\s*Time:\s*(\d{1,2}:\d{2}\s*[APM]{2})', content)
    timestamp = f"{timestamp_match.group(1)} {timestamp_match.group(2)}" if timestamp_match else None

    # Extract Acres Burned (Size)
    size_match = re.search(r'Size\s*(\d+(?:,\d+)*)(?:\s*acres)?', content, re.IGNORECASE)
    size = size_match.group(1).replace(',', '') if size_match else None

    # Extract Containment Percent
    containment_match = re.search(r'Containment\s*(\d+)%', content, re.IGNORECASE)
    containment = containment_match.group(1) if containment_match else None

    # Extract Total Personnel (if available)
    personnel_match = re.search(r'Total Personnel\s*(\d+(?:,\d+)*)', content, re.IGNORECASE)
    personnel = personnel_match.group(1).replace(',', '') if personnel_match else None

    return timestamp, size, containment, personnel

# Use pd.concat instead of append
rows = []  # Temporary list to store rows

# Process each row in the raw data
for index, row in data.iterrows():
    content = row['Page Content']
    timestamp, size, containment, personnel = extract_data(content)

    # Append the cleaned data to the list
    rows.append({
        'Update Timestamp': timestamp,
        'Acres Burned (Size)': size,
        'Containment Percent': containment,
        'Total Personnel': personnel
    })

# Convert the list of rows to a DataFrame
cleaned_data = pd.DataFrame(rows)

# Save the cleaned data to a new CSV file
cleaned_data_file = 'CleanData.csv'
cleaned_data.to_csv(cleaned_data_file, index=False)
print(f"Cleaned data has been saved to {cleaned_data_file}")