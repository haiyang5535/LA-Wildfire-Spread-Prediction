df = pd.read_csv("CleanData.csv")

# 1  Parse dates & drop blank rows
df['Update Timestamp'] = pd.to_datetime(df['Update Timestamp'], errors='coerce')
df = df.dropna(subset=['Update Timestamp', 'Acres Burned (Size)', 'Containment Percent'])

# 2  Deduplicate
df = df.sort_values('Update Timestamp').drop_duplicates(subset='Update Timestamp', keep='last')

# 3  Forward‑fill personnel (optional)
df['Total Personnel'] = df['Total Personnel'].fillna(method='ffill')

# 4  Create analytics columns
df['Δhours'] = df['Update Timestamp'].diff().dt.total_seconds() / 3600
df['Δacres'] = df['Acres Burned (Size)'].diff()
df['GrowthRate'] = df['Δacres'] / df['Δhours']
df['Δcontainment'] = df['Containment Percent'].diff()
