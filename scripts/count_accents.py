import pandas as pd

# Path to your CSV file
csv_path = "processed_data/csv/combined_metadata.csv"

# Read the CSV file
df = pd.read_csv(csv_path)

# Get unique accents
unique_accents = df['accent'].unique()

# Count unique accents
num_accents = len(unique_accents)

# Print results
print(f"Number of unique accents: {num_accents}")
print("List of accents:")
print(unique_accents.tolist())

# Optionally, print the count of each accent
print("\nAccent distribution:")
print(df['accent'].value_counts())

