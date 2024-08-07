import pandas as pd

def load_and_preprocess_libritts(file_path):
    df = pd.read_csv(file_path)
    df['dataset'] = 'LibriTTS'
    return df[['file_path', 'speaker_id', 'accent', 'transcription', 'dataset', 'gender']]

def load_and_preprocess_l2arctic(file_path):
    df = pd.read_csv(file_path)
    df['dataset'] = 'L2-Arctic'
    return df[['file_path', 'speaker_id', 'accent', 'transcription', 'dataset', 'gender']]

def combine_datasets(libritts_path, l2arctic_path, output_path):
    libritts_df = load_and_preprocess_libritts(libritts_path)
    l2arctic_df = load_and_preprocess_l2arctic(l2arctic_path)
    
    # Combine dataframes
    combined_df = pd.concat([libritts_df, l2arctic_df], ignore_index=True)
    
    # Ensure consistent column order
    columns = ['file_path', 'speaker_id', 'accent', 'transcription', 'gender', 'dataset']
    combined_df = combined_df[columns]
    
    # Save combined dataset
    combined_df.to_csv(output_path, index=False)
    print(f"Combined dataset saved to {output_path}")
    print(f"Total samples: {len(combined_df)}")
    print(f"Samples per dataset:\n{combined_df['dataset'].value_counts()}")
    print(f"Accents distribution:\n{combined_df['accent'].value_counts()}")

if __name__ == "__main__":
    libritts_path = "data/csv/libritts_metadata.csv"
    l2arctic_path = "data/csv/l2arctic_metadata.csv"
    output_path = "data/csv/combined_metadata.csv"
    
    combine_datasets(libritts_path, l2arctic_path, output_path)