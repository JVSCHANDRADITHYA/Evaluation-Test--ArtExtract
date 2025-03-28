'''
Script to check for missing images in multiple CSV files and optionally update them.
'''
import os
import pandas as pd

# Define paths
dataset_folder = r"F:\GSoc_2025\wiki_art_dataset\wikiart"
csv_files = {
    "artist": r"F:\GSoc_2025\wiki_art_dataset\artist_val.csv",
    "styles": r"F:\GSoc_2025\wiki_art_dataset\style_train.csv",
    "genres": r"F:\GSoc_2025\wiki_art_dataset\genre_train.csv"
}

# Get all existing images in dataset (ignoring subfolders)
existing_images = set()
for root, _, files in os.walk(dataset_folder):
    for file in files:
        existing_images.add(file.lower())  # Convert to lowercase for consistency

print("Total images found in dataset:", len(existing_images))

# Process each CSV file
updated_dfs = {}
for name, csv_path in csv_files.items():
    print(f"\nChecking CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    csv_images = set(df['image_path'].apply(lambda x: os.path.basename(x).lower()))
    missing_images = csv_images - existing_images  # Find missing images

    print(f"Total missing images in {name}: {len(missing_images)}")
    print("Missing images:", missing_images)

    # Filter out missing images
    df_filtered = df[~df['image_path'].apply(lambda x: os.path.basename(x).lower()).isin(missing_images)]
    updated_dfs[name] = df_filtered

# Ask user if they want to save the updated CSVs
save_choice = input("\nDo you want to save the updated CSVs? (yes/no): ").strip().lower()
if save_choice == 'yes':
    updated_folder = os.path.join(os.path.dirname(list(csv_files.values())[0]), "updated_csv")
    os.makedirs(updated_folder, exist_ok=True)

    for name, df_filtered in updated_dfs.items():
        save_path = os.path.join(updated_folder, f"{name}_updated.csv")
        df_filtered.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

print("Process completed.")
