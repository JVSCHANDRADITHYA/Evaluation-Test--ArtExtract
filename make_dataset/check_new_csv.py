import os
import pandas as pd

# Paths
csv_file = r"csv_files\class_wiki_art.csv"  # Change this to your CSV file
img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"  # Change this to your image directory
output_csv = r"csv_files\cleaned_dataset.csv"  # Output CSV without missing images

# Load CSV
df = pd.read_csv(csv_file, header=None)  # No headers in CSV
df.columns = ["image_path", "artist", "genre", "style"]  # Assign column names

# Lists to store found & missing images
found_images = []
missing_images = []

# Check if images exist
for _, row in df.iterrows():
    img_path = os.path.join(img_dir, row["image_path"])
    
    if os.path.exists(img_path):
        print(f"✅ Found: {row['image_path']}")
        found_images.append(row)
    else:
        print(f"❌ Not Found: {row['image_path']}")
        missing_images.append(row["image_path"])

# Convert found images back to DataFrame
df_filtered = pd.DataFrame(found_images)

# Save cleaned CSV
df_filtered.to_csv(output_csv, index=False, header=False)

# Get the range of artist, genre, and style labels
artist_min, artist_max = df_filtered["artist"].min(), df_filtered["artist"].max()
genre_min, genre_max = df_filtered["genre"].min(), df_filtered["genre"].max()
style_min, style_max = df_filtered["style"].min(), df_filtered["style"].max()

# Print Summary
print("\n--- Summary ---")
print(f"Total images in CSV: {len(df)}")
print(f"✅ Found images: {len(found_images)}")
print(f"❌ Missing images: {len(missing_images)}")
print(f"\n❌ Missing Image List: {missing_images}")
print(f"\n🔢 Label Ranges:")
print(f"🎨 Artist: {artist_min} to {artist_max}")
print(f"📚 Genre: {genre_min} to {genre_max}")
print(f"🖌️ Style: {style_min} to {style_max}")
print(f"\n📂 Modified CSV saved as: {output_csv}")