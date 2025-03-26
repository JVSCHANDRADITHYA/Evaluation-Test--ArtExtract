import os
import pandas as pd
import random
from collections import Counter

# Path to CSV file
csv_file = r"make_dataset\files\class_wiki_art.csv"  # Update with actual path

# Read CSV file
df = pd.read_csv(csv_file, header=None)

# Dictionary to store artist names
artist_name_map = {}

# Group by artist class (column index 1)
artist_groups = df.groupby(1)

for artist_class, group in artist_groups:
    file_names = group[0].tolist()  # List of file names

    common_words = Counter()

    print(f"\n🔹 **Processing Class {artist_class}** 🔹")
    print(f"Total Images: {len(file_names)}\n")

    # Select 5 random pairs from the class
    for _ in range(5):
        if len(file_names) < 2:
            print("❌ Not enough images to compare!\n")
            continue  # Skip if less than 2 images

        img1, img2 = random.sample(file_names, 2)

        # Extract words from both file names
        words1 = set(img1.split("/")[-1].split("_"))
        words2 = set(img2.split("/")[-1].split("_"))

        # Find common words
        common = words1.intersection(words2)

        print(f"Comparing:\n  📌 {img1}\n  📌 {img2}")
        print(f"🔍 Common Words: {common}")

        for word in common:
            common_words[word] += 1  # Count occurrences

    if common_words:
        print("\n📊 **Word Scores:**")
        for word, score in common_words.items():
            print(f"  {word}: {score}")

        # Pick the most common word (keeping hyphens)
        artist_name = max(common_words, key=common_words.get)

        artist_name_map[int(artist_class)] = artist_name  # Convert to int for sorting
        print(f"\n✅ **Final Artist Name for Class {artist_class}:** {artist_name}\n")

# Sort numerically by artist class
sorted_artist_names = sorted(artist_name_map.items())

# Save artist names to a text file
output_file = r"make_dataset\files\artist_classified.txt"  # Update path

with open(output_file, "w") as f:
    for artist_class, name in sorted_artist_names:
        f.write(f"{artist_class}: {name}\n")

print("\n🎯 **Final Results:**")
for artist_class, name in sorted_artist_names:
    print(f"  {artist_class}: {name}")

print("\n✅ Artist names saved in ascending order to:", output_file)
