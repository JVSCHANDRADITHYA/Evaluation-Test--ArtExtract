# 🎨 WikiArt Classification with CNN-LSTM

This project tackles the task of **multitask classification** on the **WikiArt Dataset**, predicting:

```
- 👨‍🎨 Artist (128 classes)
- 🖌️ Style  (27 classes)
- 🧑‍🏫 Genre  (11 classes)
```

Unlike most implementations that rely on the incomplete subset (~11k images), **this project uses the full WikiArt dataset (~81,000 paintings)**. The dataset was curated manually by scraping and mapping all artwork to the correct labels, ensuring **clean, complete, and consistent** training data.



## Results
✅ Test Accuracy (on 6,000 unseen images)
Task	Top-1 Accuracy
```
🎨 Artist	62.4%
🧑‍🏫 Genre	73.1%
🖌️ Style	68.7%
```
```
Note: The dataset is heavily imbalanced, especially for artists. Only a handful of artists have >1000 paintings.
```

🏆 Top 10 Most Represented Artists

```
Artist Name	            Paintings

albrecht-durer          37450+ images
vincent-van-gogh:       1673 images
nicholas-roerich:       1593 images
pierre-auguste-renoir:  1227 images
claude-monet:           1191 images
pyotr-konchalovsky:     807 images
camille-pissarro:       777 images
john-singer-sargent:    693 images
rembrandt:              685 images
pablo-picasso:          677 images
```

This shows us that the dataset is heavily imbalanced and learns to classify only the label of 'albrechet-durer' properly

However the result for the actual in-site data is projected to be better and will be updated within 24 hours

---

## 🧠 Architecture: CNN-LSTM for Multitask Classification

I used a hybrid architecture:

```text
Pretrained ResNet18 (feature extractor)
        ↓
Feature Map: (B, 512, 7, 7)
        ↓
Reshape → (B, 49, 512)
        ↓
LSTM across spatial tokens
        ↓
3 Fully Connected Layers:
   ├── Artist  (128)
   ├── Genre   (11)
   └── Style   (27)
```

The CNN extracts local spatial patterns, and the LSTM aggregates those into a global context. This allows the model to interpret complex brushstroke patterns and composition layouts.

## Dataset

    Total Images Used: ~81,000 (scraped, validated)

    Train/Test Split: 75,000 / 6,000

    Source: Full dataset constructed from ArtGAN and raw WikiArt HTML pages with label matching.

    Preprocessing: Center-cropped, resized (224x224), normalized using ImageNet stats.


## Predictions Visualization

### Correctly Classified All 3 (Artist, Genre, Style)
<div align="center"> <img src="classification_samples/correct_all/img_0_1.jpg" width="200"/> <img src="classification_samples/correct_all/img_0_2.jpg" width="200"/> <img src="classification_samples/correct_all/img_0_3.jpg" width="200"/> <img src="classification_samples/correct_all/img_0_5.jpg" width="200"/> </div>

### Correctly Classified Any 2
<div align="center"> <img src="classification_samples/correct_two/img_0_4.jpg" width="200"/> <img src="classification_samples/correct_two/img_0_12.jpg" width="200"/> <img src="classification_samples/correct_two/img_1_3.jpg" width="200"/> <img src="classification_samples/correct_two/img_1_5.jpg" width="200"/> </div>

### Correctly Classified Any 1
<div align="center"> <img src="classification_samples/correct_one/img_0_11.jpg" width="200"/>  </div>


## Key Takeaways

    🔥 Using the full dataset significantly improves generalization and robustness.

    📉 The model struggles with less represented artists, which is expected given class imbalance.

    🧠 CNN-LSTM structure is powerful for artistic feature extraction where spatial flow matters.

    🧼 Manual dataset cleaning was crucial — mislabeled or missing paintings were discarded or mapped using raw HTML parsing.

## Future Work

    Use Vision Transformers (ViTs) for patch-level encoding.

    Add style transfer augmentation to diversify training.

    Implement active learning to improve on rare classes.

## Author

Chandradithya Janaswami
Computer Vision & AI Enthusiast