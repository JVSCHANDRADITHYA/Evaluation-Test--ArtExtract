'''
The WikiArt Dataset provided is not completely labeled. The labels are provided in a separate file.
The dataset contains images of paintings and their corresponding labels. The labels include artist, genre, and style. However, some images are not labelled at all in neither of the three classes.
This folder make_dataset contains scripts to extract artist names from the image file names and to check for missing images in the dataset. The original one has only 13k images labelled for artist and only 22 classes of srtist. 
This Updated version completely labels the dataset with over 128 classes in artist, 11 in genre and 27 in style.

Further this would be uploaded to my github repository and the link would be shared after completion of the project.
This dataset would be completely labelled and labels would be defined for every class.

My approach to this problem is to extract artist names from the image file names and to check for missing images in the dataset. 
This I have done through the script extract_artist_name.py in the make_dataset folder where I extracted the common words from the image file names and assigned the most common word as the artist name for that class.

You can find the artist class.txt file in the make_dataset/files folder which contains the artist class and the artist name for that class alogn witht he completely labelled csv file in the files folder.
'''