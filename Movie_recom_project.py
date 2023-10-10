import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


                                                             # Data Collection and Pre-processing

# Loading the data from the csv to pandas dataframe 
movies_data = pd.read_csv("E:\movies.csv")
#print(movies_data.head())


# to find number of rows and columns in Dataframe
#print(movies_data.shape)


# selecting the relevant feature for recommendations
selected_features = ["genres","keywords","tagline","cast","director"]
#print(selected_features)


# replacing null values with null 
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')


# combining all the 5 selected features
combined_feature = movies_data["genres"] + ' ' + movies_data["keywords"] + ' ' + movies_data["tagline"] + ' ' + movies_data["cast"] + ' ' + movies_data["director"]
#print(combined_feature)


# converting the text data to feature vectors(numerical data)
vectorizer = TfidfVectorizer() 
# now to fit and transform the data
feature_vectors = vectorizer.fit_transform(combined_feature)
#print(feature_vectors)


# Cosine Similarity
# getting the similaarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
#print(similarity)
#print(similarity.shape)


# Getting the movie name from the user
movie_name = input("Enter a Movie Name :")


# Creating a list with all the movie name given in the dataset
list_of_allTitles = movies_data["title"].tolist()
#print(list_of_allTitles)


# Finding the close match for the movie name given by the user
find_close_Match = difflib.get_close_matches(movie_name,list_of_allTitles)
#print(find_close_Match)
# As we need a single movie name we take the 1st name in find_close_Match
close_match = find_close_Match[0]
#print(close_match)


# Finding the index of the movie
filtered_title = movies_data[movies_data["title"] == close_match]
title_index = filtered_title.index[0]
# OR
# title_index = movies_data[movies_data.title == close_match]['index'].values[0]
#print(title_index)


# Getting the list of similar movies
similarity_score = list(enumerate(similarity[title_index]))
#print(similarity_score)


# sorting the movies based on similarity score
sorted_similarity_score = sorted(similarity_score, key = lambda x:x[1], reverse = True)
#print(sorted_similarity_score)


# print the name of similar movies based on the index
print("THE MOVIES SUGGESTED FOR YOU")
for index in range(30):
    ind = sorted_similarity_score[index][0]
    print(index+1,"-",movies_data.title[ind])
