# Standard Library Imports
import os
import json
import sys
import math
from random import shuffle
from collections import Counter
import itertools
import operator
from time import time
# Third Party Imports
import pandas as pd
import numpy as np


NUMBER_OF_USERS = 943
GENRES = ['Action', 'Adventure', 'Animation',
          'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
          'Thriller', 'War', 'Western']


def get_data():
    """This method prepares the data for processing by BSAS.
        Especially it drops the unnecessary columns, removes the rows of movies 
        with unknown genre and computes the average rating of each user of each
        movie genre.
    """

    # Getting absolute path to the data files

    udata = os.path.abspath('dataset/u.data')
    uitem = os.path.abspath('dataset/u.item')

    # Creating User Ratings Matrix

    data_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

    ratings = pd.read_csv(udata, sep='\t', names=data_cols,
                          encoding='latin-1')

    ratings.drop(columns='timestamp', inplace=True, axis=1)

    # Creating Movies Matrix

    movie_cols = ['movie_id', 'movie_title', 'release_date', 'video_release date',
                  'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                  'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']

    movies = pd.read_csv(uitem, sep='|', names=movie_cols,
                         encoding='latin-1')


    movies = movies[movies.unknown == 0]

    movies.drop(columns=['video_release date', 'release_date',
                         'IMDb_URL', 'unknown', 'movie_title'], inplace=True, axis=1)

    # Merging Ratings and Movies

    final_matrix = pd.merge(ratings, movies, on='movie_id', how='inner')

    df = final_matrix.replace(0, np.NaN)  # all 0 values transform to NaN

    for genre in GENRES:
        df.loc[(df[genre] == 1), genre] = df['rating']


    grouped = df.groupby('user_id').mean() #group by user id and calculate the avg of non-NaN values

    grouped.drop(columns=['movie_id', 'rating'], inplace=True, axis=1) #leave only user id and agv for every genre

    grouped.fillna(0, inplace=True)
          
    grouped = grouped.as_matrix()


    return grouped




def euclidian_distance(vector_a, vector_b):
    """Calculates the euclidian distance between two vectors"""

    return math.sqrt(sum([(vector_a - vector_b) ** 2 for vector_a, vector_b in zip(vector_a, vector_b)]))
    
    


def calculate_new_mC(prev_mC, new_C):
	new_mC=[None]*18
	for i in range(len(prev_mC)):
		new_mC[i]=(prev_mC[i]+new_C[i])/2
	return new_mC
	
	



def find_minimum_distance_from_mC(mC_list, new_C):
	minimum_distance=10000000000
	mC_index=None
	for i in range(len(mC_list)):
		current_distance = euclidian_distance(new_C, mC_list[i])
		if(current_distance < minimum_distance):
				minimum_distance=current_distance
				mC_index=i
			
			
	return mC_index, minimum_distance
	

	

	
def BSAS_algorithm(vectors, theta, max_clusters):
  	mC_list=[]
	number_of_vectors_in_each_cluster=[]
			
	mC_list.append(vectors[0])
	number_of_vectors_in_each_cluster.append(1)
	
	for i in range(1,len(vectors)):
		vector=vectors[i]
		mC_index, minimum_distance=find_minimum_distance_from_mC(mC_list,vector)
		if(minimum_distance <= theta):
			mC_list[mC_index]=calculate_new_mC(mC_list[mC_index], vector)
			number_of_vectors_in_each_cluster[mC_index] += 1
		else:
			mC_list.append(vector)
			number_of_vectors_in_each_cluster.append(1)
			
		
	number_of_clusters=len(mC_list)	
				
	return number_of_clusters
	
	
def min_max_between_all(vectors):
    """
    Calculates all the possible distances between all the vector combinations
    Args:
        vectors: This is the list of all vectors.
    Returns:
       min(distances): the minimum distance
       max(distances): the maximum distance
    """

    distances = []

    for i in range(0, len(vectors)):
        for j in range(i + 1, len(vectors)):
            distance = euclidian_distance(vectors[i], vectors[j])
            distances.append(distance)

    return min(distances), max(distances)



def most_common(lst):
    return max(set(lst), key=lst.count)
  
  
def theta_range_calc(theta_min, theta_max, theta_step):
    theta_range = np.arange(theta_min, theta_max, theta_step)
    return theta_range
    
def get_clusters_count():
    """Returns the most frequently occurring number of clusters"""

    
    vectors = get_data()
    step=1
    max_clusters=1000
    a, b = min_max_between_all(vectors)
    theta_range=theta_range_calc(a,b,step)
    m=len(theta_range)
    number_of_clusters_per_theta=[[0 for i in range(m)] for j in range(m)]
    
    for times in range(1,len(theta_range)+1):
    	vectors = get_data()
    	shuffle(vectors)
    	print("Time running: "+str(times)+"/"+str(len(theta_range)))
    	for Theta in theta_range:
    		num_of_clusters=BSAS_algorithm(vectors,Theta,max_clusters)
    		number_of_clusters_per_theta[int(Theta)][times-1]=num_of_clusters
    		print("a="+str(a)+"  b="+str(b)+"  Theta="+str(Theta)+"  Number of Clusters= "+str(num_of_clusters))
    	
    print("\n\nFINAL NUMBER OF CLUSTERS PER THETA")	
    for Theta in theta_range:
    	print("a="+str(a)+"  b="+str(b)+"  Theta="+str(Theta)+"  Number of Clusters= "+str(most_common(number_of_clusters_per_theta[int(Theta)])))



if __name__ == '__main__':
	
	ts=time()
	clusters = get_clusters_count()
	print("time for calculation: ",round(time()-ts,2)," seconds")
	
	


    
