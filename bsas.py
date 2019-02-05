# Standard Library Imports
import os
import math
from random import shuffle
from collections import Counter
import itertools
import operator
from time import time
# Third Party Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
#end of Imports



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
	"""
	Update of Representative of the Cluster
	"""
	new_mC=[None]*18
	for i in range(len(prev_mC)):
		new_mC[i]=(prev_mC[i]+new_C[i])/2
	return new_mC
	
	



def find_minimum_distance_from_mC(mC_list, new_C):
	"""
	Finds the minimum distance from a Vector to a Cluster inside the Cluster list
	Args:
		mC_list: This is the list of all Representatives
		new_C: This is the vector we compare
	Returns:
		mC_index: This is the Representative's index value of the nearest cluster
		minimum_distance: This is the distance between the new_C and the nearest cluster's representative
	"""
	minimum_distance=10000000000
	mC_index=None
	for i in range(len(mC_list)):
		current_distance = euclidian_distance(new_C, mC_list[i])
		if(current_distance < minimum_distance):
				minimum_distance=current_distance
				mC_index=i
			
			
	return mC_index, minimum_distance
	

	

	
def BSAS_algorithm(vectors, theta, max_clusters):
	"""
	Performs BSAS
	Args:
		vectors: This is the list of all vectors
		theta: the threshold of dissimilarity
		max_clusters: the maximum allowable number of clusters
	Returns:
		number_of_clusters: the total number of clusters that created
	"""
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
        vectors: This is the list of all vectors
    Returns:
       min(distances): the minimum distance
       max(distances): the maximum distance
    """

    distances = []
		
    for i in tqdm(range(len(vectors))):
        for j in range(i + 1, len(vectors)):
            distance = euclidian_distance(vectors[i], vectors[j])
            distances.append(distance)

    return min(distances), max(distances)



def most_common_in_list(lst):
	"""
	Returns the value of most common number of a list
	if there are more than one common values, returns the first one
	"""
	return max(set(lst), key=lst.count)
  
  
def theta_range_calc(theta_min, theta_max, theta_step):
	"""
	Calculates all the theta range
	Args:
		theta_min: the minimum value of theta (a)
		theta_max: the maximum value of theta (b)
		theta_step: the step of theta (c)
	Returns:
		theta_range: a list with all the theta values available
	"""
	theta_range = np.arange(theta_min, theta_max, theta_step)
	return theta_range



def find_index_with_flatter_value(list_of_clusters_per_theta):
	"""
	Finds the index value with the most flatter value in the diagram of Clusters-Theta
	Args:
		list_of_clusters_per_theta: a two dimension list that the first dimension is the thetas 
		and the second dimension are all number of clusters per different execution
		(i.e. [[700,705,...],[504,500,...],[250,548,...],[128,132,...],...])
	Returns:
		index: the number of index in which the flatter value exists
	"""
	print("\n\nCalculating flatter value in diagram Clusters-Theta...")
	most_frequent_value_per_theta=[]
	for i in range(len(list_of_clusters_per_theta)):
		list_of_clusters_per_theta[i]=Counter(list_of_clusters_per_theta[i])
		if(list_of_clusters_per_theta[i].most_common()[0][0] != 1):
			most_frequent_value_per_theta.append(list_of_clusters_per_theta[i].most_common()[0][1])
	index=most_frequent_value_per_theta.index(max(most_frequent_value_per_theta)) 
	return index  
	 
	 
	 
def get_clusters_count():
    """
    Returns the most frequently occurring number of clusters and the theta value
    """

    print("Loading data...")
    try:
    	vectors = get_data()
    	print("Data Loaded Successful!")
    except:
    	print("Data failed to load.\nPlease check if data files exist")
    	return
    
    step=1
    max_clusters=1000
    print("Calculating minimum and maximum distance between all vectors...")
    a, b = min_max_between_all(vectors)
    theta_range=theta_range_calc(a,b,step)
    m=len(theta_range)
    number_of_clusters_per_theta=[[0 for i in range(m)] for j in range(m)]
    
    for times in tqdm(range(len(theta_range)), desc="Calculating clusters:"):
    	vectors = get_data()
    	shuffle(vectors)
    	#print("Time running: "+str(times)+"/"+str(len(theta_range)))
    	for Theta in theta_range:
    		num_of_clusters=BSAS_algorithm(vectors,Theta,max_clusters)
    		number_of_clusters_per_theta[int(Theta)][times]=num_of_clusters
    		#print("a="+str(a)+"  b="+str(b)+"  Theta="+str(Theta)+"  Number of Clusters= "+str(num_of_clusters))
    		
    	
    	
    print("\n\nAverage Number of Clusters per Theta")	
    for Theta in theta_range:
    	print("a="+str(a)+"  b="+str(b)+"  Theta="+str(Theta)+"  Number of Clusters= "+str(most_common_in_list(number_of_clusters_per_theta[int(Theta)])))
    	
    indx=find_index_with_flatter_value(number_of_clusters_per_theta)
    return 	number_of_clusters_per_theta[indx].most_common()[0][0], theta_range[indx]
	 
	 	


if __name__ == '__main__':
	
	ts=time()
	number_of_clusters, theta = get_clusters_count()
	print("\nSolution:\nNumber of Clusters: "+str(number_of_clusters)+" Theta: "+str(theta))
	print("Execution Time Elapsed: "+str(round(time()-ts,2))+" seconds")
	
	


    
