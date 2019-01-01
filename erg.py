import os
import sys
import math
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
def euclidian_distance(vector_a, vector_b):

    return math.sqrt(sum([(vector_a - vector_b) ** 2 for vector_a, vector_b in zip(x, y)]))

def cleaning_data():
    """This method prepares the data for processing by BSAS.
    
        Especially it drops the unnecessary columns, removes the rows of movies 
        with unknown genre and computes the average rating of each user of each
        movie genre.
    """


    # Getting absolute path to the data files

    udata = os.path.abspath("u.data")
    uitem = os.path.abspath("u.item")


    # Creating User Ratings Matrix

    data_cols = ['user id', 'movie id', 'rating', 'timestamp']

    ratings = pd.read_csv(udata, sep='\t', names=data_cols,
        encoding='latin-1')

    ratings.drop(columns="timestamp", inplace=True, axis=1)

    print(ratings.head(25))

    # Creating Movies Matrix

    movie_cols = ['movie id', 'movie title', 'release date', 'video release date',
                'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                'Thriller', 'War', 'Western']

    movies = pd.read_csv(uitem, sep='|', names=movie_cols,
                        encoding='latin-1')

    movies = movies[movies.unknown == 0]

    movies.drop(columns=['video release date', 'release date', 'IMDb URL', 'unknown'], inplace=True, axis=1)

    print(movies.head(25))

    # Merging Ratings and Movies 

    final_matrix = pd.merge(ratings, movies, on='movie id', how='inner')
    print(final_matrix.sort_values(by=['user id']).head(100))
        
    df=final_matrix.replace(0, np.NaN) #all 0 values transform to NaN

    df.loc[(df['Action']==1),'Action']=df['rating'] #if Action is the genre (value 1) then becomes the value of rating etc...
    df.loc[(df['Adventure']==1),'Adventure']=df['rating']
    df.loc[(df['Animation']==1),'Animation']=df['rating']
    df.loc[(df['Children\'s']==1),'Children\'s']=df['rating']
    df.loc[(df['Comedy']==1),'Comedy']=df['rating'] 
    df.loc[(df['Crime']==1),'Crime']=df['rating']
    df.loc[(df['Documentary']==1),'Documentary']=df['rating']
    df.loc[(df['Drama']==1),'Drama']=df['rating']
    df.loc[(df['Fantasy']==1),'Fantasy']=df['rating']
    df.loc[(df['Film-Noir']==1),'Film-Noir']=df['rating']
    df.loc[(df['Horror']==1),'Horror']=df['rating']
    df.loc[(df['Musical']==1),'Musical']=df['rating']
    df.loc[(df['Mystery']==1),'Mystery']=df['rating']
    df.loc[(df['Romance']==1),'Romance']=df['rating']
    df.loc[(df['Sci-Fi']==1),'Sci-Fi']=df['rating']
    df.loc[(df['Thriller']==1),'Thriller']=df['rating']
    df.loc[(df['War']==1),'War']=df['rating']
    df.loc[(df['Western']==1),'Western']=df['rating']
    
    grouped=df.groupby('user id').mean() #group by user id and calculate the avg of non-NaN values
    
    grouped.drop(columns=['movie id','rating'], inplace=True, axis=1) #leave only user id and agv for every genre
    print(grouped)
    
cleaning_data()

