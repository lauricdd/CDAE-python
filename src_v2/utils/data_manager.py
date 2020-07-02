#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

# data_manager.py: module for loading and preparing data. Also for displaying some statistics.

from urllib.request import urlretrieve
import zipfile, os
from subprocess import call
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def movielens_10m_prepare_data(dataset):
    '''
    load data from MovieLens 10M Dataset
    http://grouplens.org/datasets/movielens/
    
    # :return: ratings_df
    '''

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    DATASET_SUBFOLDER = "../data/movielens_10m/"
    DATASET_FILE_NAME = "movielens_10m.zip"  
    DATASET_UNZIPPED_FOLDER = "ml-10M100K/"

    try:
        data_file = zipfile.ZipFile(DATASET_SUBFOLDER + DATASET_FILE_NAME)  # open zip file

    except(FileNotFoundError, zipfile.BadZipFile):
        print("Unable to find data zip file. Downloading...")
        download_from_URL(URL=DATASET_URL, folder_path=DATASET_SUBFOLDER, file_name=DATASET_FILE_NAME)
        
        data_file = zipfile.ZipFile(DATASET_SUBFOLDER + DATASET_FILE_NAME)  # open zip file
    
    data_path = data_file.extract(DATASET_UNZIPPED_FOLDER + "ratings.dat", 
                                    path=DATASET_SUBFOLDER)  # extract data

    # load the dataset
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(data_path, delimiter='::', header=None, 
            names=cols, usecols=cols[0:3], # do not consider timestamp 
            engine='python')


    # if any test or train fold exists skip
    if not os.path.exists(DATASET_SUBFOLDER + 'Train_ratings_fold_1'):
        implicit_data_file = 'ratings_implicit.txt'

        # if implicit dataset does not exist, create
        if not os.path.exists(DATASET_SUBFOLDER + implicit_data_file): 
            
            # convert ratings into implicit
            ratings_df = convert_ratings_into_implicit(ratings_df) 
            
            # rescale user and movie IDs to successive one ranged IDs
            ratings_df = rescale_ids(ratings_df)

            # save new formatted file
            ratings_df.to_csv(DATASET_SUBFOLDER + implicit_data_file, index=False,  
                    header=None, sep="\t") # use \t as separator as in politic_old and politic_new

            # TODO: remove explicit data (ratings.dat), implicit & zip files

        # ratings five-fold splitting
        k_fold_splitting(DATASET_SUBFOLDER, implicit_data_file)


    ####################################################################################################
    # https://botbark.com/2019/12/28/scaling-data-range-using-min-max-scaler/
    # Scale features to a range using MinMaxScaler
    # Transform features by scaling each feature to a given range
    # This range can be set by specifying the feature_range parameter (default at (0,1)).
    # num_users = unique_values(ratings_df["user_id"])
    # user_id_scaler = MinMaxScaler(feature_range=(1, num_items), copy=False)
    # ratings_df["user_id"] = user_id_scaler.fit_transform(ratings_df["user_id"].values.reshape(-1, 1))

    # print("\nafter scaling ...\n")
    # print(ratings_df[100:])
    # min_max = ratings_df.describe().loc[['min','max']]
    # print("min_max values after scaling... \n", min_max)

    ####################################################################################################

    return ratings_df    


def download_from_URL(URL, folder_path, file_name):
    '''
    '''
    # if directory does not exist, create
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Downloading: {}".format(URL))
    print("In folder: {}".format(folder_path + file_name))

    try:
        urlretrieve(URL, folder_path + file_name)  # copy network object to a local file

    except urllib.request.URLError as urlerror:  # @TODO: handle network connection error
        print("Unable to complete automatic download, network error")
        raise urlerror


def convert_ratings_into_implicit(ratings_df):
    ''' transforms a explicit ratings dataset into implicit.
    Mainly keeps only ratings > 3 and treat the other ratings as missing entries. 
    Retained ratings are converted to 1 '''

    print("Converting explicit ratings into implicit ...")

    ratings_df['rating'] = ratings_df['rating'].apply(lambda x: 1 if x > 3.0 else 0)
    
    return ratings_df


def k_fold_splitting(data_dir, data_file):
    ''' execute k-fold cross-validation subset generation script '''

    print("5-fold data splitting ...")
    
    dirname = os.path.dirname(os.path.abspath(__file__))
    cmd = os.path.join(dirname, 'split_ratings.sh')
    
    # execute a Unix shell script
    call(['bash', cmd, 
        data_dir[:-1], # param 1 
        data_file # param 2
    ])


def rescale_ids(ratings_df):
    ''' create successive one ranged IDs for user_id and movie_id columns '''
    untouched_ratings_df = ratings_df
    ratings_df = gen_new_user_id(ratings_df)
    movie_id_sorted_df = gen_new_movie_id(ratings_df)

    # set NEW_movie_id values based on its corresponding movie_id by means of a
    # left join on movie_id (only column name in  both dataframes)
    final_ratings_df = ratings_df.merge(movie_id_sorted_df, on='movie_id', how='left') 
    
    # check correpondence between original and new ids
    test_rescaling(untouched_ratings_df, final_ratings_df)

    final_ratings_df = final_ratings_df[['NEW_user_id', 'NEW_movie_id', 'rating']]  
    final_ratings_df.columns = ['user_id', 'movie_id', 'rating'] # rename cols

    print("final_ratings_df\n", final_ratings_df)

    return final_ratings_df


def gen_new_user_id(ratings_df):
    # check whether user_id index is sorted in ascending order
    print("user_id in ascending order BEFORE sorting?: ", ratings_df.user_id.is_monotonic) 
    # create NEW_user_id column with values which increments by one 
    # for every change in value of user_id column. 
    i = ratings_df.user_id  
    ratings_df['NEW_user_id'] = i.ne(i.shift()).cumsum()-1 # start ID from 0

    return ratings_df


def gen_new_movie_id(ratings_df):
    print("movie_id in ascending order BEFORE sorting?: ", ratings_df.movie_id.is_monotonic)
    
    # create an object, movie_id_df, that only contains the `movie_id` column sorted
    movie_id_sorted_df = ratings_df.sort_values(by='movie_id')[['movie_id']]    

    # check whether movie_id is actually ordered
    print("movie_id in ascending order AFTER sorting?: ", movie_id_sorted_df.movie_id.is_monotonic)

    # remove duplicate ids
    movie_id_sorted_df = movie_id_sorted_df.drop_duplicates() 
    print("movie_id_sorted_df shape AFTER removing duplicates", movie_id_sorted_df.shape)

    # create NEW_movie_id column with values which increments by one 
    # for every change in value of movie_id column. 
    i = movie_id_sorted_df.movie_id  
    movie_id_sorted_df['NEW_movie_id'] = i.ne(i.shift()).cumsum()-1 # start ID from 0
    
    return movie_id_sorted_df


def test_rescaling(untouched_ratings_df, final_ratings_df):
    ''' use testing rows to check correspondance between original 
    movie_id and NEW_movie_id. Same for user_id and NEW_user_id '''

    print("Dataframe rows before rescaling user_ids")
    row_1 = untouched_ratings_df[(untouched_ratings_df['movie_id'] == 8874) & (untouched_ratings_df['user_id'] == 92 )]
    row_2 = untouched_ratings_df[(untouched_ratings_df['movie_id'] == 32076) & (untouched_ratings_df['user_id'] == 100 )]
    row_3 = untouched_ratings_df[(untouched_ratings_df['movie_id'] == 973) & (untouched_ratings_df['user_id'] == 112 )]
    row_4 = untouched_ratings_df[(untouched_ratings_df['movie_id'] == 538) & (untouched_ratings_df['user_id'] == 122 )]
    row_5 = untouched_ratings_df[(untouched_ratings_df['movie_id'] == 316) & (untouched_ratings_df['user_id'] == 135 )]
    print(row_1, "\n")
    print(row_2, "\n")
    print(row_3, "\n")
    print(row_4, "\n")
    print(row_5, "\n")

    
    print("Testing rows for checking correctness of the rescaling")
    row_1 = final_ratings_df[(final_ratings_df['movie_id'] == 8874) & (final_ratings_df['user_id'] == 92 )]
    # assert row_1['NEW_user_id'] == 84 & row_1['NEW_movie_id'] == 8171
    row_2 = final_ratings_df[(final_ratings_df['movie_id'] == 32076) & (final_ratings_df['user_id'] == 100 )]
    row_3 = final_ratings_df[(final_ratings_df['movie_id'] == 973) & (final_ratings_df['user_id'] == 112 )]
    row_4 = final_ratings_df[(final_ratings_df['movie_id'] == 538) & (final_ratings_df['user_id'] == 122 )]
    row_5 = final_ratings_df[(final_ratings_df['movie_id'] == 316) & (final_ratings_df['user_id'] == 135 )]
    print(row_1, "\n")
    print(row_2, "\n")
    print(row_3, "\n")
    print(row_4, "\n")
    print(row_5, "\n")

    min_max = final_ratings_df.describe().loc[['min','max']].astype(int)
    print("\n ratings_df min and max values AFTER RESCALING... \n", min_max)


def unique_values(column):
    ''' count distinct values of a dataframe column'''
    count = column.nunique() 
    
    return count

    
def load_movielens_10m_data():
    '''load implicit MovieLens dataset in a pandas dataframe '''
    ratings_df = pd.read_csv("../data/movielens_10m/ratings_implicit.txt", delimiter="\t", header=None,
            names=['user_id', 'movie_id', 'rating'])

    return ratings_df


def movielens_10m_statistics(ratings_df):
    num_users = unique_values(ratings_df["user_id"])
    num_items = unique_values(ratings_df["movie_id"])
    num_total_ratings =  ratings_df.shape[0]

    print("=" * 100)
    print("Movielens_10m statistics ...")
    print("=" * 100)
    
    # min and max value for each colum of a given dataframe
    min_max = ratings_df.describe().loc[['min','max']]#.astype(int)
    print("min_max values: \n",  min_max)

    statistics_string = "\nNumber of unique items: {}, \nNumber of unique users: {}, \nAverage interactions per user: {:.2f},  \
        \nAverage interactions per item {:.2f}, \nSparsity {:.2f}%".format(
        num_items,
        num_users,
        (num_total_ratings/num_users),
        (num_total_ratings/num_items),
        (1-float(num_total_ratings)/(num_items*num_users))*100
    )

    print(statistics_string)

    return num_users, num_items, num_total_ratings

