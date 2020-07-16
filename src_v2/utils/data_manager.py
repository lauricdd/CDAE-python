#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

# data_manager.py: module for loading and preparing data. Also for displaying some statistics.

from urllib.request import urlretrieve, URLError
import zipfile, os
import subprocess
import pandas as pd
import ssl
import kaggle
import numpy as np


### GENERAL UTILS ### 

def download_from_URL(URL, folder_path, file_name):
    '''
    '''
    
    # use unverified ssl 
    ssl._create_default_https_context = ssl._create_unverified_context

    # if directory does not exist, create
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Downloading: {}".format(URL))
    print("In folder: {}".format(folder_path))

    try:
        urlretrieve(URL, folder_path + file_name)  # copy network object to a local file

    except URLError as urlerror:  # @TODO: handle network connection error
        print("Unable to complete automatic download, network error")
        raise urlerror


def download_dataset_from_kaggle(dataset_name, DATASET_SUBFOLDER):
    '''
    '''
    # list datasets
    # kaggle datasets list 
    
    if(dataset_name == "netflix_prize"):
        print("\n")
        print("="*100)
        print("netflix-prize dataset files\n")
        os.system("kaggle datasets files lauraschiatti/netflix-prize")

        data_files = ["netflix_prize.txt"] 
        for filename in data_files:
            command =  "kaggle datasets download -f " + str(filename)+ " -p ../data/netflix_prize --unzip lauraschiatti/netflix-prize"
            os.system(command)

        # --unzip not working. Unzip data files manually
        unzip_all(DATASET_SUBFOLDER)  

        print("="*100, "\n")

        return DATASET_SUBFOLDER + data_files[0]


def unzip_all(DATASET_SUBFOLDER):
    '''
        unzip all files in a given directory
    '''
    for filename in os.listdir(DATASET_SUBFOLDER):
        if filename.endswith(".zip"):
            print(filename + " unzipped")
            try:
                zip = zipfile.ZipFile(DATASET_SUBFOLDER + filename)  # open zip file
                zip.extractall(path=DATASET_SUBFOLDER)  # extract data

                filepath = DATASET_SUBFOLDER + filename
                remove_file(filepath)

            except(FileNotFoundError, zipfile.BadZipFile):
                print("Unable to find data zip file")


def remove_file(filepath):
    try:
        os.remove(filepath)
        print(filepath + " removed")
    except OSError as e: 
        print("error: ", e) 


### PREPROCESSING ### 

def prepare_data(data_name, DATASET_URL=None, DATASET_SUBFOLDER=None, DATASET_FILE_NAME=None, DATASET_UNZIPPED_FOLDER=None):
    '''
    load dataset from URL
    
    # :return: ratings_df
    '''

    if data_name == "movielens_10m":

        try:
            data_file = zipfile.ZipFile(DATASET_SUBFOLDER + DATASET_FILE_NAME)  # open zip file
        
        except(FileNotFoundError, zipfile.BadZipFile):
            print("Unable to find data zip file. Downloading...")
            download_from_URL(URL=DATASET_URL, folder_path=DATASET_SUBFOLDER, file_name=DATASET_FILE_NAME)
            
            data_file = zipfile.ZipFile(DATASET_SUBFOLDER + DATASET_FILE_NAME)  # open zip file
        
        filepath = data_file.extract(DATASET_UNZIPPED_FOLDER + "ratings.dat", 
                                        path=DATASET_SUBFOLDER)  # extract data      

        # load the dataset
        # format: user_id::movie_id::rating::timestamp
        cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings_df = pd.read_csv(filepath, delimiter='::', header=None, 
                names=cols, usecols=cols[0:3], # do not consider timestamp 
                engine='python')

        # remove zip file 
        filepath = DATASET_SUBFOLDER + DATASET_FILE_NAME
        remove_file(filepath)

    elif data_name == "netflix_prize":
        
        # load the dataset
        filepath = download_dataset_from_kaggle("netflix_prize", DATASET_SUBFOLDER)
        
        # format: Cust_Id,Movie_Id,rating,timestamp
        keep_col = ['Cust_Id','Movie_Id','Rating']
        ratings_df = pd.read_csv(filepath, index_col=False, usecols=keep_col)[keep_col]
        ratings_df.rename(columns={'Cust_Id': 'user_id', 'Movie_Id': 'movie_id', 'Rating': 'rating'}, inplace=True)


    # data exploration (summary statitics) before preprocessing
    print("{} statistics BEFORE preprocessing ... ".format(data_name))
    num_users, num_items, num_total_ratings = dataset_statistics(data_name, ratings_df)
        
    implicit_data_file = 'ratings_implicit.txt'
    
    # if implicit dataset does not exist, create
    if not os.path.exists(DATASET_SUBFOLDER + implicit_data_file): 
        
        # convert ratings into implicit
        ratings_df = convert_ratings_into_implicit(ratings_df) 
        print(ratings_df)
        print("="*100)

        # rescale user IDs to successive one ranged IDs (no need to rescale movie IDs)
        
        # rescale user_id  
        original_user_id_is_sorted = ratings_df["user_id"].is_monotonic
        final_ratings_df = rescale_ids(ratings_df, "user_id", original_user_id_is_sorted)  
        print(final_ratings_df)
        print("="*100)

        # rescale movie_id
        original_movie_id_is_sorted = ratings_df["movie_id"].is_monotonic
        final_ratings_df = rescale_ids(final_ratings_df, "movie_id", original_movie_id_is_sorted) 
        print(final_ratings_df)
        print("="*100)

        # TODO: check testing
        # check correpondence between original and new IDs
        # if data_name == "movielens_10m": # only for movielens_dataset since considers specific tuples
        #     test_movielens_10m_rescaling(ratings_df, final_ratings_df)  

        # save new formatted file
        final_ratings_df.to_csv(DATASET_SUBFOLDER + implicit_data_file, index=False, 
                header=None, sep="\t") # use \t as separator as in politic_old and politic_new

        # remove explicit dataset file
        # remove_file(filepath)

    if data_name == "movielens_10m":
        # if any test or train fold exists skip
        if not os.path.exists(DATASET_SUBFOLDER + 'Train_ratings_fold_1'):
            # ratings five-fold splitting
            k_fold_splitting(DATASET_SUBFOLDER, implicit_data_file)

    return final_ratings_df    


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
    subprocess.call(['bash', cmd, 
        data_dir[:-1], # param 1 
        data_file # param 2
    ])


### RESCALING ###

def gen_new_id(ratings_df, id_name):
    sorted_id = ratings_df[id_name].is_monotonic
    
    # check if the values in the index are monotonically increasing 
    if not sorted_id:
        print("{} in ascending order BEFORE sorting?: {} \n".format(id_name, sorted_id))

        # create an object, id_name_df, that only contains the `id_name` column sorted
        id_name_sorted_df = ratings_df.sort_values(by=id_name)[[id_name]] 

        # check whether id_name is actually ordered
        print("{} in ascending order AFTER sorting?: {}".format(id_name, id_name_sorted_df[id_name].is_monotonic))  
    else:
        print("NO sorting needed for {} \n".format(id_name))
        id_name_sorted_df = ratings_df

    # remove duplicated IDs
    print("{}_sorted_df shape BEFORE removing duplicates: {}".format(id_name, id_name_sorted_df.shape))
    id_name_sorted_df = id_name_sorted_df.drop_duplicates() 
    print("{}_sorted_df shape AFTER removing duplicates: {}".format(id_name, id_name_sorted_df.shape))

    # create `NEW_id_name` column with values which increments by one 
    # for every change in value of id_name column. 
    i = id_name_sorted_df[id_name]
    new_id_name = 'NEW_' + id_name
    id_name_sorted_df[new_id_name] = i.ne(i.shift()).cumsum()-1 # start IDs from 0
    
    return id_name_sorted_df


def rescale_ids(ratings_df, id_name, original_id_name_is_sorted):
    ''' 
        create successive one ranged IDs for user_id and movie_id columns 
    '''

    min_value = ratings_df[id_name].min()
    max_value = ratings_df[id_name].max()
    unique_values = ratings_df[id_name].nunique() #num_users

    # create NEW_index for `id_name` attribute
    if (max_value-min_value) > unique_values: # TODO: >=???
        
        string = "rescaling {} ... \nmin_value: {} \nmax_value: {}  \
                \nunique_values: {}\n".format(id_name,min_value, max_value,unique_values)
        print(string)
        
        id_name_sorted_df = gen_new_id(ratings_df, id_name)
        print("id_name_sorted_df ... \n", id_name_sorted_df)

        # set `NEW_id_name` values based on its corresponding `id_name` by means of a
        # left join on `id_name` (only column name in both dataframes)
        if not original_id_name_is_sorted:
            print("left join with `NEW_{}` to final_ratings_df dataframe".format(id_name))
            final_ratings_df = ratings_df.merge(id_name_sorted_df, on=id_name, how='left') 
        
        # just append `NEW_id_name` to the  dataframe
        else:
            print("appending `NEW_{}` to final_ratings_df dataframe".format(id_name))
            final_ratings_df = pd.concat([ratings_df, id_name_sorted_df['NEW_' + id_name]], axis=1, sort=False)

        if id_name == 'user_id' and 'NEW_user_id' in final_ratings_df.columns: 
            print("renaming NEW_user_id ... \n")
            final_ratings_df = final_ratings_df[['NEW_user_id', 'movie_id', 'rating']]  
            final_ratings_df.columns = ['user_id', 'movie_id', 'rating'] # columns renaming
        
        if id_name == 'movie_id' and 'NEW_movie_id' in final_ratings_df.columns:
            print("renaming NEW_movie_id ... \n") 
            final_ratings_df = final_ratings_df[['user_id', 'NEW_movie_id', 'rating']]  
            final_ratings_df.columns = ['user_id', 'movie_id', 'rating'] # columns renaming

        return final_ratings_df
        
    else:
        print("No rescaling needed for {} ... \n".format(id_name))

        return ratings_df



def test_movielens_10m_rescaling(untouched_ratings_df, final_ratings_df):
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


### LOADING / STATISTICS ###
    
def load_data(DATASET_SUBFOLDER):
    '''load implicit dataset in a pandas dataframe '''
    ratings_df = pd.read_csv(DATASET_SUBFOLDER + "ratings_implicit.txt", delimiter="\t", header=None,
            names=['user_id', 'movie_id', 'rating'])

    return ratings_df


def dataset_statistics(data_name, ratings_df):
    num_users = unique_values(ratings_df["user_id"])
    num_items = unique_values(ratings_df["movie_id"])
    num_total_ratings =  ratings_df.shape[0]

    print("=" * 100)
    print(data_name + " statistics ...")
    
    # min and max value for each colum of a given dataframe
    min_max = ratings_df.describe().loc[['min','max']].astype(int)
    print("min_max values: \n",  min_max)

    statistics_string = "\nNumber of unique items: {}, \nNumber of unique users: {}, \n Num total ratings: {}, \nAverage interactions per user: {:.2f},  \
        \nAverage interactions per item {:.2f}, \nSparsity {:.2f}%".format(
        num_items,
        num_users,
        num_total_ratings,
        (num_total_ratings/num_users),
        (num_total_ratings/num_items),
        (1-float(num_total_ratings)/(num_items*num_users))*100
    )

    print(statistics_string)
    print("=" * 100)

    return num_users, num_items, num_total_ratings

