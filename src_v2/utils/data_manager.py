#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

# data_manager.py: module for loading and preparing data. Also for displaying some statistics.

from urllib.request import urlretrieve
import zipfile, os
from subprocess import call
import pandas as pd


def movielens_load_data(dataset):
    #'''
    # load data from MovieLens 10M Dataset
    # http://grouplens.org/datasets/movielens/
    
    # Note that this method uses ua.base and ua.test in the dataset.
    # :return: train_users, train_x, test_users, test_x
    # :rtype: list of int, numpy.array, list of int, numpy.array
    # '''

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    DATASET_SUBFOLDER = "../data/movielens_10m/"
    DATASET_FILE_NAME = "movielens_10m.zip"  
    DATASET_UNZIPPED_FOLDER = "ml-10M100K/"

    try:
        data_file = zipfile.ZipFile(DATASET_SUBFOLDER + DATASET_FILE_NAME)  # open zip file

    except(FileNotFoundError, zipfile.BadZipFile):
        print("Unable to find data zip file. Downloading...")
        download_from_URL(DATASET_URL, DATASET_SUBFOLDER, DATASET_FILE_NAME)
        data_file = zipfile.ZipFile(DATASET_SUBFOLDER + DATASET_FILE_NAME)  # open zip file
    
    data_path = data_file.extract(DATASET_UNZIPPED_FOLDER + "ratings.dat", 
                                    path=DATASET_SUBFOLDER)  # extract data

    # load the dataset in a Pandas dataframe
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(data_path, delimiter='::', header=None, 
                names=cols, usecols=cols[0:3], # do not consider timestamp 
                engine='python')

    data_dir = DATASET_SUBFOLDER + DATASET_UNZIPPED_FOLDER
    implicit_data_file = 'ratings_implicit.txt'

    # convert ratings into implicit
    ratings_df = map_ratings(ratings_df, data_dir, implicit_data_file) 

    # splitting
    five_fold_splitting(data_dir, implicit_data_file)

    return ratings_df    


def download_from_URL(URL, folder_path, file_name):
    '''
    '''
    # If directory does not exist, create
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Downloading: {}".format(URL))
    print("In folder: {}".format(folder_path + file_name))

    try:
        urlretrieve(URL, folder_path + file_name)  # copy network object to a local file

    except urllib.request.URLError as urlerror:  # @TODO: handle network connection error
        print("Unable to complete automatic download, network error")
        raise urlerror


def map_ratings(ratings_df, data_dir, implicit_data_file):
    ''' transforms a explicit ratings dataset into implicit.
    Mainly keeps only ratings > 3 and treat the other ratings as missing entries. 
    Retained ratings are converted to 1 '''

    print("Mapping into implicit ratings ...")

    ratings_df['rating'] = ratings_df['rating'].apply(lambda x: 1 if x > 3.0 else 0)

    # save new formatted file
    ratings_df.to_csv(data_dir + implicit_data_file, index=False,  
                    header=None, sep=" ")
    
    return ratings_df


def five_fold_splitting(data_dir, data_file):
    ''' execute 5-fold cross-validation subset generation script '''

    print("Five fold splitting ...")
    
    dirname = os.path.dirname(os.path.abspath(__file__))
    cmd = os.path.join(dirname, 'split_ratings.sh')
    
    call(['bash', cmd, 
        data_dir, # param 1 
        data_file # param 2
    ])