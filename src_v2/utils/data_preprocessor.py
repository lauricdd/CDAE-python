import numpy as np
from os.path import exists
import re

# Read ratings file and train/test split
def read_rating(path,data_name, num_users, num_items, num_total_ratings, a, b, test_fold,random_seed):

    # train and test sets
    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    # initialize train/test vectors (num_users*num_items size)
    R = np.zeros((num_users,num_items))

    mask_R = np.zeros((num_users, num_items))
    C = np.ones((num_users, num_items)) * b # ???

    train_R = np.zeros((num_users, num_items))
    test_R = np.zeros((num_users, num_items))

    train_mask_R = np.zeros((num_users, num_items))
    test_mask_R = np.zeros((num_users, num_items))

    if (data_name == 'politic_new') or (data_name == 'politic_old') \
        or (data_name == 'movielens_10m'):

        num_train_ratings = 0
        num_test_ratings = 0

        train_file_name = 'Train_ratings_fold_' + str(test_fold)
        test_file_name = 'Test_ratings_fold_' + str(test_fold)

        ''' load train fold '''
        print("\nLoad train fold ... ", str(test_fold))
        with open(path + train_file_name) as f1:
            lines = f1.readlines()
            for line in lines:
                user, item, voting = line.split("\t")
                user = int(user)
                item = int(item)
                voting = int(voting)
                
                # if implicit ratings are 1 and -1 (applies for politic_new and politic_old) 
                if voting == -1:
                    voting = 0

                ''' Total '''
                R[user, item] = voting
                mask_R[user, item] = 1

                ''' Train '''
                train_R[user, item] = int(voting)
                train_mask_R[user, item] = 1
                C[user, item] = a 

                user_train_set.add(user)
                item_train_set.add(item)
                num_train_ratings = num_train_ratings + 1


        ''' load test fold '''
        print("Load test fold ... ", str(test_fold))
        with open(path + test_file_name) as f2:
            lines = f2.readlines()
            for line in lines:
                user, item, voting = line.split("\t")
                user = int(user)
                item = int(item)
                voting = int(voting)
                
                # if implicit ratings are 1 and -1 (applies for politic_new and politic_old)
                if voting == -1:
                    voting = 0

                ''' Total '''
                R[user, item] = voting
                mask_R[user, item] = 1

                ''' Test '''
                test_R[user, item] = int(voting)
                test_mask_R[user, item] = 1

                user_test_set.add(user)
                item_test_set.add(item)

                num_test_ratings = num_test_ratings + 1

    # train_mask_R_sum = np.sum(train_mask_R).astype(np.int32)

    # print("num_train_ratings", num_train_ratings)
    # print("np.sum(train_mask_R)", np.sum(train_mask_R))
    # assert num_train_ratings == train_mask_R_sum
    
    # print("num_test_ratings", num_test_ratings)
    # print("np.sum(test_mask_R)", np.sum(test_mask_R))
    # assert num_test_ratings == np.sum(test_mask_R)

    # print("num_total_ratings", num_total_ratings)
    # print("num_train_ratings + num_test_ratings", num_train_ratings + num_test_ratings)
    # assert num_total_ratings == num_train_ratings + num_test_ratings

    return R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, \
        user_train_set,item_train_set,user_test_set,item_test_set