#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import numpy as np
import os
from numpy import inf
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import batch_norm
import functools
import scipy.sparse as sps
import pandas as pd

'''  
    ERROR METRICS used for rating prediction task. Compare predicted rating values (Estimated_R) 
    with observed values (ratings in the test set (test_R)) num_test_ratings = number of observations (N)
'''
def evaluation(test_R, test_mask_R, Estimated_R, num_test_ratings): 
    ''' 
        Root-Mean-Square Error 
        RMSE = sqrt(sum(Pi – Oi)^2 * 1/N)
    '''
    pre_numerator = np.multiply((test_R - Estimated_R), test_mask_R) # observed - predicted values
    numerator = np.sum(np.square(pre_numerator))  # squared differences
    RMSE = np.sqrt(numerator / float(num_test_ratings))

    ''' 
        Mean Absolute Error
        MAE = 1/N * sum(Pi – Oi)
    '''
    pre_numerator = np.multiply((test_R - Estimated_R), test_mask_R)
    numerator = np.sum(np.abs(pre_numerator))
    MAE = numerator / float(num_test_ratings)

    ''' 
        Accuracy 
    '''
    pre_numerator1 = np.sign(Estimated_R - 0.5)
    tmp_test_R = np.sign(test_R - 0.5)

    pre_numerator2 = np.multiply((pre_numerator1 == tmp_test_R), test_mask_R)
    numerator = np.sum(pre_numerator2)
    ACC = numerator / float(num_test_ratings)
    
    '''
        Negative Average Log-Likelihood (NALL)
        loss=-log(y)
    '''
    a = np.log(Estimated_R)
    b = np.log(1 - Estimated_R)
    a[a == -inf] = 0
    b[b == -inf] = 0

    tmp_r = test_R
    tmp_r = a * (tmp_r > 0) + b * (tmp_r == 0)
    tmp_r = np.multiply(tmp_r, test_mask_R)
    numerator = np.sum(tmp_r)
    AVG_loglikelihood = numerator / float(num_test_ratings)

    return RMSE, MAE, ACC, AVG_loglikelihood

''' 
    Get test set relevant items for a given user
'''
def get_relevant_items(user_id, URM_test):
    relevant_items = URM_test[user_id].indices

    return relevant_items

'''
    Check whether recommended items are relevant
'''
def get_is_relevant(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True) # compare elements in both arrays

    return is_relevant

'''
    Mean Average Precision (MAP@K) gives insight into how relevant the list of recommended items are
    Cumulative sum: precision at k=1, at k=2, at k=3 ...
'''
def MAP(is_relevant, relevant_items):
    # MAP@K to average the AP@N metric over all your |U| users. 
    # rel(k) is just an indicator that says whether that kth item was relevant
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    # sum(ratings of recommended items)/N recommended items
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score

def compute_apk(y_true, y_pred, k):
    """
    average precision at k, y_pred is assumed 
    to be truncated to length k prior to feeding
    it to the function
    """
    # convert to set since membership 
    # testing in a set is vastly faster
    actual = set(y_true)
    
    # precision at i is a percentage of correct 
    # items among first i recommendations; the
    # correct count will be summed up by n_hit
    n_hit = 0
    precision = 0
    for i, p in enumerate(y_pred, 1):
        if p in actual:
            n_hit += 1
            precision += n_hit / i

    # divide by recall at the very end
    avg_precision = precision / min(len(actual), k)
    return avg_precision

''' 
    RANKING METRICS used for top-N recommendation task. 
    These metrics treat the recommendation list as a classification of relevant items. 
'''
def top_k_evaluation(test_R, Estimated_R, k): # , num_test_ratings, user_test_set, item_test_set, k=5)
    """
        mean average precision at rank k for the ALS model

        Parameters
        ----------
        model : ALSWR instance
            fitted ALSWR model

        ratings : scipy sparse csr_matrix [n_users, n_items]
            sparse matrix of user-item interactions

        k : int
            mean average precision at k's k
            
        Returns
        -------
        mapk : float
            the mean average precision at k's score
    """
    MAP_k = 0.0
    num_eval = 0

    URM_test = sps.csr_matrix(test_R)

    n_users = URM_test.shape[0]
    
    # print("MAP@", k)
    for user_id in range(n_users):
        # if user_id % 1000 == 0:
        #     print("Evaluated user {} of {}".format(user_id, n_users))

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos-start_pos>0:
            y_true = URM_test.indices[start_pos:end_pos]

            if len(y_true) > 0:
                # predicted ratings for user_id
                u_pred = Estimated_R[user_id] 
                # sort ratings and get top-k highest estimated ratings
                y_pred = np.argsort(u_pred)[::-1][:k] 
                num_eval += 1
                MAP_k += compute_apk(y_true, y_pred, k)

    MAP_k /= num_eval

    return MAP_k


def make_records(result_path,test_acc_list,test_rmse_list,test_mae_list,test_avg_loglike_list,
                 test_map_at_5_list,test_map_at_10_list, current_time, args):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    ##########################################################

    model_info = result_path + 'model_params.txt'

    model_params_dict = {
        'data_name': args.data_name,
        'test_fold': args.test_fold,
        'model_name': args.model_name,
    }

    if(args.model_name == 'CDAE'):
        model_params_dict.update({
            'train_epoch': args.train_epoch,
            'lr': args.lr,
            'optimizer_method': args.optimizer_method,
            'keep_prob': args.keep_prob,
            'grad_clip': args.grad_clip,
            'batch_normalization': args.batch_normalization,
            'hidden_neuron': args.hidden_neuron,
            'corruption_level': args.corruption_level,
            'lambda_value': args.lambda_value,
            'f_act': args.f_act,
            'g_act': args.g_act,
            'encoder_method': args.encoder_method
        })

    elif(args.model_name == 'SLIMElasticNet'):
        model_params_dict.update({
            'l1_reg': args.l1_reg,
            'l2_reg': args.l2_reg,
            'learner': args.learner
        })

    with open(model_info, 'a') as f:
        for k,v in model_params_dict.items():
            s = str(k) + "   " + str(v) + "\n"
            f.write(s)

    # with open(model_info, 'a') as f:
    #     for arg in vars(args): # parser.parse_args() variables
    #         s = str(arg) + "   " + str(getattr(args, arg)) + "\n"
    #         f.write(s)
    
    ##########################################################
    
    # evaluation metrics (on text_fold) by epoch
    test_record = result_path + 'test_record.txt'

    test_record_df = pd.DataFrame({
        'RMSE': test_rmse_list,
        'MAE': test_mae_list,
        'ACC': test_acc_list,
        'NALL': test_avg_loglike_list,
        'MAP@5': test_map_at_5_list,
        'MAP@10': test_map_at_10_list
    })

    test_record_df.index.name = 'Epoch'

    trfile = open(test_record, 'a')
    trfile.write(test_record_df.to_string())
    trfile.close()

    ##########################################################

    plots_dir = result_path + 'plots/'
    os.makedirs(plots_dir)

    print("Plotting error metrics.....")

    plt.plot(test_acc_list, label="ACC")
    plt.plot(test_rmse_list, label="RMSE")  
    plt.plot(test_mae_list, label="MAE")  
    plt.plot(test_avg_loglike_list, label="NALL")  

    plt.xlabel('Epochs')
    plt.ylabel('Test evaluation metrics')
    plt.legend()
    plt.savefig(plots_dir + "test_evaluation.png")
    plt.clf()

    print("Plotting ranking metrics.....")

    plt.plot(test_map_at_5_list, label="MAP@5")
    plt.plot(test_map_at_10_list, label="MAP@10")   

    plt.xlabel('Epochs')
    plt.ylabel('MAP')
    plt.legend()
    plt.savefig(plots_dir + "test_top_k_evaluation.png")
    plt.clf()

def variable_save(result_path,model_name,train_var_list1,train_var_list2,Estimated_R,test_v_ud,mask_test_v_ud):
    for var in train_var_list1:
        var_value = var.eval()
        var_name = ((var.name).split('/'))[1]
        var_name = (var_name.split(':'))[0]
        np.savetxt(result_path + var_name , var_value)

    for var in train_var_list2:
        if model_name == "DIPEN_with_VAE":
            var_value = var.eval()
            var_name = (var.name.split(':'))[0]
            print (var_name)
            var_name = var_name.replace("/","_")
            #var_name = ((var.name).split('/'))[2]
            #var_name = (var_name.split(':'))[0]
            print (var.name)
            print (var_name)
            print ("================================")
            np.savetxt(result_path + var_name, var_value)
        else:
            var_value = var.eval()
            var_name = ((var.name).split('/'))[1]
            var_name = (var_name.split(':'))[0]
            np.savetxt(result_path + var_name , var_value)

    Estimated_R = np.where(Estimated_R<0.5,0,1)
    Error_list = np.nonzero( (Estimated_R - test_v_ud) * mask_test_v_ud )
    user_error_list = Error_list[0]
    item_error_list = Error_list[1]
    np.savetxt(result_path+"Estimated_R",Estimated_R)
    np.savetxt(result_path+"test_v_ud",test_v_ud)
    np.savetxt(result_path+"mask_test_v_ud",mask_test_v_ud)
    np.savetxt(result_path + "user_error_list", user_error_list)
    np.savetxt(result_path + "item_error_list", item_error_list)

def SDAE_calculate(model_name,X_c, layer_structure, W, b, batch_normalization, f_act,g_act, model_keep_prob,V_u=None):
    hidden_value = X_c
    for itr1 in range(len(layer_structure) - 1):
        ''' Encoder '''
        if itr1 <= int(len(layer_structure) / 2) - 1:
            if (itr1 == 0) and (model_name == "CDAE"):
                ''' V_u '''
                before_activation = tf.add(tf.add(tf.matmul(hidden_value, W[itr1]),V_u), b[itr1])
            else:
                before_activation = tf.add(tf.matmul(hidden_value, W[itr1]), b[itr1])
            if batch_normalization == "True":
                before_activation = batch_norm(before_activation)
            hidden_value = f_act(before_activation)

            ''' Decoder '''
        elif itr1 > int(len(layer_structure) / 2) - 1:
            before_activation = tf.add(tf.matmul(hidden_value, W[itr1]), b[itr1])
            if batch_normalization == "True":
                before_activation = batch_norm(before_activation)
            hidden_value = g_act(before_activation)
        
        # fraction of the activations coming from g_act that will be disactivated (dropped)
        if itr1 < len(layer_structure) - 2: # add dropout except final layer. 
            # hidden_value = tf.nn.dropout(hidden_value, 1 - (model_keep_prob))
            hidden_value = tf.nn.dropout(hidden_value, rate = (1 - model_keep_prob))
        
        if itr1 == int(len(layer_structure) / 2) - 1:
            Encoded_X = hidden_value

    sdae_output = hidden_value

    return Encoded_X, sdae_output

def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(input_tensor=tf.square(tensor)))

def softmax(w, t = 1.0):
    npa = np.array
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist