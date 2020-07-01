import tensorflow as tf
import time
import argparse

from utils.data_preprocessor import *
from utils.data_manager import *

from CDAE import CDAE
from DAE import DAE
from recommenders.SLIM import SLIM

# Ignore warning 
# TODO: check warnings
import warnings
warnings.filterwarnings('ignore')

# current_time = time.time()
named_tuple = time.localtime() # get struct_time
current_time = time.strftime("%m%d%Y_%H:%M:%S", named_tuple)



# ------------------------------------------------------------------ #
                    ##### Model setup #####
# ------------------------------------------------------------------ #

#TODO: use easydict

parser = argparse.ArgumentParser(description='Collaborative Denoising Autoencoder')
parser.add_argument('--model_name', choices=['CDAE', 'SLIMElasticNet'], default='SLIMElasticNet')
parser.add_argument('--random_seed', type=int, default=1000)

# dataset name
parser.add_argument('--data_name', choices=['politic_old', 'politic_new', 'movielens_10m'], default='politic_old')

# train/test fold for training
# for politic_old and politic_new: 0,1,2,3,4. In the case of movielens_10m 1,2,3,4,5
parser.add_argument('--test_fold', type=int, default=0) # TODO: iterate all folds at once 

# training epochs
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--display_step', type=int, default=1)

# learning rate
parser.add_argument('--lr', type=float, default=1e-3) 

# gradient-based optimization algorithms
parser.add_argument('--optimizer_method', choices=['Adam','Adadelta','Adagrad','RMSProp', \
                    'GradientDescent','Momentum'], default='Adam')

# dropout: keep_prob to specify the fraction of the input units to keep while training 
# setting keep_prob to exactly 1.0, this means the probability of dropping any node becomes 0
parser.add_argument('--keep_prob', type=float, default=1.0) 

# gradient clipping: prevent exploding gradients
parser.add_argument('--grad_clip', choices=['True', 'False'], default='True')

# normalize activations of the previous layer at each batch
parser.add_argument('--batch_normalization', choices=['True','False'], default = 'False')

# number of latent dimensions (K) 
parser.add_argument('--hidden_neuron', type=int, default=50)

# input corruption 
parser.add_argument('--corruption_level', type=float, default=0.0)

# regularization rate
parser.add_argument('--lambda_value', type=float, default=0.001)

# activation functions
parser.add_argument('--f_act', choices=['Sigmoid','Relu','Elu','Tanh','Identity'], default = 'Identity')
parser.add_argument('--g_act', choices=['Sigmoid','Relu','Elu','Tanh','Identity'], default = 'Identity')

# for reading ratings file
parser.add_argument('--a', type=float, default=1)
parser.add_argument('--b', type=float, default=0)

# Autoencoder types:
'''SDAE: Stacked Denoising Autoencoder
VAE: Variational Autoencoder '''
parser.add_argument('--encoder_method', choices=['SDAE','VAE'], default='SDAE')

# SLIM parameters

# l1 regularization constant
parser.add_argument('--l1_reg', type=float, default = 0.001)
# l2 regularization constant
parser.add_argument('--l2_reg', type=float, default = 0.0001)
# underlying learner for SLIM learner
parser.add_argument('--learner', choices=['sgd','elasticnet','fs_sgd'], default = 'elasticnet')

args = parser.parse_args()

random_seed = args.random_seed
tf.compat.v1.reset_default_graph()
np.random.seed(random_seed)
tf.compat.v1.set_random_seed(random_seed)


# ------------------------------------------------------------------ #
                    ##### Data managing #####
# ------------------------------------------------------------------ #


model_name = args.model_name

# Data directory
data_name = args.data_name
data_base_dir = "../data/"
path = data_base_dir + "%s" % data_name + "/"

''' 
Attributes of Politic2013 and Politic2016 datasets
Num of legislators (|U|) = num_users
Num of bills (|D|) = num_items
Num of votings (|D|) = num_total_ratings
'''

print("Loading", data_name, "data ... ", end="\n")

# politic_new and politic_old 
# User IDs are in ranges from 1 to 1537-1

if data_name == 'politic_new': # Politic2016
    num_users = 1537 
    num_items = 7975
    num_total_ratings = 2999844

elif data_name == 'politic_old': # Politic2013
    num_users = 1540
    num_items = 7162
    num_total_ratings = 2779703

elif data_name == 'movielens_10m': 
    
    data_path = "../data/movielens_10m/"
   
    if not os.path.isdir(data_path): # run just once
        ratings_df = movielens_10m_prepare_data(data_name)
    else: 
        ratings_df = load_movielens_10m_data()

    # Data exploration (summary statitics) 
    num_users, num_items, num_total_ratings = movielens_10m_statistics(ratings_df)

else:
    raise NotImplementedError("ERROR")


# ------------------------------------------------------------------ #
                    ##### Training config #####
# ------------------------------------------------------------------ #
 
a = args.a
b = args.b

test_fold = args.test_fold
hidden_neuron = args.hidden_neuron

keep_prob = args.keep_prob
batch_normalization = args.batch_normalization

batch_size = 256
lr = args.lr
train_epoch = args.train_epoch
optimizer_method = args.optimizer_method
display_step = args.display_step

# learning rate schedules: adapt lr based on number of epochs
decay_epoch_step = 10000 
decay_rate = 0.96
grad_clip = args.grad_clip

if args.f_act == "Sigmoid":
    f_act = tf.nn.sigmoid
elif args.f_act == "Relu":
    f_act = tf.nn.relu
elif args.f_act == "Tanh":
    f_act = tf.nn.tanh
elif args.f_act == "Identity":
    f_act = tf.identity
elif args.f_act == "Elu":
    f_act = tf.nn.elu
else:
    raise NotImplementedError("ERROR")

if args.g_act == "Sigmoid":
    g_act = tf.nn.sigmoid
elif args.g_act == "Relu":
    g_act = tf.nn.relu
elif args.g_act == "Tanh":
    g_act = tf.nn.tanh
elif args.g_act == "Identity":
    g_act = tf.identity
elif args.g_act == "Elu":
    g_act = tf.nn.elu
else:
    raise NotImplementedError("ERROR")


result_path = '../results/' + data_name + '/' + model_name + '/' + str(test_fold) +  '/' + str(current_time)+"/"

# read ratings file and train/test split
R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set \
    = read_rating(path, data_name, num_users, num_items, num_total_ratings, a, b, test_fold,random_seed)

# model params
# print("model arguments:\n", args, end="\n")
model_string = "\nType of model: {}, \nDataset: {}, \nTest fold: {}, \nHidden neurons: {} \n".format(
        model_name,
        data_name,
        test_fold,
        hidden_neuron
    )
print(model_string)

# Matrix Compressed Sparse Row format
# -----------------------------------
import scipy.sparse as sps
def csr_sparse_matrix(data, row, col, shape=None):
    csr_matrix = sps.coo_matrix((data, (row, col)), shape=shape)
    csr_matrix = csr_matrix.tocsr()

    return csr_matrix


# Random holdout split: take interactions randomly
# and do not care about which users were involved in that interaction
def split_train_validation_random_holdout(URM, train_split):
    URM = sps.csr_matrix(URM) 

    number_interactions = URM.nnz  # number of nonzero values
    URM = URM.tocoo()  # Coordinate list matrix (COO)
    shape = URM.shape

    #  URM.row: user_list, URM.col: item_list, URM.data: rating_list

    # Sampling strategy: take random samples of data using a boolean mask
    train_mask = np.random.choice(
        [True, False],
        number_interactions,
        p=[train_split, 1 - train_split])  # train_perc for True, 1-train_perc for False

    URM_train = csr_sparse_matrix(URM.data[train_mask],
                                  URM.row[train_mask],
                                  URM.col[train_mask],
                                  shape=shape)

    test_mask = np.logical_not(train_mask)  # remaining samples
    URM_test = csr_sparse_matrix(URM.data[test_mask],
                                 URM.row[test_mask],
                                 URM.col[test_mask],
                                 shape=shape)

    return URM_train, URM_test


# Hyper-parameters tuning: Bayesian Optimization Approach
def hyperparams_tuning(recommender_class, URM_train, URM_validation, URM_test):

    from utils.Evaluation.Evaluator import EvaluatorHoldout

    # Step 1: Import the evaluator objects
    print("Evaluator objects ... ")
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10])
    # evaluator_validation_earlystopping = EvaluatorHoldout(URM_train, cutoff_list=[5, 10])
    
    # Step 2: Create BayesianSearch object
    print("BayesianSearch objects ... ")

    from utils.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
    
    parameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation,
                                         evaluator_test=evaluator_test)
    
    # Step 3: Define parameters range   
    print("Parameters range ...") 
    from skopt.space import Real, Integer, Categorical
    from utils.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
    hyperparameters_range_dictionary["l1_ratio"] = Real(low=1e-5, high=1.0, prior='log-uniform')
    hyperparameters_range_dictionary["alpha"] = Real(low=1e-3, high=1.0, prior='uniform')

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )
    
    output_folder_path = "result_experiments/"
    
    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # # Step 4: run

    n_cases = 2
    metric_to_optimize = "MAP"
    output_file_name_root = "{}_metadata.zip".format(recommender_class.RECOMMENDER_NAME)

    best_parameters = parameterSearch.search(recommender_input_args, # the function to minimize
                                                parameter_search_space=hyperparameters_range_dictionary, # the bounds on each dimension of x
                                                n_cases=n_cases,   # the number of evaluations of f
                                                n_random_starts=1, # the number of random initialization points
                                                #n_random_starts = int(n_cases/3),
                                                save_model="no",
                                                output_folder_path=output_folder_path,
                                                output_file_name_root=output_file_name_root,
                                                metric_to_optimize=metric_to_optimize
                                            )
                                
    print("best_parameters", best_parameters)

    # # Step 5: return best_parameters
    # # from utils.DataIO import DataIO
    # # data_loader = DataIO(folder_path=output_folder_path)
    # # search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")
    # # print("search_metadata", search_metadata)
    
    # best_parameters = search_metadata["hyperparameters_best"]

    return best_parameters


# Best hyperparameters found by tuning (by dataset)
# -------------------------------------------------

SLIMElasticNet_best_parameters_list = {
    'politic_old': {'topK': 1000, 'l1_ratio': 1e-05, 'alpha': 0.001},
}

''' Launch the evaluation graph in a session '''
with tf.compat.v1.Session() as sess:
   
    if model_name == "CDAE":
        lambda_value = args.lambda_value
        corruption_level = args.corruption_level

        #layer_structure = [num_items, hidden_neuron, num_items]
        layer_structure = [num_items, 512, 128, hidden_neuron, 128, 512, num_items]
        n_layer = len(layer_structure)
        
        # Initialize weights
        pre_W = dict()
        pre_b = dict()
        
        for itr in range(n_layer - 1):
            initial_DAE = DAE(layer_structure[itr], layer_structure[itr + 1], num_items, itr, "sigmoid")

            # get initial weights using do_not_pretrain
            pre_W[itr], pre_b[itr] = initial_DAE.do_not_pretrain()

            # get initial weights using do_pretrain??

        cdae_model = CDAE(sess,args,layer_structure,n_layer,pre_W,pre_b,keep_prob,batch_normalization,current_time,
                    num_users,num_items,hidden_neuron,f_act,g_act,
                    R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                    train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                    decay_epoch_step,lambda_value,
                    user_train_set, item_train_set, user_test_set, item_test_set,
                    result_path,data_name,model_name,test_fold,corruption_level) 
                    
        # train and test the model
        cdae_model.run()

    
    # Machine learning approach to Item-based CF
    # Sparse LInear Method

    elif model_name == "SLIMElasticNet":      
        
        # Model specific cross-validation
        # Elastic Net model with iterative fitting along a regularization path

        print("Splitting ... ")
        from sklearn.model_selection import train_test_split

        # Split dataset into train, validation and test with 0.6, 0.2, 0.2
        # URM_train, URM_test = train_test_split(R, train_size = 0.8, test_size = 0.2, random_state = args.random_seed)
        # URM_train, URM_validation = train_test_split(train_R, train_size = 0.8, test_size = 0.2, 
        #                                                 random_state = args.random_seed)

        URM_train, URM_test = split_train_validation_random_holdout(R, train_split=0.8)
        URM_train, URM_validation = split_train_validation_random_holdout(URM_train, train_split=0.9)

        apply_hyperparams_tuning = True

        from recommenders.Recommenders.SLIMElasticNetRecommender import SLIMElasticNetRecommender

        SLIMElasticNet = SLIMElasticNetRecommender(URM_train)
        
        if apply_hyperparams_tuning:
            best_parameters_SLIMElasticNet = hyperparams_tuning(SLIMElasticNetRecommender, 
                                                                    URM_train, URM_validation, URM_test)
        else:
            best_parameters_SLIMElasticNet = SLIMElasticNet_best_parameters_list[args.data_name]

        SLIMElasticNet.fit(**best_parameters_SLIMElasticNet)

        exit(0)


        # model = SLIM(l1_reg=args.l1_reg,l2_reg=args.l2_reg,model=args.learner)
        # print(model)
        # model.fit(train_R)

        # self.test_model(epoch_itr)
        # evaluate_algorithm(URM_test, recommender)
        
        # make_records(result_path,test_acc_list,test_rmse_list,test_mae_list,test_avg_loglike_list,
        #               test_map_at_5_list,test_map_at_10_list, current_time, args)

    else:
        raise NotImplementedError("ERROR")

