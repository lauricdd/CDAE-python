import tensorflow as tf
import time
import argparse

from utils.data_preprocessor import *
from utils.data_manager import *
from utils.Evaluation.Evaluator import EvaluatorHoldout
from utils.utils import save_dictionary

from recommenders.Recommenders.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from recommenders.CDAE import CDAE
from recommenders.DAE import DAE

# Ignore warning 
# TODO: check warnings
import warnings
warnings.filterwarnings('ignore')

                        
#TODO: use easydict
parser = argparse.ArgumentParser(description='Collaborative Denoising Autoencoder')

# model
parser.add_argument('--model_name', choices=['CDAE', 'SLIMElasticNet'], default='SLIMElasticNet')
parser.add_argument('--random_seed', type=int, default=1000)

# dataset name
parser.add_argument('--data_name', choices=['politic_old', 'politic_new', 'movielens_10m', 'netflix_prize'], 
                        default='netflix_prize')


######################################################################
# CDAE parameters
######################################################################

# train/test fold for training
# for politic_old and politic_new: 0,1,2,3,4. 
# In the case of movielens_10m and netflix_prize: 1,2,3,4,5
parser.add_argument('--test_fold', type=int, default=1) # TODO: iterate all folds at once 

# training epochs
parser.add_argument('--train_epoch', type=int, default=50)
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
'''SDAE: Stacked Denoising Autoencoder VAE: Variational Autoencoder '''
parser.add_argument('--encoder_method', choices=['SDAE','VAE'], default='SDAE')


######################################################################
# SLIM parameters
######################################################################

parser.add_argument('--apply_hyperparams_tuning', choices=['True','False'], default='True')
# TODO: not working for netflix_prize 
# EvaluatorHoldout: WARNING: No users had a sufficient number of relevant items

# best hyperparamas config evaluated with evaluator_test. 
SLIMElasticNet_best_parameters_list = {
    # fold 0
    'politic_old': {'topK': 1000, 'l1_ratio': 1e-05, 'alpha': 0.001}, 
    
    # using random splitting on the entire dataset (using R matrix)
    'movielens_10m': {'topK': 533, 'l1_ratio': 0.025062993365157635, 'alpha': 0.18500803626703258},

    'netflix_prize': {'topK': 209, 'l1_ratio': 0.011269764260753398, 'alpha': 0.011248017606952841}
}


args = parser.parse_args()

random_seed = args.random_seed
tf.compat.v1.reset_default_graph()
np.random.seed(random_seed)
tf.compat.v1.set_random_seed(random_seed)


model_name = args.model_name
data_name = args.data_name
test_fold = args.test_fold

model_string = "\nType of model: {} \nDataset: {} \nTest fold: {}".format(
    model_name,
    data_name,
    test_fold
)

print(model_string)
print("="*100)

# Data directory
data_base_dir = "../data/"
path = data_base_dir + "%s" % data_name + "/"

''' 
    Attributes of Politic2013(politic_old) and Politic2016(politic_new) datasets
    Num of legislators (|U|) = num_users
    Num of bills (|D|) = num_items
    Num of votings (|D|) = num_total_ratings

    User IDs are in ranges from 1 to 1537-1
'''


if data_name == 'politic_new': # Politic2016
    num_users = 1537 
    num_items = 7975
    num_total_ratings = 2999844

elif data_name == 'politic_old': # Politic2013
    num_users = 1540
    num_items = 7162
    num_total_ratings = 2779703

elif data_name == 'movielens_10m' or data_name == 'netflix_prize': 

    if data_name == 'movielens_10m':
        ''' 
            load data from MovieLens 10M Dataset
            http://grouplens.org/datasets/movielens/ 
        '''

        DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
        DATASET_SUBFOLDER = "../data/movielens_10m/"
        DATASET_FILE_NAME = "movielens_10m.zip"  
        DATASET_UNZIPPED_FOLDER = "ml-10M100K/"

    elif data_name == 'netflix_prize': 
        '''
            load data from netflix prize dataset using kaggle
        '''

        DATASET_URL = None
        DATASET_SUBFOLDER = "../data/netflix_prize/"
        DATASET_FILE_NAME = None
        DATASET_UNZIPPED_FOLDER = None
    
    if not os.path.isdir(DATASET_SUBFOLDER): # run just first time
        ratings_df = prepare_data(data_name, DATASET_URL, DATASET_SUBFOLDER, DATASET_FILE_NAME, DATASET_UNZIPPED_FOLDER)
    else: 
        ratings_df = load_data(DATASET_SUBFOLDER)

    # data exploration (summary statitics) 
    print("{} statistics AFTER preprocessing ... ".format(data_name))
    num_users, num_items, num_total_ratings = dataset_statistics(data_name, ratings_df)

else:
    raise NotImplementedError("ERROR")


named_tuple = time.localtime() # get struct_time
current_time = time.strftime("%m%d%Y_%H:%M:%S", named_tuple)

a = args.a
b = args.b

result_path = '../results/' + data_name + '/' + model_name + '/' + str(test_fold) +  '/' + str(current_time) + '/'

# read ratings file and train/test split
R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings,\
user_train_set, item_train_set, user_test_set, item_test_set \
    = read_rating(path, data_name, num_users, num_items, num_total_ratings, a, b, test_fold, random_seed)


######################################################################

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
    
    output_folder_path = "../results/result_experiments/"
    
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

    # Step 5: return best_parameters
    # from utils.DataIO import DataIO
    # data_loader = DataIO(folder_path=output_folder_path)
    # search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")
    # print("search_metadata", search_metadata)
    
    # best_parameters = search_metadata["hyperparameters_best"]

    return best_parameters

######################################################################

from sklearn.model_selection import train_test_split

def data_split(R, train_frac, random_state=None):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    URM_train, URM_tmp = train_test_split(R, train_size=train_frac, random_state=random_state)

    URM_validation, URM_test = train_test_split(URM_tmp, train_size=0.5, random_state=random_state)

    return URM_train, URM_validation, URM_test



# Launch the evaluation graph in a session 
with tf.compat.v1.Session() as sess:
   
    if model_name == "CDAE":
        
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

            # TODO: get initial weights using do_pretrain??

        CDAE = CDAE(sess,args,layer_structure,n_layer,pre_W,pre_b,keep_prob,batch_normalization,current_time,
                    num_users,num_items,hidden_neuron,f_act,g_act,
                    R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                    train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                    decay_epoch_step,lambda_value,
                    user_train_set, item_train_set, user_test_set, item_test_set,
                    result_path,data_name,model_name,test_fold,corruption_level) 

        # train and evaluate the model
        CDAE.run()


    # Sparse LInear Method: Machine learning approach to Item-based CF
    elif model_name == "SLIMElasticNet":     
        
        # Split dataset into train and test
        # Holdout data: for each user, randomly hold 20% of the ratings in the test set
        print("Splitting dataset ... ")
        # URM_train, URM_test = split_train_validation_random_holdout(R, train_split=0.8)

        # from sklearn.model_selection import train_test_split
        # Split dataset into train, validation and test with 0.8, 0.1, 0.1
        # URM_train, URM_test = train_test_split(R, train_size = 0.8, test_size = 0.2, random_state = args.random_seed)

        URM_train, URM_validation, URM_test = data_split(R, train_frac=0.8, random_state=args.random_seed)

        import scipy.sparse as sps
        # convert to CSR format 
        URM_train = URM_train.tocsr()
        URM_validation = URM_validation.tocsr()
        URM_test = URM_test.tocsr()

        print("URM_train", type(URM_train))
        print("URM_validation", type(URM_validation))
        print("URM_test", type(URM_test))
        exit(0)

        # SLIM model
        SLIMElasticNet = SLIMElasticNetRecommender(URM_train)
        
        # hyperparameters tuning
        # URM_train, URM_validation = split_train_validation_random_holdout(URM_train, train_split=0.9)
        # URM_train, URM_validation = train_test_split(URM_train, train_size = 0.9, test_size = 0.1, 
        #                                             random_state = args.random_seed)

        if args.apply_hyperparams_tuning == "True":
            apply_hyperparams_tuning = True
        else:
            apply_hyperparams_tuning = False

        if apply_hyperparams_tuning:
            best_parameters_SLIMElasticNet = hyperparams_tuning(SLIMElasticNetRecommender, 
                                                                    URM_train, URM_validation, URM_test)
        else:
            best_parameters_SLIMElasticNet = SLIMElasticNet_best_parameters_list[data_name]
        
        print("Fitting SLIMElasticNet model using best tuned parameters", best_parameters_SLIMElasticNet)

        # train the model
        SLIMElasticNet.fit(**best_parameters_SLIMElasticNet)
        
        # evaluate the model
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10])
        result_dict, _ = evaluator_test.evaluateRecommender(SLIMElasticNet)
        
        print("{} result_dict MAP@5 {}".format(SLIMElasticNet.RECOMMENDER_NAME, result_dict[5]["MAP"]))    
        print("{} result_dict MAP@10 {}".format(SLIMElasticNet.RECOMMENDER_NAME, result_dict[10]["MAP"]))

        result_path = '../results/' + data_name + '/' + model_name + '/' + str(current_time) + '/'
        save_dictionary(result_path, best_parameters_SLIMElasticNet, result_dict, args)

    else:
        raise NotImplementedError("ERROR")

