import tensorflow as tf
import time
import argparse

from utils.data_preprocessor import *
from utils.data_manager import *

from CDAE import CDAE
from DAE import DAE
from recommenders.SLIM import SLIM

# Ignore warning TODO: check warnings
import warnings
warnings.filterwarnings('ignore')

current_time = time.time()


# ------------------------------------------------------------------ #
                    ##### Model setup #####
# ------------------------------------------------------------------ #

#TODO: use easydict

parser = argparse.ArgumentParser(description='Collaborative Denoising Autoencoder')
parser.add_argument('--model_name', choices=['CDAE', 'SLIMElasticNet'], default='SLIMElasticNet')
parser.add_argument('--random_seed', type=int, default=1000)

# dataset name
parser.add_argument('--data_name', choices=['politic_old','politic_new','movielens_10m'], default='politic_new')

# train/test fold for training
# for politic_old and politic_new: 0,1,2,3,4. In the case of movielens_10m 1,2,3,4,5
parser.add_argument('--test_fold', type=int, default=1) # TODO: iterate all folds at once 

# training epochs
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--display_step', type=int, default=1)

# learning rate
parser.add_argument('--lr', type=float, default=1e-3) 

# gradient-based optimization algorithms
parser.add_argument('--optimizer_method', choices=['Adam','Adadelta','Adagrad','RMSProp', \
                    'GradientDescent','Momentum'],default='Adam')

# dropout: keep_prob to specify the fraction of the input units to keep while training 
# NOTE: setting keep_prob to exactly 1.0, this means the probability of dropping any node becomes 0
parser.add_argument('--keep_prob', type=float, default=0.0) 

# gradient clipping: prevent exploding gradients
parser.add_argument('--grad_clip', choices=['True', 'False'], default='True')

# normalize activations of the previous layer at each batch
parser.add_argument('--batch_normalization', choices=['True','False'], default = 'False')

# number of latent dimensions (K) 
parser.add_argument('--hidden_neuron', type=int, default=50)

# input corruption 
parser.add_argument('--corruption_level', type=float, default=0.3)

# regularization rate
parser.add_argument('--lambda_value', type=float, default=0.001)

# activation functions
parser.add_argument('--f_act', choices=['Sigmoid','Relu','Elu','Tanh','Identity'], default = 'Sigmoid')
parser.add_argument('--g_act', choices=['Sigmoid','Relu','Elu','Tanh','Identity'], default = 'Sigmoid')

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
# and Bill IDs

if data_name == 'politic_new': # Politic2016
    num_users = 1537 
    num_items = 7975
    num_total_ratings = 2999844

elif data_name == 'politic_old': # Politic2013
    num_users = 1540
    num_items = 7162
    num_total_ratings = 2779703

    # df2 = pd.read_csv("../data/politic_old/Test_ratings_fold_0" ,sep = "\t")

    # print("df2\n", df2.count()) 
    ## df2
    #  835     555939
    # 4554    555939
    # 1       555939
    # dtype: int64

    # num_users = df2[2].nunique()
    # num_total_ratings =  df2.shape[0]

    # print("num_users", num_users)
    # print("num_total_ratings", num_total_ratings)

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


date = "0203"
result_path = '../results/' + data_name + '/' + model_name + '/' + str(test_fold) +  '/' + str(current_time)+"/"

# read ratings file and train/test split
R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings \
    = read_rating(path, data_name, num_users, num_items, num_total_ratings, a, b, test_fold,random_seed)


# model params
# print("model arguments:\n", args, end="\n")
model_string = "\nType of Model: {}, \nDataset: {}, \nTest fold: {}, \nHidden neurons: {} \n".format(
    model_name,
    data_name,
    test_fold,
    hidden_neuron
)
print(model_string)

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

        model = CDAE(sess,args,layer_structure,n_layer,pre_W,pre_b,keep_prob,batch_normalization,current_time,
                    num_users,num_items,hidden_neuron,f_act,g_act,
                    R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                    train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                    decay_epoch_step,lambda_value,
                    result_path,date,data_name,model_name,test_fold,corruption_level) 
                    
        # train and test the model
        model.run()

    
    # Machine learning approach to Item-based CF
    # Sparse LInear Method

    elif model_name == "SLIMElasticNet":      
        # TODO: apply_hyperparams_tuning?

        model = SLIM(l1_reg=args.l1_reg,l2_reg=args.l2_reg,model=args.learner)
        print(model)
        model.fit(R)

    else:
        raise NotImplementedError("ERROR")
