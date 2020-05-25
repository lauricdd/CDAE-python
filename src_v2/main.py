from utils.data_preprocessor import *
from utils.data_manager import *
from CDAE import CDAE
from DAE import DAE
import tensorflow as tf
import time
import argparse


# Ignore warning TODO: check warnings
import warnings
warnings.filterwarnings('ignore')

current_time = time.time()


''' ==============================================================
                        Experiment setup
============================================================== '''

parser = argparse.ArgumentParser(description='Collaborative Denoising Autoencoder')
parser.add_argument('--model_name', choices=['CDAE'], default='CDAE')
parser.add_argument('--random_seed', type=int, default=1000)

# dataset name
parser.add_argument('--data_name', choices=['politic_old','politic_new','movielens_10m'], default='movielens_10m')

# train/test fold for training
parser.add_argument('--test_fold', type=int, default=0)

# training epochs
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--display_step', type=int, default=1)

# learning rate
parser.add_argument('--lr', type=float, default=1e-3) 

# gradient-based optimization algorithms
parser.add_argument('--optimizer_method', choices=['Adam','Adadelta','Adagrad','RMSProp', \
                    'GradientDescent','Momentum'],default='Adam')


# In tensorflow 2.0
# ValueError: rate must be a scalar tensor or a float in the range [0, 1), got 1
# 1.0 / (1 - rate) 

# used to control the dropout rate when training 
# parser.add_argument('--keep_prob', type=float, default=1.0) # tf 1.0
######Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`. 
parser.add_argument('--keep_prob', type=float, default=0.3)

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
parser.add_argument('--f_act', choices=['Sigmoid','Relu','Elu','Tanh',"Identity"], default = 'Sigmoid')
parser.add_argument('--g_act', choices=['Sigmoid','Relu','Elu','Tanh',"Identity"], default = 'Sigmoid')

# for reading ratings ???
parser.add_argument('--a', type=float, default=1)
parser.add_argument('--b', type=float, default=0)

# Autoencoder types:
'''SDAE: Stacked Denoising Autoencoder
VAE: Variational Autoencoder '''
parser.add_argument('--encoder_method', choices=['SDAE','VAE'],default='SDAE')

args = parser.parse_args()

random_seed = args.random_seed
tf.compat.v1.reset_default_graph()
np.random.seed(random_seed)
tf.compat.v1.set_random_seed(random_seed)


''' ==============================================================
                        Data attributes
============================================================== '''

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

if data_name == 'politic_new': # Politic2016
    num_users = 1537 
    num_items = 7975
    num_total_ratings = 2999844

elif data_name == 'politic_old': # Politic2013
    num_users = 1540
    num_items = 7162
    num_total_ratings = 2779703

elif data_name == 'movielens_10m': 
    ratings_df = movielens_load_data(data_name)

    num_users = ratings_df["user_id"].nunique() # count distinct values
    num_items = ratings_df["movie_id"].nunique() 
    num_total_ratings =  ratings_df.shape[0]

    print("num_users", num_users)
    print("num_items", num_items)
    print("num_total_ratings", num_total_ratings)

    print(ratings_df)

else:
    raise NotImplementedError("ERROR")

exit(-1)

# from sklearn.model_selection import KFold 
# kf = KFold(n_splits=5, random_state=1000, shuffle=True) # random seed when shuffle=True
# kf.get_n_splits(ratings)

# print(kf)

# for train_index, test_index in kf.split(ratings):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     # ratings_train, ratings_test = ratings[train_index], ratings[test_index]
#     # traing ra




''' ==============================================================
                        Training config
============================================================== '''
 
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
R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set \
    = read_rating(path, data_name, num_users, num_items, num_total_ratings, a, b, test_fold,random_seed)

# X_dw = read_bill_term(path,data_name,num_items,num_voca)


''' ==============================================================
                        Model config
============================================================== '''

print ("Type of Model : %s" %model_name)
print ("Type of Data : %s" %data_name)
print ("# of User : %d" %num_users)
print ("# of Item : %d" %num_items)
print ("Test Fold : %d" %test_fold)
print ("Random seed : %d" %random_seed)
print ("Hidden neuron : %d" %hidden_neuron)


with tf.compat.v1.Session() as sess:
    if model_name == "CDAE":
        lambda_value = args.lambda_value
        corruption_level = args.corruption_level

        #layer_structure = [num_items, hidden_neuron, num_items]
        layer_structure = [num_items, 512, 128, hidden_neuron, 128, 512, num_items]
        n_layer = len(layer_structure)

        pre_W = dict()
        pre_b = dict()
        
        for itr in range(n_layer - 1):
            initial_DAE = DAE(layer_structure[itr], layer_structure[itr + 1], num_items, itr, "sigmoid")
            # initial_DAE = DAE(layer_structure[itr], layer_structure[itr + 1], num_items, num_voca, itr, "sigmoid")
            
            # get initial weights using do_not_pretrain
            pre_W[itr], pre_b[itr] = initial_DAE.do_not_pretrain()

            # get initial weights using do_pretrain

        model = CDAE(sess,args,layer_structure,n_layer,pre_W,pre_b,keep_prob,batch_normalization,current_time,
                    num_users,num_items,hidden_neuron,f_act,g_act,
                    R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                    train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                    decay_epoch_step,lambda_value,
                    user_train_set, item_train_set, user_test_set, item_test_set,
                    result_path,date,data_name,model_name,test_fold,corruption_level)
    else:
        raise NotImplementedError("ERROR")

    # train and test the model
    model.run()


