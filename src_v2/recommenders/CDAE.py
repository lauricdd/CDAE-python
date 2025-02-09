import tensorflow as tf
import time
import numpy as np
# from utils.utils import *
from utils.utils import make_records, SDAE_calculate, evaluation, top_k_evaluation
from numpy import inf

class CDAE():
    def __init__(self,sess,args,layer_structure,n_layer,pre_W,pre_b,keep_prob,batch_normalization,current_time,
                    num_users,num_items,hidden_neuron,f_act,g_act,
                    R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                    train_epoch,batch_size, lr, optimizer_method, 
                    display_step, random_seed,
                    decay_epoch_step,lambda_value,
                    user_train_set, item_train_set, user_test_set, item_test_set,
                    result_path,data_name,model_name,test_fold,corruption_level):

        self.sess = sess
        self.args = args
        self.layer_structure = layer_structure
        self.n_layer = n_layer
        self.Weight = pre_W
        self.bias = pre_b
        self.keep_prob = keep_prob
        self.batch_normalization = batch_normalization

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_neuron = hidden_neuron

        self.current_time = current_time

        self.R = R # URM
        self.mask_R = mask_R
        self.C = C
        self.train_R = train_R  # URM_train
        self.train_mask_R = train_mask_R
        self.test_R = test_R    # URM_test
        self.test_mask_R = test_mask_R
        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings

        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.num_batch = int(self.num_users / float(self.batch_size)) + 1

        self.lr = lr
        self.optimizer_method = optimizer_method
        self.display_step = display_step
        self.random_seed = random_seed #TODO: is it random seed necessary?

        self.f_act = f_act
        self.g_act = g_act

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch

        self.lambda_value = lambda_value

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []
        self.test_mae_list = []
        self.test_acc_list = []
        self.test_avg_loglike_list = []

        self.test_map_at_5_list = []
        self.test_map_at_10_list = []

        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.user_test_set = user_test_set
        self.item_test_set = item_test_set

        self.result_path = result_path
        self.data_name = data_name

        self.model_name = model_name

        self.corruption_level = corruption_level

        self.earlystop_switch = False
        self.min_RMSE = 99999
        self.min_epoch = -99999
        self.patience = 0
        self.total_patience = 20

    def run(self):
        
        self.prepare_model()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        for epoch_itr in range(self.train_epoch):
            
            # early stopping
            if self.earlystop_switch:
                break
            else:
                self.train_model(epoch_itr)
                self.test_model(epoch_itr)

        make_records(self.result_path,self.test_acc_list,self.test_rmse_list,self.test_mae_list,self.test_avg_loglike_list,
                    self.test_map_at_5_list,self.test_map_at_10_list, self.current_time, self.args)

    def prepare_model(self):
        self.model_mask_corruption = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_items])
        self.input_R = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R")
        self.input_mask_R = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_mask_R")
        self.model_batch_data_idx = tf.compat.v1.placeholder(dtype=tf.int32)
        real_batch_size = tf.cast(tf.shape(input=self.input_R)[0], tf.int32)

        U = tf.cast(real_batch_size, tf.float32)
        corrupted_R = tf.multiply(self.model_mask_corruption, self.input_R)  ### Corrupted input
        corrupted_input_mask_R = tf.multiply(self.model_mask_corruption, self.input_mask_R)

        V = tf.compat.v1.get_variable(name="V", initializer=tf.random.truncated_normal(shape=[self.num_users, self.layer_structure[1]],
                                         mean=0, stddev=0.03),dtype=tf.float32)
        batch_V = tf.reshape(tf.gather(V, self.model_batch_data_idx), [real_batch_size, self.layer_structure[1]])

        Encoded_X, self.Decoder = SDAE_calculate(self.model_name,corrupted_R, self.layer_structure, self.Weight, self.bias, self.batch_normalization, self.f_act,self.g_act, self.keep_prob,batch_V)

        pre_cost1 = -1 * tf.multiply(corrupted_R, tf.math.log(self.Decoder)) - tf.multiply((1-corrupted_R) , tf.math.log(1-self.Decoder))
        pre_cost1 = tf.multiply(pre_cost1,corrupted_input_mask_R)
        cost1 = tf.reduce_sum(input_tensor=pre_cost1) / U

        pre_cost2 = tf.constant(0, dtype=tf.float32)
        for itr in range(len(self.Weight.keys())):
            pre_cost2 = tf.add(pre_cost2,
                               tf.add(tf.nn.l2_loss(self.Weight[itr]), tf.nn.l2_loss(self.bias[itr])))
        pre_cost2 = pre_cost2 + tf.nn.l2_loss(batch_V)
        cost2 = self.lambda_value * 0.5 * pre_cost2

        self.cost = cost1 + cost2

        if self.optimizer_method == "Adam":
            optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "Adadelta":
            optimizer = tf.compat.v1.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "Adagrad":
            # optimizer = tf.compat.v1.train.AdadeltaOptimizer(self.lr)
            optimizer = tf.compat.v1.train.AdagradOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.compat.v1.train.RMSPropOptimizer(self.lr)
        elif self.optimizer_method == "GradientDescent":
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.lr)
        elif self.optimizer_method == "Momentum":
            optimizer = tf.compat.v1.train.MomentumOptimizer(self.lr,0.9)
        else:
            raise ValueError("Optimizer Key ERROR")

        gvs = optimizer.compute_gradients(self.cost)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def train_model(self,itr):
        start_time = time.time()
        mask_corruption_np = np.random.binomial(1, 1 - self.corruption_level,
                                                (self.num_users, self.num_items))
        random_perm_doc_idx = np.random.permutation(self.num_users)

        batch_cost = 0
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i+1) * self.batch_size]

            _, Cost = self.sess.run(
                [self.optimizer, self.cost],
                feed_dict={self.model_mask_corruption: mask_corruption_np[batch_set_idx, :],
                           self.input_R: self.train_R[batch_set_idx, :],
                           self.input_mask_R: self.train_mask_R[batch_set_idx, :],
                           self.model_batch_data_idx: batch_set_idx})

            batch_cost = batch_cost + Cost
        self.train_cost_list.append(batch_cost)
        
        if itr % self.display_step == 0:
            print ("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(batch_cost),
                   "Elapsed time : %d sec" % (time.time() - start_time))

    def test_model(self,itr):
        start_time = time.time()
        mask_corruption_np = np.random.binomial(1, 1 - 0,
                                                (self.num_users, self.num_items))
        batch_set_idx = np.arange(self.num_users)

        # Use decoder (obtained by training) to estimate ratings
        Cost,Decoder = self.sess.run(
            [self.cost,self.Decoder],
            feed_dict={self.model_mask_corruption: mask_corruption_np,
                       self.input_R: self.test_R,
                       self.input_mask_R: self.test_mask_R,
                       self.model_batch_data_idx: batch_set_idx})

        self.test_cost_list.append(Cost)
        Estimated_R = Decoder.clip(min=0, max=1) # values smaller than 0 become 0, and values larger than 1 become 1
        
        # Error metrics
        RMSE, MAE, ACC, AVG_loglikelihood = evaluation(self.test_R, self.test_mask_R, 
                                                        Estimated_R, self.num_test_ratings)

        self.test_rmse_list.append(RMSE)
        self.test_mae_list.append(MAE)
        self.test_acc_list.append(ACC)
        self.test_avg_loglike_list.append(AVG_loglikelihood)

        # Ranking metrics
        # from utils.Evaluation.Evaluator import EvaluatorHoldout
        
        # evaluator_test = EvaluatorHoldout(self.test_R, cutoff_list=[5, 10])
        # result_dict, _ = evaluator_test.evaluateRecommender(CDAE)
        
        # print("CDAE result_dict MAP@5 {}".format(result_dict[5]["MAP"]))    
        # print("CDAE result_dict MAP@10 {}".format(result_dict[10]["MAP"]))

        # self.test_map_at_5_list.append(result_dict[5]["MAP"])
        # self.test_map_at_10_list.append(result_dict[10]["MAP"])

        MAP_at_5 = top_k_evaluation(self.test_R, Estimated_R, k=5)
        MAP_at_10 = top_k_evaluation(self.test_R, Estimated_R, k=10)
        
        self.test_map_at_5_list.append(MAP_at_5)
        self.test_map_at_10_list.append(MAP_at_10)
        
        if itr % self.display_step == 0:
            print("Testing //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(Cost),
                  "Elapsed time : %d sec" % (time.time() - start_time))
            print("RMSE = {:.4f}".format(RMSE), "MAE = {:.4f}".format(MAE), "ACC = {:.10f}".format(ACC),
                  "AVG Loglike = {:.4f}".format(AVG_loglikelihood))
            print("MAP@5 = {:.4f}".format(MAP_at_5), "MAP@10 = {:.4f}".format(MAP_at_10))
            print("=" * 100)

        # Check min_RMSE, patience, total_patiencep
        if RMSE <= self.min_RMSE:
            self.min_RMSE = RMSE
            self.min_epoch = itr
            self.patience = 0
        else:
            self.patience = self.patience + 1

        # Early stopping
        if (itr > 100) and (self.patience >= self.total_patience):
            self.test_rmse_list.append(self.test_rmse_list[self.min_epoch])
            self.test_mae_list.append(self.test_mae_list[self.min_epoch])
            self.test_acc_list.append(self.test_acc_list[self.min_epoch])
            self.test_avg_loglike_list.append(self.test_avg_loglike_list[self.min_epoch])
            self.earlystop_switch = True
            print ("========== Early Stopping at Epoch %d" %itr)


    def get_URM_train(self):
        return self.R_train.copy()

    def l2_norm(self,tensor):
        return tf.sqrt(tf.reduce_sum(input_tensor=tf.square(tensor)))
