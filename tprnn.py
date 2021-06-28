#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math,os
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(784))
def reshape_perm(xs):
    xs_shf = []
    for x in xs:
        x = np.reshape(x, [784])
        x = np.array(list(x))
        x_shf = x[shuffle_indices]
        xs_shf.append(x_shf)
    xs_shf = np.array(xs_shf)
    return xs_shf


class RTModel(object):
    def __init__(self, config):
        self.counts = 0
        self.lr = tf.placeholder(tf.float32, [] ,'learning_rate')
        self.keep_prob = tf.placeholder(tf.float32 , [] , 'keep_prob')
        # self.x = tf.placeholder(tf.int32 , [None , None] , "data")
        self.x = tf.placeholder(tf.float32 , [config.batch_size , config.time_length] , "data")

        x = tf.reshape(self.x , shape = [config.batch_size , config.time_length , config.embedding_size])
        
        self.x_dropout = tf.nn.dropout(x ,  self.keep_prob)
        self.y = tf.placeholder(tf.int32 , [config.batch_size] , 'label')
        self.y_one_hot  = tf.one_hot(self.y, config.n_classes)
        self.para_trainable_list = []


        '''
        self.embedding = tf.Variable(tf.random_normal([config.vocab_size , config.embedding_size] , stddev = 0.1) , trainable = config.embedding_trainable , name = 'embedding')
        # self.sess.run(tf.assign(self.embedding , config.matrix))
        self.initial_hidden_states = tf.nn.embedding_lookup(self.embedding , self.x)
        self.initial_hidden_states = tf.nn.dropout(self.initial_hidden_states , self.keep_prob)
        '''

        # every tree layer and ite layer parameter
        self.w_in_h = [[] for _ in range(config.n_layers)]
        self.w_out_h = [[] for _ in range(config.n_layers)]
        self.w_in_c = [[] for _ in range(config.n_layers)]
        self.w_out_c = [[] for _ in range(config.n_layers)]
        self.w_in_ite = [[] for _ in range(config.n_layers)]
        self.w_out_ite = [[] for _ in range(config.n_layers)]
        for layer in range(config.n_layers) :
            self.w_in_h[layer] = tf.Variable( tf.truncated_normal(shape = [config.agg_num[layer] , config.up_dim * config.agg_num[layer] ] , stddev = 0.1) , name = 'w_in_h_'+str(layer))
            self.w_out_h[layer] = tf.Variable(tf.truncated_normal(shape = [config.up_dim * config.agg_num[layer] , config.agg_num[layer] ] , stddev = 0.1) , name = 'w_out_h_'+str(layer))

            self.w_in_c[layer] = tf.Variable( tf.truncated_normal(shape = [config.agg_num[layer] , config.up_dim * config.agg_num[layer] ] , stddev = 0.1) , name = 'w_in_c_'+str(layer))
            self.w_out_c[layer] = tf.Variable(tf.truncated_normal(shape = [config.up_dim * config.agg_num[layer] , config.agg_num[layer] ] , stddev = 0.1) , name = 'w_out_c_'+str(layer))


            self.w_in_ite[layer] = tf.Variable(tf.truncated_normal(shape = [2 , 2 * config.up_dim] , stddev = 0.1) , name = 'w_in_ite_'+str(layer))
            self.w_out_ite[layer] = tf.Variable(tf.truncated_normal(shape = [config.up_dim * 2 , 2] , stddev = 0.1) , name = 'w_out_ite_'+str(layer))

            self.para_trainable_list += ['w_in_h_'+str(layer) , 'w_out_h_'+str(layer) , 'w_in_c_'+str(layer) , 'w_out_c_'+str(layer) , 'w_in_ite_'+str(layer) , 'w_in_ite_'+str(layer)]


        with tf.variable_scope('MultiLSTM'):
            # 11.0
            self.lstm_cells = [tf.nn.rnn_cell.LSTMCell(config.hidden_size , state_is_tuple = True ,forget_bias = 0.1) for _ in range(config.n_layers)]
            # self.lstm_cells = [tf.contrib.rnn.BasicLSTMCell(config.hidden_size , state_is_tuple = True ,forget_bias = 0.1) for _ in range(config.n_layers)]
            self.lstm_outputs = []
            '''
            x = tf.identity(self.initial_hidden_states)
            '''
            x = self.x_dropout
            time_length = config.time_length
            # windows_size = config.windows_size if config.windows_size % 2 == 0 else config.windows_size + 1
            
            for l in range(config.n_layers):
                x_next = []                                   # next layer input
                counter = 0                                   # calculate the aggregate times for last moment 
                cs = []                                       # save a layer memory cell for feeding to next time
                hs = []
                tree_height = int(math.ceil(math.log(config.windows_size[l] , config.agg_num[l] )))  
                agg_cs = [ [] for _ in range(tree_height)]     # save a layer aggregate result
                agg_hs = [ [] for _ in range(tree_height)]
                agg_ite = []                                   # save the window root of every layer
                initial_state = self.lstm_cells[l].zero_state(tf.shape(x)[0], tf.float32)

                with tf.variable_scope('LSTM_%d' % (l)):
                    for t in range(time_length):
                        if t == 0:
                            _ , (c,h) = self.lstm_cells[l](x[:,t,:], initial_state)
                            cs.append(c)
                            hs.append(h)
                            counter += 1
                            continue
                        tf.get_variable_scope().reuse_variables()

                        agg_times = self._general_count_agg_times(counter , config.agg_num[l])
                        # print(t , counter , agg_times)
                        if agg_times >= 1:
                            # handle the first aggreate layer of tree for every node
                            agg_c = self._self_agg(cs[-config.agg_num[l] :] , l , sign = 'c')
                            agg_h = self._self_agg(hs[-config.agg_num[l] :] , l , sign = 'h')
                            x_next.append(agg_h)
                            agg_cs[0].append(agg_c)
                            agg_hs[0].append(agg_h)
                            # handle higher aggregate for every node
                            for i in range( 1 , agg_times):   
                                agg_c = self._self_agg(agg_cs[i-1][-config.agg_num[l] :] , l , sign = 'c')
                                agg_h = self._self_agg(agg_hs[i-1][-config.agg_num[l] :] , l , sign = 'h')
                                agg_cs[i].append(agg_c)
                                agg_hs[i].append(agg_h)
                                # Pass the result of the aggregation to the next node
                            # _ , (c,h) = self.lstm_cells[l](x[:,t,:], (agg_c ,agg_h))
                        # else:
                        #     _ , (c,h) = self.lstm_cells[l](x[:,t,:],(cs[t-1], hs[t-1]))

                        # handle the window boundary , the last node of window must be tree height 
                        if counter >= config.windows_size[l]:
                            for i in range(agg_times , tree_height): 
                                # the first layer of tree has remain some node need to aggregate
                                if i == 0 :
                                    remain_num = config.windows_size[l] % config.agg_num[l] 
                                    agg_c = self._self_agg(cs[-remain_num:] , l , sign = 'c')
                                    agg_h = self._self_agg(hs[-remain_num:] , l , sign = 'h')
                                    # agg_cs[i].append(agg_c)
                                    # agg_hs[i].append(agg_h)
                                    x_next.append(agg_h)
                                # has remain some node which not aggregate that need to aggregate for creating a tree root
                                else:
                                    remain = len(agg_cs[i-1]) % config.agg_num[l] 
                                    if remain :
                                        list_c = agg_cs[i-1][-remain:]
                                        list_h = agg_hs[i-1][-remain:]
                                        agg_c  = self._self_agg(list_c + [agg_c] , l , sign = 'c')
                                        agg_h  = self._self_agg(list_h + [agg_h] , l , sign = 'h')
                                        # agg_cs[i].append(agg_c)
                                        # agg_hs[i].append(agg_h)
                            # add the root of every window to the agg_ite for aggregate 
                            agg_ite.append(agg_h)
                            # initiate for next window
                            counter = 0
                            agg_cs = [[] for _ in range(tree_height)]
                            agg_hs = [[] for _ in range(tree_height)]

                        if agg_times >=1 or counter >= config.windows_size[l] :
                            _ , (c,h) = self.lstm_cells[l](x[:,t,:], (agg_c ,agg_h))
                        else:
                            _ , (c,h) = self.lstm_cells[l](x[:,t,:],(cs[t-1], hs[t-1]))

                        cs.append(c)
                        hs.append(h)
                        counter += 1  
                    # handle the last node's aggregate time of sequence 
                    last_tree_height = int(math.ceil(math.log(counter , config.agg_num[l] )))
                    agg_times = self._general_count_agg_times(counter , config.agg_num[l])
                    if agg_times >= 1:
                        # agg_c = self._self_agg(cs[-config.agg_num:] , sign = 'c')
                        agg_h = self._self_agg(hs[-config.agg_num[l] :] , l , sign = 'h')
                        x_next.append(agg_h)
                        # agg_cs[0].append(agg_c)
                        agg_hs[0].append(agg_h)
                        for i in range( 1 , agg_times):   
                            # agg_c = self._self_agg(agg_cs[i-1][-config.agg_num:] , sign = 'c')
                            agg_h = self._self_agg(agg_hs[i-1][-config.agg_num[l] :] , l , sign = 'h')
                            # agg_cs[i].append(agg_c)
                            agg_hs[i].append(agg_h)
                    for t in range(agg_times , last_tree_height):
                        if t == 0:
                            remain_num = counter % config.agg_num[l]
                            # agg_c = self._self_agg(cs[-remain_num:] , sign = 'c')
                            agg_h = self._self_agg(hs[-remain_num:] , l , sign = 'h')
                            # agg_cs[i].append(agg_c)
                            # agg_hs[i].append(agg_h)
                            x_next.append(agg_h)
                        else:
                            remain = len(agg_cs[t-1]) % config.agg_num[l]
                            if remain :
                                # list_c = agg_cs[t-1][-remain:]
                                list_h = agg_hs[t-1][-remain:]
                                # agg_c  = self._self_agg(list_c + [agg_c])
                                agg_h  = self._self_agg(list_h + [agg_h] , l , sign = 'h')
                    agg_ite.append(agg_h)

                    # aggregate window root of every layer
                    agg_h  = agg_ite[0]
                    for i in range(1,len(agg_ite)):
                        agg_h  = self._self_agg_ite([agg_ite[i], agg_h] , l )
                    # add the last aggregate result of every layer to classification 
                    self.lstm_outputs.append(agg_h)
                    # coonstruct next layer input
                    time_length = len(x_next)
                    x = tf.reshape(tf.concat(x_next, axis=1) , shape=[-1, time_length , config.hidden_size])
                    ''' window size halve ,need to satisfy window size > agg_num
                    temp = windows_size // 2
                    if temp >= config.agg_num :
                        windows_size = temp 
                        if windows_size % 2 != 0:
                            windows_size += 1'''
        
        if config.n_layers > 1:
            up_dim = 3
            stddev = 0.1
            self.lstm_outputs = tf.concat(self.lstm_outputs , axis = 1)
            self.lstm_outputs = tf.reshape(self.lstm_outputs , shape = [-1 , config.n_layers , config.hidden_size])
            # map to same spaces
            lstm_outputs = []
            for l in range(config.n_layers):
                w = tf.Variable(tf.truncated_normal(shape = [config.hidden_size , config.hidden_size], stddev = stddev) , name = 'map_'+ str(l))
                lstm_outputs.append(tf.matmul(self.lstm_outputs[:,l,:],w))
                self.para_trainable_list += ['map_'+ str(l)]

            lstm_outputs = tf.concat(lstm_outputs , axis = 1)
            lstm_outputs = tf.reshape(lstm_outputs , shape = [-1 , config.n_layers , config.hidden_size])
            lstm_outputs_cp = tf.reshape(tf.transpose(lstm_outputs , perm = [0,2,1]) ,shape = [-1 , config.n_layers ])
            w = tf.Variable(tf.truncated_normal(shape = [config.n_layers , up_dim * config.n_layers],stddev = stddev) , name = 'ouput_up_w')
            s = tf.nn.relu(tf.matmul(lstm_outputs_cp, w) )
            w = tf.Variable(tf.truncated_normal(shape = [up_dim * config.n_layers , config.n_layers] ,stddev = stddev) , name = 'ouput_do_w')
            s = tf.nn.sigmoid(tf.matmul(s, w) )
            self.lstm_output_agg  = tf.reduce_sum(tf.multiply(lstm_outputs , tf.reshape(s, shape= [-1, config.n_layers ,config.hidden_size])), axis=1)
            self.para_trainable_list += ['ouput_up_w' , 'ouput_do_w']
        
        with tf.name_scope('Softmax'):
            self.w = tf.Variable(tf.truncated_normal(shape = [config.hidden_size , config.n_classes], stddev = 0.1) , name = 'softmax_w')
            self.b = tf.Variable(tf.constant(0.1 , shape = [config.n_classes , ]) , name = 'softmax_b')
            self.para_trainable_list += ['softmax_w' , 'softmax_b']
            if config.n_layers > 1:
                self.prediction  = tf.nn.softmax(tf.matmul(self.lstm_output_agg , self.w) + self.b)
            else:
                self.prediction = tf.nn.softmax(tf.matmul(self.lstm_outputs[0], self.w) + self.b)
            
        self.loss = tf.reduce_mean( -tf.reduce_sum(self.y_one_hot * tf.log(self.prediction + 1e-10), axis = 1))
        tvars = tf.trainable_variables()
        grads , _ =tf.clip_by_global_norm(tf.gradients(self.loss , tvars) , config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads , tvars) , global_step = tf.train.get_or_create_global_step())

        correct_predict  = tf.equal(tf.argmax(self.prediction , 1) , tf.argmax(self.y_one_hot , 1))
        self.acc = tf.reduce_mean(tf.cast(correct_predict ,"float"))

        print('Total number of trainable parameters: %d' % (np.sum([np.prod(v.get_shape().as_list()) for v in tvars])))

        self.var_trainable_list = tf.trainable_variables()
        print('count times : ', self.counts)

    def _self_agg(self , nodes , layer , up_dim = 3 , stddev = 0.1 , sign = 'c'):
        self.counts += 1
        n_nodes = len(nodes)
        if n_nodes < config.agg_num[layer] :
            len_zeros = config.agg_num[layer] - n_nodes
        else :
            len_zeros = 0
        concat_nodes = tf.concat(nodes, axis = 1)
        nodes_cp = tf.reshape(concat_nodes ,shape = [-1 , n_nodes , config.hidden_size])
        nodes = tf.reshape(tf.transpose(nodes_cp , perm = [0,2,1]) ,shape = [-1 , n_nodes ])
        if len_zeros :
            if sign == 'c':
                w_in_slice = tf.slice(self.w_in_c[layer] , [0,0] , [n_nodes , n_nodes * config.up_dim])
                w_out_slice = tf.slice(self.w_out_c[layer] , [0,0] , [n_nodes * config.up_dim , n_nodes])
            else :
                w_in_slice = tf.slice(self.w_in_h[layer] , [0,0] , [n_nodes , n_nodes * config.up_dim])
                w_out_slice = tf.slice(self.w_out_h[layer] , [0,0] , [n_nodes * config.up_dim , n_nodes])
        else :
            if sign == 'h':
                w_in_slice = self.w_in_h[layer]
                w_out_slice = self.w_out_h[layer]
            else :
                w_in_slice = self.w_in_c[layer]
                w_out_slice = self.w_out_c[layer]
        s = tf.nn.relu(tf.matmul(nodes, w_in_slice) )
        s = tf.nn.sigmoid(tf.matmul(s, w_out_slice) )
        node_agg  = tf.reduce_sum(tf.multiply(nodes_cp , tf.reshape(s, shape= [-1, n_nodes ,config.hidden_size])), axis=1)
        node_agg = tf.nn.tanh(node_agg)
        return node_agg

    def _self_agg_ite(self , nodes ,layer , up_dim = 3 , stddev = 0.1):
        self.counts += 1
        n_nodes = len(nodes)
        concat_nodes = tf.concat(nodes, axis = 1)
        nodes_cp = tf.reshape(concat_nodes ,shape = [-1 , n_nodes , config.hidden_size])
        nodes = tf.reshape(tf.transpose(nodes_cp , perm = [0,2,1]) ,shape = [-1 , n_nodes ])
        s = tf.nn.relu(tf.matmul(nodes, self.w_in_ite[layer]) )
        s = tf.nn.sigmoid(tf.matmul(s, self.w_out_ite[layer]) )
        node_agg  = tf.reduce_sum(tf.multiply(nodes_cp , tf.reshape(s, shape= [-1, n_nodes ,config.hidden_size])), axis=1)
        node_agg = tf.nn.tanh(node_agg)
        return node_agg

    def _count_agg_times(self , counter , agg_len = 2):
        agg_times = 0
        while(counter & 1 == 0):
            agg_times += 1
            counter >>= 1
        return agg_times

    def _general_count_agg_times(self , data , num = 2):
        remainders = []
        zero_nums = 0
        while(data):
            re = data % num
            if re == 0:
                zero_nums += 1
            else:
                break
            remainders.append(re)
            data //= num
        #print(remainders)
        return zero_nums

class Config(object):
    def __init__(self, time_length, hidden_size, n_layers, windows_size, agg_num, keep_prob = 0.5 , max_step = 20000):
        self.embedding_size = 1
        self.time_length = time_length      # the length of input sequence
        self.n_classes = 10                 # the class num 
        self.hidden_size = hidden_size      # the hidden size of tprnn
        self.n_layers = n_layers            # the number of tprnn layer
        self.windows_size = windows_size    # the parameter L (L=T/N)
        self.agg_num = agg_num              # the parameter of aggregation granularity $g$
        self.lr = 1e-3
        self.lr_decay = 0.99
        self.keep_prob = keep_prob          # dropout rate
        self.max_grad_norm = 3
        self.batch_size = 50
        self.max_step = max_step
        self.up_dim = 3
        self.seed = 1234

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

	#----------------------------params------------------------------------
    time_length = 784
    hidden_size = 100
    n_layers = 3
    windows_size = [49 , 16 , 4]
    agg_num = [7 , 4 , 2]
    keep_prob = 1.0 
    max_step = 220000
    fold_num = 10

    #----------------------------data------------------------------------------
    mnist = input_data.read_data_sets('data/' )
    train_num = mnist.train.num_examples # 55000
    dev_num = mnist.validation.num_examples # 10000
    test_num = mnist.test.num_examples # 5000
    
    print('keep prob : '+str(kp)+' hidden size: '+ str(hs) +' layers: '+ str(nl) +' windows size: '+ str(windows_size) + ' agg num: '+ str(agg_num))
    
    with tf.Graph().as_default() :
        config = Config(time_length , hs , nl , windows_size , agg_num , kp , max_step)		
        
        tf_config = tf.ConfigProto(allow_soft_placement = True)
        tf_config.gpu_options.allow_growth = True
        model = RTModel(config)
        #saver = tf.train.Saver()
        #folder = os.path.exists(config.save_path)
        #if not folder :
        #    os.makedirs(config.save_path)
        
        with tf.Session(config = tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            max_dev_acc = 0
            max_test_acc = 0
            lr = config.lr
            
            # train stage
            for step in range(1, config.max_step + 1):
                data_batch, label_batch = mnist.train.next_batch(config.batch_size)
                data_batch = reshape_perm(data_batch)
                _ , loss, acc = sess.run([model.train_op, model.loss, model.acc], feed_dict = {model.x:data_batch, model.y:label_batch, model.lr:lr, model.keep_prob:config.keep_prob})
                
                if step % 1100 == 0:
                    # dev stage
                    dev_batch = int(math.ceil(dev_num / config.batch_size))
                    dev_accs = []
                    for d in range(dev_batch):
                        dev_data , dev_label = mnist.validation.next_batch(config.batch_size)
                        dev_data = reshape_perm(dev_data)
                        dev_acc =  model.acc.eval( feed_dict = { model.x : dev_data , model.y : dev_label , model.keep_prob : 1.0 })
                        dev_accs.append(dev_acc * dev_data.shape[0])
                    dev_accuracy = sum(dev_accs) / dev_num
                    # test stage
                    test_batch = int(math.ceil(test_num / config.batch_size))
                    test_accs = []
                    for d in range(test_batch):
                        test_data , test_label = mnist.test.next_batch(config.batch_size)
                        test_data = reshape_perm(test_data)
                        test_acc =  model.acc.eval( feed_dict = { model.x : test_data , model.y : test_label , model.keep_prob : 1.0 })
                        test_accs.append(test_acc * test_data.shape[0])
                    test_accuracy = sum(test_accs) / test_num
                    if dev_accuracy > max_dev_acc :
                        max_dev_acc = dev_accuracy
                        max_test_acc = test_accuracy
                        print('save model of step %d : %f , %f' %(step , max_dev_acc , max_test_acc))
                    print('step : %d , loss : %f , train_acc : %f , dev_acc : %f , test_acc : %f'% (step, loss , acc , dev_accuracy ,  test_accuracy))
                    
                    if step >= 165000 :
                        lr *= config.lr_decay
                if step % 55000 == 0:
                    print('step : %d , max_dev_acc : %f , max_test_acc : %f'% (step, max_dev_acc,  max_test_acc))
            
            print('The result:')
            print('max_dev_acc : %f , max_test_acc : %f'% (max_dev_acc,  max_test_acc))
    print('\n\n----------------------------------------------------------------------\n\n')