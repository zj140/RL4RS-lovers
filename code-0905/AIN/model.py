import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Model(object):
    def __init__(self, item_vec, item_price, item_loc, item_max = 400, hist_max = 60):
        self.item_max = item_max
        self.hist_max = hist_max
        self.emb_size = 8
        self.gru_size = 64
        self.init_item_emb(item_vec, item_price, item_loc)
        self.init_user_emb()

    def init_item_emb(self, item_vec, item_price, item_loc):
        self.item_id_emb = tf.get_variable('id_emb', [self.item_max, self.emb_size*4], initializer=tf.random_normal_initializer(stddev=0.01))
        self.item_loc_emb = tf.get_variable('loc_emb', [5, self.emb_size//2], initializer=tf.random_normal_initializer(stddev=0.01))
        self.item_price_emb = tf.get_variable('price_emb', [25, self.emb_size], initializer=tf.random_normal_initializer(stddev=0.01))
        item_set_loc = tf.nn.embedding_lookup(self.item_loc_emb, tf.cast(item_loc, tf.int32))
        item_set_price = tf.nn.embedding_lookup(self.item_price_emb, tf.cast(item_price, tf.int32))

        self.item_feat_emb = []
        for i in range(5):
            self.item_feat_emb.append(
                tf.get_variable('iFeatEmb_'+str(i), [25, self.emb_size//2], initializer=tf.random_normal_initializer(stddev=0.01)))
        item_vec_emb = []
        for i in range(5):
            item_vec_emb.append(tf.nn.embedding_lookup(self.item_feat_emb[i], tf.cast(item_vec[:,i], tf.int32)))
        item_vec_emb = tf.concat(item_vec_emb, axis = -1)
        self.item_emb = tf.concat([self.item_id_emb, item_set_loc, item_set_price, item_vec_emb], axis = -1) # 64

    def init_user_emb(self):
        feat_size = [10, 1400, 30, 20, 200, 50, 10, 20, 10, 2200]
        self.user_feat_emb = []
        self.reg_loss = 0
        for i in range(10):
            self.user_feat_emb.append(
                tf.get_variable('uFeatEmb_'+str(i), [feat_size[i], self.emb_size], initializer=tf.random_normal_initializer(stddev=0.01)))
            self.reg_loss += tf.nn.l2_loss(self.user_feat_emb[i])

    def get_user_feat(self):
        user_rep = []
        for i in range(10):
            user_rep.append(tf.nn.embedding_lookup(self.user_feat_emb[i], self.user_feat[:,i]))
        user_rep_concat = tf.concat(user_rep, axis = -1) # [B, 8*10]
        user_rep_dense = tf.layers.dense(user_rep_concat, 64)
        return user_rep_dense

    def item_prediction(self, context_rep, item_rep, name):
        with tf.variable_scope(name, reuse = False):
            context_rep = tf.tile(tf.expand_dims(context_rep, axis = 1), [1,3,1]) # [B, 3, size]
            final_rep = tf.concat([context_rep, item_rep], axis = -1)
            h1 = tf.layers.dense(final_rep, 64, activation = tf.nn.relu)
            h2 = tf.layers.dense(h1, 16)
            h3 = tf.layers.dense(h2, 2) # [B, 3, 1]
            # pred = tf.reshape(h3, [-1, 3])
        return h3
    
    def inter_list_cxt(self, list1, list2, name):
        with tf.variable_scope(name, reuse = False):
            item_in_list1_W = tf.layers.dense(list1, 64)
            item_in_list1tolist2 = tf.matmul(item_in_list1_W, list2, transpose_b = True) # [B, 3, 3]
            item_in_list1_att = tf.nn.softmax(item_in_list1tolist2, axis = -1)
            item_in_list1_cxt = tf.matmul(item_in_list1_att, list2) # [B, 3, 64]
            overall_rep = tf.concat([list1, item_in_list1_cxt], axis = -1)
            overall_rep = tf.reshape(overall_rep, [-1, 3*64*2])
        return overall_rep # [B, 3, 64*2]

    def build(self):
        self.hist = tf.placeholder(tf.int32, [None, self.hist_max])
        self.hist_len = tf.placeholder(tf.int32, [None])
        self.user_feat = tf.placeholder(tf.int32, [None, 10])
        self.item_rec = tf.placeholder(tf.int32, [None, 9])
        self.label = tf.placeholder(tf.int32, [None, 9])
        self.listLabel = tf.placeholder(tf.int32, [None])
        self.training = tf.placeholder(tf.bool)

        # user feat rep
        user_feat_rep = self.get_user_feat() # [B, 64]

        # Emb for each item in the rec list and emb for each rec list
        item_rec_emb = tf.nn.embedding_lookup(self.item_emb, self.item_rec) # [B, 9, emb_size] 64
        rec_avg = tf.reduce_mean(item_rec_emb, axis = 1)
        item_rec_emb = tf.reshape(item_rec_emb, [-1, 3, 3, 64])
        item_in_list1 = item_rec_emb[:,0,:,:] # [B, 3, emb_size]
        item_in_list2 = item_rec_emb[:,1,:,:]
        item_in_list3 = item_rec_emb[:,2,:,:]
        list1 = tf.reduce_mean(item_in_list1, axis = 1) # [B, emb_size]
        list2 = tf.reduce_mean(item_in_list2, axis = 1)
        list3 = tf.reduce_mean(item_in_list3, axis = 1)

        # user hist: att
        hist_emb = tf.nn.embedding_lookup(self.item_emb, self.hist) # [B, hist_len, 64]
        rec_avg_tile = tf.tile(tf.expand_dims(rec_avg, axis = 1), [1, self.hist_max, 1]) # [B, hist_len, 64]
        product = hist_emb * rec_avg_tile
        w1 = tf.layers.dense(tf.concat([hist_emb, rec_avg_tile, product], axis = -1), 36, activation = tf.nn.relu)
        w2 = tf.layers.dense(w1, 1) # [B, hist_len, 1]
        seq_mask = tf.expand_dims(tf.sequence_mask(self.hist_len, self.hist_max, dtype = tf.float32), axis = -1)
        hist_rep = tf.reduce_sum(hist_emb * w2 * seq_mask, axis = 1) # [B, 64]

        # context_rep for each list
        context_rep_1 = tf.concat([user_feat_rep, hist_rep, list1, list2], axis = -1) # [B, 64*4]
        context_rep_2 = tf.concat([user_feat_rep, hist_rep, list2, list3], axis = -1)
        pad = tf.zeros_like(list3)
        context_rep_3 = tf.concat([user_feat_rep, hist_rep, list3], axis = -1)

        '''
            list prediction
        '''
        # attention list representation
        item_in_list1_with_cxt = self.inter_list_cxt(item_in_list1, item_in_list2, name = 'list1tolist2')
        item_in_list2_with_cxt = self.inter_list_cxt(item_in_list2, item_in_list3, name = 'list2tolist3')
        item_in_list3_flat = tf.reshape(item_in_list3, [-1, 3*64])
        # context_rep_all = tf.concat(
        #    [user_feat_rep, hist_rep, item_in_list1_with_cxt, item_in_list2_with_cxt, item_in_list3_flat], axis = -1)
        item_flat = tf.reshape(item_rec_emb, [-1, 9*64])
        item_rep_with_cxt = tf.concat([item_flat, list1, list2], axis = -1)
        context_rep_all = tf.concat([user_feat_rep, hist_rep, item_rep_with_cxt], axis = -1)
        
        # MLP --> list prediction
        list_h1 = tf.layers.dense(context_rep_all, 128, activation = tf.nn.relu)
        list_h2 = tf.layers.dense(list_h1, 64)
        self.list_h3 = tf.layers.dense(list_h2, 22)
        self.list_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.list_h3, labels=self.listLabel))
        self.list_pred = tf.cast(tf.argmax(self.list_h3, axis = -1), tf.int32)
        self.list_acc = tf.reduce_sum(tf.cast(tf.equal(self.list_pred, self.listLabel), tf.float32))

        # MLP --> item prediction
        pred1 = self.item_prediction(context_rep_1, item_in_list1, 'list1') # [B, 3, 2]
        pred2 = self.item_prediction(context_rep_2, item_in_list2, 'list2')
        pred3 = self.item_prediction(context_rep_3, item_in_list3, 'list3')
        pred = tf.concat([pred1, pred2, pred3], axis = 1) # [B, 9, 2]
        item_label = tf.reshape(self.label, [-1])
        item_pred = tf.reshape(pred, [-1, 2])
        self.cross_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=item_pred, labels=item_label))

        self.reg_loss += tf.nn.l2_loss(self.item_emb)
        # self.cross_loss = -tf.reduce_sum((self.label * tf.log(pred) + (1 - self.label) * tf.log(1 - pred)))
        # self.loss = self.cross_loss + self.list_loss # + 0.0001 * self.reg_loss
        self.loss = self.list_loss # + 0.0001 * self.reg_loss
        self.pred_label = tf.cast(tf.argmax(pred, axis = -1), tf.int32)
        acc = tf.cast(self.pred_label * self.label + (1-self.pred_label) * (1-self.label), tf.float32)
        self.acc = tf.reduce_sum(acc)
        self.acc_all = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(acc, axis = -1), 8.5), dtype=tf.float32))
