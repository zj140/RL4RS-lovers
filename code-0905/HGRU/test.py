from load_data import *
from model import *
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

def get_feed_dict(data, training = True, max_num = 10, max_len = 22):
    # [hist_item, items, user_feat, labels]
    item_rec, label, listLabel = [], [], []
    hist, hist_len, sess_num = [], [], []
    user_feat = []

    for d in data:
        hist_session = d[0]
        cur_sess_num = len(hist_session)
        if cur_sess_num > max_num:
            hist_session = hist_session[-max_num:]
            sess_num.append(max_num)
        else:
            sess_num.append(cur_sess_num)
        cur_sess, cur_sess_len = [], [] # [max_num, max_len], [max_num]
        for i, seq in enumerate(hist_session):
            seq_len = len(seq)
            if seq_len < max_len:
                cur_sess.append(seq + (max_len-seq_len)*[0])
                cur_sess_len.append(seq_len)
            else:
                cur_sess.append(seq[-max_len:])
                cur_sess_len.append(max_len)
        if cur_sess_num < max_num:
            for i in range(cur_sess_num, max_num):
                cur_sess.append([0] * max_len)
                cur_sess_len.append(0)
        hist.append(cur_sess)
        hist_len.append(cur_sess_len)

        item_rec.append(d[1])
        user_feat.append(d[2])
        label.append(d[3])
        listLabel.append(d[4])
    return {model.item_rec: item_rec, model.sess_num: sess_num, model.hist: hist, model.hist_len: hist_len, \
        model.training: training, model.user_feat: user_feat, model.label: label, model.listLabel: listLabel}

def evaluate(sess, data):
    start = 0
    data_size = len(data)
    batch_size = 128
    total_num, total_acc, total_acc_all, total_list_acc = 0, 0.0, 0.0, 0.0
    while start < data_size:
        end = min(data_size, start + batch_size)
        total_num += 9 * (end-start)
        feed_dict = get_feed_dict(data[start:end], training = False)
        acc, acc_all, list_acc = sess.run(
            [model.acc, model.acc_all, model.list_acc], feed_dict)
        start = end
        total_acc += acc
        total_acc_all += acc_all
        total_list_acc += list_acc
    return total_acc/total_num*100.0, total_acc_all/total_num*900.0, total_list_acc/total_num*900.0

def prediction(sess, data):
    start = 0
    data_size = len(data)
    batch_size = 128
    pred_all, pred_list_all = [], []
    while start < data_size:
        end = min(data_size, start + batch_size)
        feed_dict = get_feed_dict(data[start:end], training = False)
        acc, acc_all, pred_label, list_pred = sess.run(
            [model.acc, model.acc_all, model.pred_label, model.list_pred], feed_dict)
        start = end
        pred_all.extend(pred_label)
        pred_list_all.extend(list_pred)
    return pred_list_all

def load_list_label_dict():
    id2list = {}
    with open('../data/list_dict', 'r') as f:
        while True:
            try:
                line = f.readline()
                temp = line.strip().split('\t')
                id2list[int(temp[1])] = ' '.join(temp[0].split(','))
            except:
                break
    return id2list

if __name__ == '__main__':
    random_seed = 1234
    batch_size = 128
    epoch_num =  100
    early_stop = 0
    best_acc = 0
    item_dict, vec4_bound, vec5_bound, price_bound = load_item('../data/item_info.csv')
    item_dict = cut_value(item_dict, vec4_bound, vec5_bound, price_bound)
    item_vec, item_price, item_loc = item_dict2list(item_dict)
    model = Model(item_vec=item_vec, item_price=item_price, item_loc=item_loc, item_max = 400)
    model.build()
    tf.set_random_seed(random_seed)

    data_all, user_feat_list = load_data('../data/trainset.csv')
    train, val, test = split_data(data_all)
    train_size = len(train)
    gt_test, _ = load_data('../data/track1_testset.csv', user_feat_list)
    print('='*20)
    print(gt_test[0])

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(model.loss)
    train_op = optimizer
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver_path =  sys.argv[2] + "_saver/model"

    saver.restore(sess, tf.train.latest_checkpoint(sys.argv[2] + '_saver'))
    result, acc_all, list_acc = evaluate(sess, val)
    print('='*10, 'Best result on Validation: %.2f full acc: %.2f list acc: %.2f' %(
        result, acc_all, list_acc))
    result, acc_all, list_acc = evaluate(sess, test)
    print('='*10, 'Best result on Test: %.2f full acc: %.2f list acc: %.2f' %(
        result, acc_all, list_acc))


    id2list = load_list_label_dict()
    print(id2list)
    pred_list_label = prediction(sess, gt_test)

    with open('pred_' + sys.argv[2], 'w') as f:
        f.write('id,category\n')
        for i, pred in enumerate(pred_list_label):
            f.write('%d,%s\n' %(i+1, id2list[pred]))
