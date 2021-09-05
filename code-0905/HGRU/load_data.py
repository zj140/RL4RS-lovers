# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:20:27 2021

@author: Shirley Wu
"""

import numpy as np

def load_data(filename, user_feat_list = None):
    id2list, list2id = load_list_dict()
    session_len, session_num = [], []
    print(list2id)
    data = []
    length = []
    f = open(filename, 'r')
    f.readline()
    if user_feat_list == None:
        user_feat_list = []
        for i in range(10):
            user_feat_list.append({})
    while True:
        try:
            line = f.readline()
            temp = line.strip().split(' ')
            # user_id user_click_history user_protrait exposed_items labels time
            hist = temp[1].split(',')
            hist_session, cur_session, last_time = [], [], 0
            for tmp in hist:
                i,t = tmp.split(':')
                i,t = int(i), int(t)
                if t - last_time > 7200:
                    if len(cur_session) > 0:
                        hist_session.append(cur_session)
                        session_len.append(len(cur_session))
                    cur_session = [i]
                else:
                    cur_session.append(i)
                last_time = t
            if len(cur_session) > 0:
                hist_session.append(cur_session)
            session_num.append(len(hist_session))

            user_feat = list(map(int, temp[2].split(',')))
            uFeat_id = []
            for i in range(10):
                try:
                    uid = user_feat_list[i][user_feat[i]]
                except:
                    uid = len(user_feat_list[i])
                    user_feat_list[i][user_feat[i]] = uid
                uFeat_id.append(uid)
            items = list(map(int, temp[3].split(',')))
            labels = list(map(int, temp[4].split(',')))
            if sum(labels[:3]) < 3 and sum(labels[3:]) > 0 or sum(labels[:6]) < 6 and sum(labels[6:]) > 0:
                continue
            listLabel = list2id[temp[4]]
            time = int(temp[5])
            data.append([hist_session, items, uFeat_id, labels, listLabel])
        except:
            print(hist_session)
            break
    f.close()
    for i in range(10):
        print(i, len(user_feat_list[i]))
    return data, user_feat_list

def split_data(data):
    num = len(data)
    idxs = list(range(num))
    np.random.shuffle(idxs)
    train_idxs, val_idxs, test_idxs = idxs[:int(num*0.8)], idxs[int(num*0.8):int(num*0.9)], idxs[int(num*0.9):]
    train, val, test = [], [], []
    for idx in train_idxs:
        train.append(data[idx])
    for idx in val_idxs:
        val.append(data[idx])
    for idx in test_idxs:
        test.append(data[idx])
    return train, val, test

def load_item(filename):
    f = open(filename, 'r')
    f.readline()
    # item_id item_vec price location
    item_dict = {}
    price_list, vec4_list, vec5_list = [], [], []
    loc_dict = {}
    while True:
        try:
            line = f.readline()
            temp = line.strip().split(' ')
            iid = int(temp[0])
            item_vec = list(map(float, list(temp[1].split(','))))
            price = float(temp[2])
            loc = int(temp[3])
            price_list.append(price)
            vec4_list.append(item_vec[3])
            vec5_list.append(item_vec[4])
            try:
                loc_dict[loc] += 1
            except:
                loc_dict[loc] = 1
            item_dict[iid] = {'vec': item_vec, 'price': price, 'loc': loc}
        except:
            break
    price_bound = np.percentile(price_list, range(0,101,5))
    vec4_bound = np.percentile(vec4_list, range(0,101,5))
    vec5_bound = np.percentile(vec5_list, range(0,101,5))
    print(price_bound)
    print(vec4_bound)
    print(vec5_bound)
    print(loc_dict)
    return item_dict, vec4_bound, vec5_bound, price_bound

def cut_value(item_dict, vec4_bound, vec5_bound, price_bound):
    for iid in item_dict.keys():
        item_dict[iid]['pid'] = sum(price_bound <= item_dict[iid]['price'])
        vec_id = item_dict[iid]['vec']
        vec_id[3] = sum(vec4_bound <= item_dict[iid]['vec'][3])
        vec_id[4] = sum(vec5_bound <= item_dict[iid]['vec'][4])
        item_dict[iid]['vecID'] = vec_id
    return item_dict

def item_dict2list(item_dict, item_max = 400):
    item_vec = np.zeros((item_max, 5))
    item_price = np.zeros(item_max)
    item_loc = np.zeros(item_max)
    for iid, value in item_dict.items():
        item_vec[iid] = np.array(value['vecID'])
        item_price[iid] = value['pid']
        item_loc[iid] = value['loc']
    return item_vec, item_price, item_loc

def load_list_dict():
    id2list, list2id = {}, {}
    with open('../data/list_dict', 'r') as f:
        while True:
            try:
                line = f.readline()
                temp = line.strip().split('\t')
                id2list[int(temp[1])] = temp[0].split(',')
                list2id[temp[0]] = int(temp[1])
            except:
                print(temp)
                print(id2list)
                print(list2id)
                break 
    return id2list, list2id

# train = load_data('trainset.csv')
# print('Train sample')
# print(train[0])
# train, val, test = split_data(train)
# item_dict, vec4_bound, vec5_bound, price_bound = load_item('item_info.csv')
# item_dict = cut_value(item_dict, vec4_bound, vec5_bound, price_bound)
# item_vec, item_price, item_loc = item_dict2list(item_dict)
# item_vec, item_price, item_loc = item_dict2list(cut_price(load_item('item_info.csv')))
