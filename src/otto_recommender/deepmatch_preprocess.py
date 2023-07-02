import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


def gen_model_input(train_set, seq_max_len, item_pad, type_pad):
    print('extracting arrays')
    # train_uid = np.array([line[0] for line in train_set])
    train_iid = np.array([line[1] for line in train_set])
    train_label = np.array([line[2] for line in train_set])
    train_seq = [line[3] for line in train_set]
    train_hist_len = np.array([line[4] for line in train_set])
    train_seq_types = np.array([line[5] for line in train_set])
    # train_types = np.array([line[6] for line in train_set])
    print('padding')
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=item_pad)
    train_seq_types_pad = pad_sequences(train_seq_types, maxlen=seq_max_len, padding='post', truncating='post',
                                         value=type_pad)
    train_model_input = {
        # "user_id": train_uid,
        "item_id": train_iid,
        "hist_item_id": train_seq_pad,
        "hist_types": train_seq_types_pad,
        "hist_len": train_hist_len,
        # "types": train_types
    }

    # for key in ["gender", "age", "occupation", "zip"]:
    #     train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


def gen_data_set(data, pad, seq_max_len=10, negsample=0, col_ts='ts', col_item='aid', col_type='type', col_user='user', n_iter=5, mode='train'):
    data.sort_values(col_ts, inplace=True)
    # item_ids = data[col_item].unique()
    # item_id_types_map = dict(zip(data[col_item].values, data[col_type].values))
    train_set = list()
    for reviewerID, hist in tqdm(data.groupby(col_user)):
        pos_list = hist[col_item].tolist()
        types_list = hist[col_type].tolist()
        # rating_list = hist['rating'].tolist()

        # if negsample > 0:
        #     candidate_set = list(set(item_ids) - set(pos_list))
        #     neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        if mode == 'train':
            seed = reviewerID
            np.random.seed(seed)
            idxs = np.arange(1, len(pos_list) - 1)
            idxs = list(np.random.choice(idxs, min(n_iter, len(idxs)), replace=False))
            if len(idxs) > 0:
                idxs.append(len(pos_list) - 1)
                idxs = sorted(list(set(idxs)))
                idxs = [idx for idx in idxs if pos_list[idx] != pad]

                for i in idxs:
                    hist = pos_list[:i]
                    types_hist = types_list[:i]
                    seq_len = min(i, seq_max_len)
                    train_set.append((
                            reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, types_hist[::-1][:seq_len],
                            types_list[i],
                            # rating_list[i]
                        ))
                    # for negi in range(negsample):
                    #     train_set.append((reviewerID, neg_list[i * negsample + negi], 0, hist[::-1][:seq_len], seq_len,
                    #                       genres_hist[::-1][:seq_len], item_id_types_map[neg_list[i * negsample + negi]]))
                # else:
                #     test_set.append((reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                #                      genres_list[i],
                #                      rating_list[i]))

        # random.shuffle(train_set)
        # random.shuffle(test_set)
        else:
            i = len(pos_list) - 1
            hist = pos_list[:i]
            types_hist = types_list[:i]
            seq_len = min(i, seq_max_len)
            train_set.append((
                            reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, types_hist[::-1][:seq_len],
                            types_list[i],
                            # rating_list[i]
                        ))

    print(len(train_set[0]))#, len(test_set[0]))

    return train_set#, test_set



def gen_data_set_sdm(data, seq_short_max_len=5, seq_prefer_max_len=10, col_ts='ts', col_item='aid', col_type='type', col_user='user', n_iter=5):
    data.sort_values(col_ts, inplace=True)
    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby(col_user)):
        pos_list = hist[col_item].tolist()
        genres_list = hist[col_type].tolist()
        # rating_list = hist['rating'].tolist()
        seed = reviewerID
        np.random.seed(seed)
        idxs = np.arange(1, len(pos_list) - 1)
        idxs = list(np.random.choice(idxs, min(n_iter, len(idxs)), replace=False))
        idxs.append(len(pos_list) - 1)
        for i in idxs:
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            seq_short_len = min(i, seq_short_max_len)
            seq_prefer_len = min(max(i - seq_short_len, 0), seq_prefer_max_len)
            array = [reviewerID, #train_uid
                     pos_list[i], #train_iid
                     1, #train_label
                     hist[::-1][:seq_short_len][::-1], #short_train_seq
                     hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], #prefer_train_seq
                     seq_short_len, # train_short_len
                     seq_prefer_len, # train_prefer_len
                     genres_hist[::-1][:seq_short_len][::-1], #short_train_seq_genres
                     genres_hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], #prefer_train_seq_genres
                     # rating_list[i]
                    ]
            train_set.append(tuple(array))

    # random.shuffle(train_set)
    # random.shuffle(test_set)

    print(len(train_set[0]))#, len(test_set[0]))

    return train_set#, test_set


def gen_model_input_sdm(train_set, #user_profile, 
                        seq_short_max_len, seq_prefer_max_len, item_pad, type_pad=3):
    train_uid = np.array([line[0] for line in tqdm(train_set, desc='train_uid')])
    train_iid = np.array([line[1] for line in tqdm(train_set, desc='train_iid')])
    train_label = np.array([line[2] for line in tqdm(train_set, desc='train_label')])
    short_train_seq = [line[3] for line in tqdm(train_set, desc='short_train_seq')]
    prefer_train_seq = [line[4] for line in tqdm(train_set, desc='prefer_train_seq')]
    train_short_len = np.array([line[5] for line in tqdm(train_set, desc='train_short_len')])
    train_prefer_len = np.array([line[6] for line in tqdm(train_set, desc='train_prefer_len')])
    short_train_seq_types = np.array([line[7] for line in tqdm(train_set, desc='short_train_seq_types')])
    prefer_train_seq_types = np.array([line[8] for line in tqdm(train_set, desc='prefer_train_seq_types')])

    train_short_item_pad = pad_sequences(short_train_seq, maxlen=seq_short_max_len, padding='post', truncating='post',
                                         value=item_pad)
    train_prefer_item_pad = pad_sequences(prefer_train_seq, maxlen=seq_prefer_max_len, padding='post',
                                          truncating='post',
                                          value=item_pad)
    train_short_types_pad = pad_sequences(short_train_seq_types, maxlen=seq_short_max_len, padding='post',
                                           truncating='post',
                                           value=type_pad)
    train_prefer_types_pad = pad_sequences(prefer_train_seq_types, maxlen=seq_prefer_max_len, padding='post',
                                            truncating='post',
                                            value=type_pad)

    train_model_input = {"user_id": train_uid, "item_id": train_iid, "short_item_id": train_short_item_pad,
                         "prefer_item_id": train_prefer_item_pad,
                         "prefer_sess_length": train_prefer_len,
                         "short_sess_length": train_short_len, 
                         'short_types': train_short_types_pad,
                         'prefer_types': train_prefer_types_pad}

    # for key in ["gender", "age", "occupation", "zip"]:
    #     train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label
