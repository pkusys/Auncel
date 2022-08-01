import time
import numpy as np
import sys
import re
import os.path
import math
import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import lightgbm as lgb
import util

def train_helper(f, params, lgb_train, feature_name, features, targets, tag,
    filename, num_round=100, log_target=False, pred_thresh=0,
    full_feature=True):
    # Train
    start = time.time()
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_round,
                    feature_name=feature_name)
    f.append(['Train time', time.time()-start])
    gbm.save_model(filename)

    # Feature importance
    f.append(['Calculate feature importance...'])
    importance = list(gbm.feature_importance(importance_type="gain"))
    dictImportFeature = {}
    dictFeatureImport = {}
    importance_sum = sum(importance)
    query_sum = 0.0
    for i in range(len(importance)):
        value = importance[i]/importance_sum
        if (gbm.feature_name())[i][:2] == u'F0':
            query_sum += value
        else:
            if value not in dictImportFeature:
                dictImportFeature[value] = []
            dictImportFeature[value].append((gbm.feature_name())[i])
            dictFeatureImport[(gbm.feature_name())[i]] = value 
    if query_sum not in dictImportFeature:
        dictImportFeature[query_sum] = []
    dictImportFeature[query_sum].append('F0_query')
    dictFeatureImport['F0_query'] = query_sum 
    for k in sorted(dictImportFeature.keys(), reverse=True):
        f.append([dictImportFeature[k], k])

    # Performance simulation.
    pred_max = 0
    for i in range(len(features)):
        pred = gbm.predict(features[i])
        count = len(targets[i])
        if log_target:
            if tag[i] == 'Train':
                for j in range(len(pred)):
                    if targets[i][j] >= 0:
                        targets[i][j] = 2.0**targets[i][j]
            pred = [2.0**max(x,0) for x in pred]
        # Exclude the cases that we don't have the ground truth when
        # calculating accuracy metrics.
        pred_valid = []
        targets_valid = []
        count_miss = 0
        for j in range(len(pred)):
            if targets[i][j] > 0:
                pred_valid.append(pred[j])
                targets_valid.append(targets[i][j])
            else:
                targets[i][j] = float('inf')
                count_miss += 1
        ape = []
        for j in range(len(pred_valid)):
            ape.append(abs((targets_valid[j]-pred_valid[j])/targets_valid[j]))
        rmse = mean_squared_error(targets_valid, pred_valid)** 0.5
        mae = mean_absolute_error(targets_valid, pred_valid)
        max_gt = max(targets_valid)
        min_gt = min(targets_valid)
        avg_gt = sum(targets_valid)/float(len(targets_valid))
        f.append(['{} ground truth max'.format(tag[i]), max_gt])
        f.append(['{} ground truth min'.format(tag[i]), min_gt])
        f.append(['{} ground truth avg'.format(tag[i]), avg_gt])
        f.append(['{} ground truth missing'.format(tag[i]), count_miss])
        f.append(['{} rmse'.format(tag[i]), rmse])
        f.append(['{} mae'.format(tag[i]), mae])
        f.append(['{} mape'.format(tag[i]), sum(ape)/float(len(ape))])
        if i == 0: 
            # Training data. No need to simulate the performance. But need to
            # record the maximum ground truth value as the upper bound when
            # simulating the performance of the testing case.
            sorted_gt = sorted(targets[i])
            pred_max = sorted_gt[-1]
        if i == 1:
            # Testing data. Simulate the learned early termination with
            # different multipliers to estimate the performance. Note that this
            # is just an estimation: the prediction overhead is not considered.
            percent = [0.90,0.95,0.96,0.97,0.98,0.99,1.0]
            percent_max = float(count-count_miss)/count
            while percent[-1] > percent_max:
                percent.pop()
            if percent[-1] < percent_max:
                percent.append(percent_max)
            percentile = [(count-1)*x+1 for x in percent]
            
            sorted_gt = sorted(targets[i])
            fixed_config = []
            ratio = []
            multi = []
            for j in range(len(pred)):
                if targets[i][j] > 0 and targets[i][j] <= pred_thresh:
                    ratio.append(1)
                elif targets[i][j] > 0:
                    ratio.append(targets[i][j]/max(pred[j], 1))
                else:
                    ratio.append(1000000000)
            ratio = sorted(ratio)
            for p in range(len(percentile)):
                fixed_config.append(sorted_gt[int(percentile[p])-1])
                f.append(['multiplier P{}: {}'.format(percent[p]*100.0,
                    ratio[int(percentile[p])-1])])
                multi.append(ratio[int(percentile[p])-1])
            fix_count = [0]*len(fixed_config)
            adapt_count = [0]*len(multi)
            adapt_sum = [0]*len(multi)
            for j in range(len(pred)):
                for m in range(len(multi)):
                    if full_feature:
                        # When using the intermediate search result features,
                        # for each query we need to always search until the
                        # feature is ready.
                        prediction_val = math.ceil(
                            max(min(max(pred[j],1)*multi[m], pred_max),
                            pred_thresh))
                    else:
                        prediction_val = math.ceil(
                            min(max(pred[j],1)*multi[m], pred_max))
                    if targets[i][j] > 0 and prediction_val >= targets[i][j]:
                        adapt_count[m] += 1                        
                    adapt_sum[m] += prediction_val
                for t in range(len(fixed_config)):
                    if targets[i][j] > 0 and fixed_config[t] >= targets[i][j]:
                        fix_count[t] += 1
            fix_count = [float(x)/count for x in fix_count]
            adapt_count = [float(x)/count for x in adapt_count]
            adapt_sum = [float(x)/count for x in adapt_sum]
            f.append(['fixed_config_perf =', [fixed_config, fix_count]])
            f.append(['early_termination_perf = ', [adapt_sum, adapt_count]])

def preprocess_and_train(training_dir, model_dir, dbname, index_key, xt, xq,
    full_feature, pred_thresh, feature_idx, billion_scale=False):
    train_file = '{}{}_{}_train.tsv'.format(training_dir, dbname, index_key)
    test_file = '{}{}_{}_test.tsv'.format(training_dir, dbname, index_key)
    if not os.path.isfile(train_file):
        print('training file {} not found'.format(train_file))
        return
    if not os.path.isfile(test_file):
        print('testing file {} not found'.format(train_file))
        return

    if dbname.startswith('SIFT'):
        dim = 128
    elif dbname.startswith('DEEP'):
        dim = 96
    elif dbname.startswith('SPACEV'):
        dim = 100
    elif dbname.startswith('GLOVE'):
        dim = 100
    elif dbname.startswith('TEXT'):
        dim = 200
    elif dbname.startswith('GIST'):
        dim = 960
    else:
        print(sys.stderr, 'unknown dataset', dbname)
        sys.exit(1)

    if index_key[:4] == 'HNSW':
        log_target = True
    else:
        if billion_scale:
            log_target = True
        else:
            log_target = False

    out_buffer = []
    suffix = ''
    if log_target:
        suffix += '_Log'
    if full_feature:
        suffix += '_Full'
    else:
        suffix += '_Query'
    model_name = '{}{}_{}_model_thresh{}{}.txt'.format(model_dir, dbname,
        index_key, pred_thresh, suffix)
    log_name = '{}{}_{}_log_thresh{}{}.tsv'.format(model_dir, dbname,
        index_key, pred_thresh, suffix)
    
    df_train = pd.read_csv(train_file, header=None, sep='\t')
    df_test = pd.read_csv(test_file, header=None, sep='\t')
    
    # Which intermediate search result features will be used.
    if index_key[:4] == 'HNSW':
        keep_idx = [2]+list(range(feature_idx*4+3, feature_idx*4+7))
    else:
        keep_idx = list(range(2,12))+list(range(feature_idx*4+12,
            feature_idx*4+16))
    drop_idx = list(set(list(range(len(df_train.columns)))) - set(keep_idx))

    train_target = (df_train[0].values).astype('float32')
    train_query = xt[df_train[1].values]
    if full_feature:
        train_other = df_train.drop(drop_idx, axis=1).values
        train_feature = np.concatenate((train_query,train_other), axis=1)
    else:
        train_feature = train_query
    valid_training = []
    for i in range(len(train_feature)):
        if train_target[i] > 0:
            if log_target:
                train_target[i] = math.log(train_target[i], 2)
            valid_training.append(i)
    train_target = train_target[valid_training]
    train_feature = train_feature[valid_training,:]
    out_buffer.append(['training count: {} valid rows out of {} total'.format(
        len(train_target), len(df_train))])

    test_target = (df_test[0].values).astype('float32')
    test_query = xq[df_test[1].values]
    if full_feature:
        test_other = df_test.drop(drop_idx, axis=1).values
        test_feature = np.concatenate((test_query,test_other), axis=1)
    else:
        test_feature = test_query
    out_buffer.append(['testing count: {} total rows'.format(
        len(test_target))])

    if train_feature.shape[0] < 2:
        print ('training file {} too small'.format(train_file))
        return

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(train_feature, train_target)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'boost_from_average' : False,
        'learning_rate': 0.2,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'verbose': 0,
        'num_threads': 20        
    }
    num_round = 100
    if billion_scale:
        params['learning_rate'] = 0.05
        num_round = 500

    if train_feature.shape[0] < 100:
        params['min_data'] = 1
        params['min_data_in_bin'] = 1

    feature_name = []
    
    for i in range(dim):
        feature_name.append('F0_query_dim{}'.format(i))
    
    if full_feature:
        if index_key[:4] == 'HNSW':
            feature_name += ['F1_d_start', 'F2_d_1st', 'F3_d_10th',
                'F4_1st_to_start', 'F5_10th_to_start']
        else:
            feature_name += [
                'F1_c_10th_to_c_1st', 'F1_c_20th_to_c_1st',
                'F1_c_30th_to_c_1st', 'F1_c_40th_to_c_1st',
                'F1_c_50th_to_c_1st', 'F1_c_60th_to_c_1st',
                'F1_c_70th_to_c_1st', 'F1_c_80th_to_c_1st',
                'F1_c_90th_to_c_1st', 'F1_c_100th_to_c_1st',
                'F2_d_1st', 'F3_d_10th',
                'F4_d_1st_to_d_10th', 'F5_d_1st_to_c_1st']

    train_helper(out_buffer, params, lgb_train, feature_name,
        [train_feature, test_feature], [train_target, test_target],
        ['Train', 'Test'], model_name, num_round=num_round,
        log_target=log_target, pred_thresh=pred_thresh,
        full_feature=full_feature)
    util.write_tsv(out_buffer, log_name)

if __name__ == "__main__":
    # Where the dataset base, query, learn files are stored.
    DB_DIR = '/workspace/data/'
    # Where the trained prediction model and training logs are stored.
    MODEL_DIR = 'training_model/'
    # Where the training and testing data files are stored.
    TRAINING_DIR = 'training_data/'


    parser = argparse.ArgumentParser(description='training GBDT models')
    parser.add_argument('-train', '--trainsize', help='train size',
        default='1', required=True)
    parser.add_argument('-thresh', '--predthresh',
        help='prediction thresholds', default='1', required=True)
    parser.add_argument('-db', '--dbname', help='database name', required=True)
    parser.add_argument('-idx', '--indexkey', help='index key', required=True)
    args = vars(parser.parse_args())

    train_size = int(args['trainsize']) # num training vectors (in millions)
    # This is related to the intermediate search result features.
    pred_thresh = [int(x) for x in args['predthresh'].split(',')]
    dbname = args['dbname'] # e.g.: SIFT1M
    index_key = args['indexkey'] # e.g.: IVF1000

    if dbname.startswith('SIFT'):
        # xt = util.mmap_fvecs('{}sift/sift1M.fvecs'.format(
        #     DB_DIR))[:train_size*1000*1000]
        xt = util.mmap_fvecs('{}sift/sift10M/query.fvecs'.format(DB_DIR))[:5000]
        xq = util.mmap_fvecs('{}sift/sift10M/query.fvecs'.format(DB_DIR))
    elif dbname.startswith('DEEP'):
        # xt = util.mmap_fvecs('{}deep10M.fvecs'.format(
        #     DB_DIR))[:train_size*1000*1000]
        xt = util.mmap_fvecs('{}deep/query.fvecs'.format(DB_DIR))[:5000]
        xq = util.mmap_fvecs('{}deep/query.fvecs'.format(DB_DIR))
    elif dbname.startswith('GIST'):
        # xt = util.mmap_fvecs('{}gist_learn.fvecs'.format(DB_DIR))
        xt = util.mmap_fvecs('{}gist/query.fvecs'.format(DB_DIR))[:500]
        xq = util.mmap_fvecs('{}gist/query.fvecs'.format(DB_DIR))
    elif dbname.startswith('SPACEV'):
        # xt = util.mmap_fvecs('{}gist_learn.fvecs'.format(DB_DIR))
        xt = util.mmap_fvecs('{}spacev/query.fvecs'.format(DB_DIR))[:5000]
        xq = util.mmap_fvecs('{}spacev/query.fvecs'.format(DB_DIR))
    elif dbname.startswith('GLOVE'):
        # xt = util.mmap_fvecs('{}gist_learn.fvecs'.format(DB_DIR))
        xt = util.mmap_fvecs('{}glove/query.fvecs'.format(DB_DIR))[:5000]
        xq = util.mmap_fvecs('{}glove/query.fvecs'.format(DB_DIR))
    elif dbname.startswith('TEXT'):
        # xt = util.mmap_fvecs('{}gist_learn.fvecs'.format(DB_DIR))
        xt = util.mmap_fvecs('{}text/query.fvecs'.format(DB_DIR))[:5000]
        xq = util.mmap_fvecs('{}text/query.fvecs'.format(DB_DIR))

    for p in range(len(pred_thresh)):
        preprocess_and_train(TRAINING_DIR, MODEL_DIR, dbname, index_key,
            xt, xq, True, pred_thresh[p], p)
        # WE don't need non-full features version
        # preprocess_and_train(TRAINING_DIR, MODEL_DIR, dbname, index_key,
        #     xt, xq, False, pred_thresh[p], p, int(dbname[4:-1]) == 1000)

