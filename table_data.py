"""
Make table data by aggregating features
"""
import numpy as np
import pandas as pd
import csv
import sys
import os
import logging
import warnings
import click

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def create_feature_table_data(bbob_suite, table_file_path, feature_dir_path, ap_dir_path, all_feature_classes, dims, per_metric):    
    with warnings.catch_warnings():
        warnings.filterwarnings("error")

        all_fun_ids = range(1, 24+1)
        if bbob_suite == 'bbob-noisy':
            all_fun_ids = range(101, 130+1)
        all_instance_ids = range(1, 5+1)
        all_instance_ids_plus0 = range(0, 5+1)

        ap_config_file_path = os.path.join(ap_dir_path, 'ap_config.csv')
        ap_algs = np.loadtxt(ap_config_file_path, delimiter=",", comments="#", dtype=str)

        # 1. Set the name of each column
        column_names = []
        my_basic_feature_names = ['dim', 'fun', 'instance']
        column_names.extend(my_basic_feature_names)
        # column_names.extend(my_high_level_prop_names)

        # 'best_alg' is for multiclass classfication
        for left_instance_id in all_instance_ids_plus0:
            column_names.append('best_alg_liid{}'.format(left_instance_id))

        # for instance_id in all_instance_ids:    
        #     column_names.append('best_alg_i{}'.format(instance_id))        
        #column_names.extend(ap_algs)
        for alg in ap_algs:
            for left_instance_id in all_instance_ids_plus0:    
                column_names.append('{}_liid{}'.format(alg, left_instance_id))
            
        # Extract the name of all the features
        dim = dims[0]
        instance_id = all_instance_ids[0]
        fun_id = all_fun_ids[0]

        for ela_feature_class in all_feature_classes:
            feature_file_path = os.path.join(feature_dir_path, '{}_{}_f{}_DIM{}_i{}.csv'.format(ela_feature_class, bbob_suite, fun_id, dim, instance_id))        
            if not os.path.exists(feature_file_path):
                print(feature_file_path+" does not exist")
            else:
                try:
                    feature_data_set = np.loadtxt(feature_file_path, delimiter=",", comments="#", dtype=str)
                    feature_names = feature_data_set[:, 0].tolist()
                    column_names.extend(feature_names)
                except IndexError:
                    print("IndexError at "+feature_file_path)
                except UserWarning as warning:
                    print("UserWarning at "+feature_file_path)
                
            
        # 2. Make table data
        table_df = pd.DataFrame(columns=column_names)

        for dim in dims:
            for fun_id in all_fun_ids:
                data_dict = {}

                # Save the relative metric value and the best algorithm
                for left_instance_id in all_instance_ids_plus0:                
                    tmp_metric_dict = {}
                    for alg in ap_algs:
                        sum_metric_value = 0
                        if left_instance_id == 0:
                            metric_file_path = os.path.join(ap_dir_path, per_metric, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim, left_instance_id))
                        else:
                            metric_file_path = os.path.join(ap_dir_path, per_metric, '{}_f{}_DIM{}_liid{}.csv'.format(alg, fun_id, dim, left_instance_id))
                        with open(metric_file_path, 'r') as fh:
                            for str_line in fh:
                                iid, metric_value = str_line.split(',')
                                sum_metric_value += float(metric_value.replace('\n',''))

                            mean_metric_value = sum_metric_value / (len(all_instance_ids_plus0) - 1)
                            alg_name = '{}_liid{}'.format(alg, left_instance_id)
                            data_dict[alg_name] = mean_metric_value
                            tmp_metric_dict[alg] = mean_metric_value
                            
                    data_dict['best_alg_liid{}'.format(left_instance_id)] = min(tmp_metric_dict, key=tmp_metric_dict.get)
                
                data_dict['dim'] = dim
                data_dict['fun'] = fun_id           
                        
                # For each instance, recode the feature values
                for instance_id in all_instance_ids:
                    data_dict['instance'] = instance_id                
                    for ela_feature_class in all_feature_classes:
                        feature_file_path = os.path.join(feature_dir_path, '{}_{}_f{}_DIM{}_i{}.csv'.format(ela_feature_class, bbob_suite, fun_id, dim, instance_id))
                        if not os.path.exists(feature_file_path):
                            print(feature_file_path+" does not exist")
                        else:
                            try:
                                feature_data_set = np.loadtxt(feature_file_path, delimiter=",", comments="#", dtype=str)
                                for key, value in feature_data_set:
                                    data_dict[key] = value
                            except UserWarning as warning:
                                print("UserWarning at "+feature_file_path)
                    table_df = pd.concat([table_df, pd.Series(data_dict).to_frame().T])

        table_df.to_csv(table_file_path, index=False)

def run(surrogate_multiplier, surrogate_sample, separation, dir_sampling_method='ihs_multiplier50_sid0', ap_name='kt_ecj19', per_metric='accuracy', all_feature_classes=['basic', 'ela_distr', 'pca', 'limo', 'ic', 'disp', 'nbc', 'ela_level', 'ela_meta']):
    bbob_suite = 'bbob'
    # all_feature_classes = ['basic', 'ela_distr', 'pca', 'limo', 'ic', 'disp', 'nbc', 'ela_level', 'ela_meta']
    # all_feature_classes = ['basic', 'ela_distr', 'pca', 'limo', 'ic', 'disp', 'ela_meta']
    dims = [2, 3, 5, 10]

    if separation == 'none':
        option = '{}n_{}'.format(surrogate_multiplier, surrogate_sample)
    else:
        option = '{}n_{}_{}'.format(surrogate_multiplier, surrogate_sample, separation)

    feature_dir_path = os.path.join('./ela_feature_dataset/{}'.format(option), dir_sampling_method)
    ap_dir_path = os.path.join('./alg_portfolio', ap_name)
    
    table_dir_path = os.path.join(ap_dir_path, 'feature_table_data/{}'.format(option))
    os.makedirs(table_dir_path, exist_ok=True)
    features_str = '_'.join(all_feature_classes)
    dims_str = '_'.join([str(d) for d in dims])
    table_file_path = os.path.join(table_dir_path, '{}_{}_{}_dims{}.csv'.format(dir_sampling_method, per_metric, features_str, dims_str))

    create_feature_table_data(bbob_suite, table_file_path, feature_dir_path, ap_dir_path, all_feature_classes, dims, per_metric)
        
@click.command()
# @click.option('--surrogate_number', '-sur', required=False, default=2000, type=int, help='Number of surrogated X')
@click.option('--surrogate_multiplier', '-surm', required=True, default=100, type=int, help='Multiplier of surrogated X')
@click.option('--surrogate_sample', '-surs', required=True, default='random', type=str, help='the way of surrogate')
@click.option('--separation', '-spr', required=True, default='none', type=str, help='separate or not')
def main(surrogate_multiplier, surrogate_sample, separation):
    # sampling_method = 'ihs'
    sampling_method = 'lhs'
    
    # sample_multiplier = 10
    # portfolio_list = ['kt', 'dlvat', 'jped', 'bmtp', 'mk', 'ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']
    # portfolio_list = ['ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']
    portfolio_list = ['ls0', 'ls10', 'ls15', 'ls20', 'ls25', 'ls50']

    all_feature_classes_other = ['basic', 'ela_distr', 'pca', 'limo', 'ic', 'disp', 'nbc', 'ela_level', 'ela_meta']
    all_feature_classes_10 = ['basic', 'ela_distr', 'pca', 'limo', 'ic', 'disp', 'ela_level', 'ela_meta']


    # nbc除外条件で実験するため変更

    for sample_multiplier in [0, 10, 15, 20, 25, 50]:
    # for sample_multiplier in [0, 15, 20, 25, 50]:
    # for ap_name in portfolio_list:
        ap_name = 'ls{}'.format(sample_multiplier)
        if sample_multiplier == 0:

            # sample_multiplier == 0, つまりサンプリングを行わないポートフォリオでは, 全てのサンプリングサイズについてAASを行う.
            for sample in [10, 15, 20, 25, 50]:

                for sid in range(0, 31):
                    dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample, sid)
                    # for per_metric in ["accuracy", "RMSE"]:
                    for per_metric in ["accuracy"]:

                        
                        if sample == 10:
                            run(surrogate_multiplier, surrogate_sample, separation, dir_sampling_method, ap_name, per_metric, all_feature_classes_10)
                        else:
                            run(surrogate_multiplier, surrogate_sample, separation, dir_sampling_method, ap_name, per_metric, all_feature_classes_other)

        else:
            for sid in range(0, 31):
                dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                # for per_metric in ["accuracy", "RMSE"]:
                for per_metric in ["accuracy"]:

                    
                    if sample_multiplier == 10:
                        run(surrogate_multiplier, surrogate_sample, separation, dir_sampling_method, ap_name, per_metric, all_feature_classes_10)
                    else:
                        run(surrogate_multiplier, surrogate_sample, separation, dir_sampling_method, ap_name, per_metric, all_feature_classes_other)
                
            logger.info("Done: %s", dir_sampling_method)


if __name__ == '__main__':
    main()