from codecs import ignore_errors
import numpy as np
import pandas as pd
import statistics as stat
import csv
import os
import sys
import logging
# import random
import json
import shutil
# from fopt_info import bbob_fopt

from scipy import stats
# from statistics import mean, stdiv

from matplotlib import pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(filename='{}.log'.format(__file__), level=logging.DEBUG)



def precision_datarframize(dims, sampling_method, per_metric, feature_selector, n_features_to_select, bbob_suite='bbob'):

    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'

    aggregate_dir_path = 'pp_aggregate'
    os.makedirs(aggregate_dir_path, exist_ok=True)

    all_id_funs = range(1, 24+1)
    if bbob_suite == 'bbob_noisy':
         all_id_funs = range(101, 130+1)

    all_test_instance_id = range(1, 5+1)

    # multiplier_list = [10, 15, 20, 25, 50]
    # ap_list = ['ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']
    multiplier_list = [10, 15, 20, 25, 50]
    ap_list = ['ls0', 'ls10', 'ls15', 'ls20', 'ls15', 'ls50']
    dim_list = [2, 3, 5, 10]

    all_sid = range(0, 31)


    # df_total = pd.DataFrame(columns=['setting', 'sbs_or_over', 'vbs_precision', 'vbs_or_over'])


    for sample_multiplier in multiplier_list:
        multiplier_path = os.path.join(aggregate_dir_path, 'multiplier{}'.format(sample_multiplier))
        os.makedirs(multiplier_path, exist_ok=True)

        if sample_multiplier == 10:
            ela_feature_classes = ela_feature_classes_10
        else:
            ela_feature_classes = ela_feature_classes_other

        df_multiplier = pd.DataFrame(columns=['setting', 'sbs_or_over', 'diffs_from_sbs', 'vbs_precision', 'vbs_or_over', 'diffs_from_vbs', 'soo_percentage', 'v_p_percentage', 'voo_percentage'])


        for dim in dim_list:
            dim_path = os.path.join(multiplier_path, 'DIM{}'.format(dim))
            os.makedirs(dim_path, exist_ok=True)
            df_dim = pd.DataFrame(columns=['setting', 'sbs_or_over', 'diffs_from_sbs', 'vbs_precision', 'vbs_or_over', 'diffs_from_vbs', 'soo_percentage', 'v_p_percentage', 'voo_percentage'])

            # for pre_solver in [None, 'slsqp_multiplier50', 'smac_multiplier50']:
            for pre_solver in [None]:
                for selector in ['hiearchical_regression']:
                    # for cross_valid_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
                    for cross_valid_type in ['lopo_cv']:

                        
                        

                        # for ap_name in ['kt', 'dlvat', 'jped', 'bmtp', 'mk', 'ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']:
                        for ap_name in ap_list:
                            if ap_name != 'ls0':
                                if ap_name != 'ls{}'.format(sample_multiplier):
                                    continue
                            
                            df_ap = pd.DataFrame(columns=['setting', 'sbs_or_over', 'diffs_from_sbs', 'vbs_precision', 'vbs_or_over', 'diffs_from_vbs', 'soo_percentage', 'v_p_percentage', 'voo_percentage'])


                            for fun_id in all_id_funs:

                                sbs_or_over_summary = 0
                                vbs_precision_summary = 0
                                vbs_or_over_summary = 0

                                sbs_diffs_summary = 0
                                vbs_diffs_summary = 0

                                for sid in all_sid:
                                    dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                    table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                    pp_as_result_dir_path = os.path.join('pp_as_results', '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector))
                                    if feature_selector != 'none':
                                        pp_as_result_dir_path += '_nfs{}'.format(n_features_to_select)

                                    csv_file_path = os.path.join(pp_as_result_dir_path, 'f{}_DIM{}.csv'.format(fun_id, dim))
                                    
                                    sbs_or_over_count = 0
                                    vbs_precision_count = 0
                                    vbs_or_over_count = 0

                                    sbs_diffs = 0
                                    vbs_diffs = 0

                                    df_csv_file = pd.read_csv(csv_file_path, dtype={'selected_alg_error':float, 
                                                                                    'sbs_error':float, 
                                                                                    'vbs_alg_error':float,
                                                                                    'sbs_or_over':bool,
                                                                                    'vbs_selecting_precision':bool,
                                                                                    'vbs_or_over':bool
                                                                                    })

                                    sbs_or_over_count = sum(df_csv_file['sbs_or_over'])
                                    vbs_precision_count = sum(df_csv_file['vbs_selecting_precision'])
                                    vbs_or_over_count = sum(df_csv_file['vbs_or_over'])

                                    sbs_diffs = sum(df_csv_file['selected_alg_error']-df_csv_file['sbs_error'])
                                    vbs_diffs = sum(df_csv_file['selected_alg_error']-df_csv_file['vbs_alg_error'])

                                    sbs_or_over_summary += sbs_or_over_count
                                    vbs_precision_summary += vbs_precision_count
                                    vbs_or_over_summary += vbs_or_over_count
                                    sbs_diffs_summary += sbs_diffs
                                    vbs_diffs_summary += vbs_diffs

                            
                                dict_ap_fun = {
                                    'setting':'f{}'.format(fun_id),
                                    'sbs_or_over':sbs_or_over_summary,
                                    'vbs_precision':vbs_precision_summary,
                                    'vbs_or_over':vbs_or_over_summary,
                                    'diffs_from_sbs':sbs_diffs_summary,
                                    'diffs_from_vbs':vbs_diffs_summary,
                                    'soo_percentage':(sbs_or_over_summary / ( len(all_sid) * len(all_test_instance_id) )),
                                    'v_p_percentage':(vbs_precision_summary / ( len(all_sid) * len(all_test_instance_id) )),
                                    'voo_percentage':(vbs_or_over_summary / ( len(all_sid) * len(all_test_instance_id) ))
                                }
                                
                                df_ap = df_ap.append(dict_ap_fun, ignore_index=True, sort=False)
                            


                            dict_ap = {
                                'setting':'{}'.format(ap_name),
                                'sbs_or_over':df_ap['sbs_or_over'].sum(),
                                'vbs_precision':df_ap['vbs_precision'].sum(),
                                'vbs_or_over':df_ap['vbs_or_over'].sum(),
                                'diffs_from_sbs':df_ap['diffs_from_sbs'].sum(),
                                'diffs_from_vbs':df_ap['diffs_from_vbs'].sum(),
                                'soo_percentage':(df_ap['sbs_or_over'].sum() / ( len(all_sid) * len(all_test_instance_id) * len(all_id_funs) )),
                                'v_p_percentage':(df_ap['vbs_precision'].sum() / ( len(all_sid) * len(all_test_instance_id) * len(all_id_funs) )),
                                'voo_percentage':(df_ap['vbs_or_over'].sum() / ( len(all_sid) * len(all_test_instance_id) * len(all_id_funs) ))
                            }
                            
                            df_ap = df_ap.append(dict_ap, ignore_index=True, sort=False)
                            aggregate_ap_dim_dir_path = os.path.join(dim_path, '{}_DIM{}.csv'.format(ap_name, dim))
                            df_ap.to_csv(aggregate_ap_dim_dir_path, index=False)


                            df_dim = df_dim.append(dict_ap, ignore_index=True, sort=False)



                        dict_dim = {
                            'setting':'DIM{}'.format(dim),
                            'sbs_or_over':df_dim['sbs_or_over'].sum(),
                            'vbs_precision':df_dim['vbs_precision'].sum(),
                            'vbs_or_over':df_dim['vbs_or_over'].sum(),
                            'diffs_from_sbs':df_dim['diffs_from_sbs'].sum(),
                            'diffs_from_vbs':df_dim['diffs_from_vbs'].sum(),
                            'soo_percentage':(df_dim['sbs_or_over'].sum() / ( len(all_sid) * len(all_test_instance_id) * len(all_id_funs) * len(ap_list) )),
                            'v_p_percentage':(df_dim['vbs_precision'].sum() / ( len(all_sid) * len(all_test_instance_id) * len(all_id_funs) * len(ap_list) )),
                            'voo_percentage':(df_dim['vbs_or_over'].sum() / ( len(all_sid) * len(all_test_instance_id) * len(all_id_funs) * len(ap_list) ))
                        }
                        
                        df_dim = df_dim.append(dict_dim, ignore_index=True, sort=False)
                        aggregate_dim_dir_path = os.path.join(dim_path, 'DIM{}.csv'.format(dim))
                        df_dim.to_csv(aggregate_dim_dir_path, index=False)


                        df_multiplier = df_multiplier.append(dict_dim, ignore_index=True, sort=False)

                        logger.info("Aggregation of %s was done.", dim_path)


        dict_multiplier = {
            'setting':'total',
            'sbs_or_over':df_multiplier['sbs_or_over'].sum(),
            'vbs_precision':df_multiplier['vbs_precision'].sum(),
            'vbs_or_over':df_multiplier['vbs_or_over'].sum(),
            'diffs_from_sbs':df_multiplier['diffs_from_sbs'].sum(),
            'diffs_from_vbs':df_multiplier['diffs_from_vbs'].sum(),
            'soo_percentage':(df_multiplier['sbs_or_over'].sum() / ( len(all_sid) * len(all_test_instance_id) * len(all_id_funs) * len(dim_list) * len(ap_list) )),
            'v_p_percentage':(df_multiplier['vbs_precision'].sum() / ( len(all_sid) * len(all_test_instance_id) * len(all_id_funs) * len(dim_list) * len(ap_list) )),
            'voo_percentage':(df_multiplier['vbs_or_over'].sum() / ( len(all_sid) * len(all_test_instance_id) * len(all_id_funs) * len(dim_list) * len(ap_list) ))
        }
                        
        df_multiplier = df_multiplier.append(dict_multiplier, ignore_index=True, sort=False)
        aggregate_multiplier_dir_path = os.path.join(multiplier_path, '{}.csv'.format(sample_multiplier))
        df_multiplier.to_csv(aggregate_multiplier_dir_path, index=False)


        logger.info("Aggregation of %s was done.", multiplier_path)



def Friedman(dims, sampling_method, per_metric, feature_selector, n_features_to_select, bbob_suite = 'bbob'):

    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set3')


    # 有意差検定には31試行のデータをそれぞれで使うべき？
    # portfolio、multiplierに対応する31試行それぞれのデータを取得して比較
    # precision_datarframizeを改変（multiplier/sidで分類）

    df_sta_sig = pd.DataFrame(columns=['multiplier10', 'multiplier15', 'multiplier20', 'multiplier25', 'multiplier50', 'ave', 'std', 'sig_bool'])

    metric = 'v_p_percentage'

    pp_agg_dir = 'pp_aggregate/pp_aggregate'

    
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'



    all_id_funs = range(1, 24+1)
    if bbob_suite == 'bbob_noisy':
         all_id_funs = range(101, 130+1)

    all_test_instance_id = range(1, 5+1)

    multiplier_list = [10, 15, 20, 25, 50]
    ap_list = ['ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']
    dim_list = [2, 3, 5, 10]

    all_sid = range(0, 31)


    for precision_metric in ['sbs_or_over', 'vbs_selecting_precision']:

        for dim in dim_list:

            for ap_name in ap_list:


                value_10 = [0]*len(all_sid)
                value_15 = [0]*len(all_sid)
                value_20 = [0]*len(all_sid)
                value_25 = [0]*len(all_sid)
                value_50 = [0]*len(all_sid)

                for sid in all_sid:


                    for sample_multiplier in multiplier_list:
                        if sample_multiplier == 10:
                            ela_feature_classes = ela_feature_classes_10
                        else:
                            ela_feature_classes = ela_feature_classes_other

                        count = 0


                        # for pre_solver in [None, 'slsqp_multiplier50', 'smac_multiplier50']:
                        for pre_solver in [None]:
                            for selector in ['hiearchical_regression']:
                                # for cross_valid_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
                                for cross_valid_type in ['lopo_cv']:


                                

                                    for fun_id in all_id_funs:

                                    

                                        dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                        table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                        csv_file_path = os.path.join('pp_as_results', '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))


                                        df_csv_file = pd.read_csv(csv_file_path, dtype={'selected_alg_error':float, 
                                                                                        'sbs_error':float, 
                                                                                        'vbs_alg_error':float,
                                                                                        'sbs_or_over':bool,
                                                                                        'vbs_selecting_precision':bool,
                                                                                        'vbs_or_over':bool
                                                                                        })


                                        count += sum(df_csv_file[precision_metric])



                        if sample_multiplier == 10:
                            value_10[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif sample_multiplier == 15:
                            value_15[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif sample_multiplier == 20:
                            value_20[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif sample_multiplier == 25:
                            value_25[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif sample_multiplier == 50:
                            value_50[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        
                        # print(count, end=' ')

                

                # print(value_10[sid], value_15[sid], value_20[sid], value_25[sid], value_50[sid])
            

            
                cai, p = stats.friedmanchisquare(value_10, value_15, value_20, value_25, value_50)

                print("{}, {} results at dim{} is".format(precision_metric, ap_name, dim))
                print(cai, p)

                df = pd.DataFrame({
                    '10*D': value_10,
                    '15*D': value_15,
                    '20*D': value_20,
                    '25*D': value_25,
                    '50*D': value_50
                })

                df_melt = pd.melt(df)

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.set_xlabel('sampling size')
                ax.set_ylabel('percentage')
                # if precision_metric == 'sbs_or_over':
                #     ax.set_ylim(0.7, 1)
                # elif ap_name == 'ls2':
                #     ax.set_ylim(0.5, 1)
                sns.boxplot(x='variable', y='value', data=df_melt, showfliers=False, ax=ax)
                sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax=ax)
                plt.savefig("./plt/Friedman/{}_DIM{}_{}_p{:.4f}.png".format(precision_metric, dim, ap_name, p))


                plt.clf()
                bins = np.linspace(min(value_10+value_50),max(value_10+value_50),10)
                plt.hist(value_10,bins,alpha=0.5,color="red")
                plt.hist(value_50,bins,alpha=0.5,color="blue")
                statistic, p = stats.mannwhitneyu(value_10, value_50)
                superior = sorted(value_10)[15] > sorted(value_50)[15]
                plt.savefig("./plt/Wilcoxon/{}_DIM{}_{}_p{:.4f}_{}.png".format(precision_metric, dim, ap_name, p, superior))


            

                


def run():
    # ela_feature_classes = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    # ela_feature_classes = 'basic_ela_distr_pca_limo_ic_disp_ela_meta'
    dims = 'dims2_3_5_10'
    sampling_method = 'lhs'
    # sample_multiplier = 50
    per_metric = 'RMSE'
    # per_metric = 'accuracy'
    feature_selector = 'none'
    n_features_to_select = 0
    
    
    

    precision_datarframize(dims, sampling_method, per_metric, feature_selector, n_features_to_select)
    Friedman(dims, sampling_method, per_metric, feature_selector, n_features_to_select)

    # Friedman()






if __name__ == '__main__':
    
    

    # for sample_multiplier in [10, 15, 20, 25, 50]:
    #     # for ap_name in ['kt', 'dlvat', 'jped', 'bmtp', 'mk', 'ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']:
    #     for ap_name in ['ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']:
    #         # for pre_solver in [None, 'slsqp_multiplier50', 'smac_multiplier50']:
    #         for pre_solver in [None]:
    #             for sid in range(0, 31):
    #                 dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
    #                 table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims
    #                 # for selector in ['multiclass_classification', 'hiearchical_regression', 'clustering', 'pairwise_regression', 'pairwise_classification']:
    #                 for selector in ['hiearchical_regression']:
    #                     # for cross_valid_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
    #                     for cross_valid_type in ['lopo_cv']:
    #                         as_result_dir_path = os.path.join('as_results', '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector))
    #                         if feature_selector != 'none':
    #                             as_result_dir_path += '_nfs{}'.format(n_features_to_select)    
    #                         run(as_result_dir_path, pre_solver=pre_solver)
    #                         logger.info("Postprocessing of %s was done.", as_result_dir_path)

    run()