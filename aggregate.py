from codecs import ignore_errors
import numpy as np
import pandas as pd
import statistics as stat
import csv
import os
import sys
import logging
import click
# import random
import json
import shutil
# from fopt_info import bbob_fopt
# from statistics import mean, stdiv

from scipy import stats
# from statistics import mean, stdiv

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(filename='{}.log'.format(__file__), level=logging.DEBUG)



def precision_datarframize(multiplier_list, ap_list, dim_list, dims, sampling_method, per_metric, feature_selector, n_features_to_select, bbob_suite='bbob'):

    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'

    aggregate_dir_path = 'pp_aggregate'
    os.makedirs(aggregate_dir_path, exist_ok=True)

    all_id_funs = range(1, 24+1)
    if bbob_suite == 'bbob_noisy':
         all_id_funs = range(101, 130+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)


    # df_total = pd.DataFrame(columns=['setting', 'sbs_or_over', 'vbs_precision', 'vbs_or_over'])


    for sample_multiplier in multiplier_list:
        multiplier_path = os.path.join(aggregate_dir_path, 'multiplier{}'.format(sample_multiplier))
        os.makedirs(multiplier_path, exist_ok=True)

        if sample_multiplier == 10:
            ela_feature_classes = ela_feature_classes_10
        else:
            ela_feature_classes = ela_feature_classes_other

        
        # nbcを除外した実験のため変更
        ela_feature_classes = ela_feature_classes_10


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
                                
                                # df_ap = df_ap.append(dict_ap_fun, ignore_index=True, sort=False)
                                df_ap = pd.concat([df_ap, pd.DataFrame(dict_ap_fun, index=[0])], ignore_index=True, sort=False)
                            


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
                            
                            # df_ap = df_ap.append(dict_ap, ignore_index=True, sort=False)
                            df_ap = pd.concat([df_ap, pd.DataFrame(dict_ap, index=[0])], ignore_index=True, sort=False)
                            aggregate_ap_dim_dir_path = os.path.join(dim_path, '{}_DIM{}.csv'.format(ap_name, dim))
                            df_ap.to_csv(aggregate_ap_dim_dir_path, index=False)


                            # df_dim = df_dim.append(dict_ap, ignore_index=True, sort=False)
                            df_dim = pd.concat([df_dim, pd.DataFrame(dict_ap, index=[0])], ignore_index=True, sort=False)



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
                        
                        # df_dim = df_dim.append(dict_dim, ignore_index=True, sort=False)
                        df_dim = pd.concat([df_dim, pd.DataFrame(dict_dim, index=[0])], ignore_index=True, sort=False)
                        aggregate_dim_dir_path = os.path.join(dim_path, 'DIM{}.csv'.format(dim))
                        df_dim.to_csv(aggregate_dim_dir_path, index=False)


                        # df_multiplier = df_multiplier.append(dict_dim, ignore_index=True, sort=False)
                        df_multiplier = pd.concat([df_multiplier, pd.DataFrame(dict_dim, index=[0])], ignore_index=True, sort=False)

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
                        
        # df_multiplier = df_multiplier.append(dict_multiplier, ignore_index=True, sort=False)
        df_multiplier = pd.concat([df_multiplier, pd.DataFrame(dict_multiplier, index=[0])], ignore_index=True, sort=False)
        aggregate_multiplier_dir_path = os.path.join(multiplier_path, '{}.csv'.format(sample_multiplier))
        df_multiplier.to_csv(aggregate_multiplier_dir_path, index=False)


        logger.info("Aggregation of %s was done.", multiplier_path)



def Friedman(multiplier_list, ap_list, dim_list, dims, sampling_method, per_metric, feature_selector, n_features_to_select, bbob_suite = 'bbob'):

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

    

    all_sid = range(0, 31)


    dict_p = {}
    dict_Median = {}



    # for precision_metric in ['sbs_or_over', 'vbs_selecting_precision', 'vbs_or_over']:
    for precision_metric in ['vbs_selecting_precision']:

        df_p = pd.DataFrame(0, index=['dim2', 'dim3', 'dim5', 'dim10'], columns=['sampling10D', 'sampling15D', 'sampling20D', 'sampling25D', 'sampling50D'])
        df_Median = pd.DataFrame(False, index=['dim2', 'dim3', 'dim5', 'dim10'], columns=['sampling10D', 'sampling15D', 'sampling20D', 'sampling25D', 'sampling50D'])

        for dim in dim_list:

            for sample_multiplier in multiplier_list:
                
                if sample_multiplier == 10:
                    ela_feature_classes = ela_feature_classes_10
                else:
                    ela_feature_classes = ela_feature_classes_other

                # value_0 = [0]*len(all_sid)
                # value_10 = [0]*len(all_sid)
                # value_15 = [0]*len(all_sid)
                # value_20 = [0]*len(all_sid)
                # value_25 = [0]*len(all_sid)
                # value_50 = [0]*len(all_sid)

                value_1 = [0]*len(all_sid)
                value_2 = [0]*len(all_sid)

                for ap_name in ap_list:

                    if ap_name != 'ls0':
                        if ap_name != 'ls{}'.format(sample_multiplier):
                            continue

                    for sid in all_sid:

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



                        # if ap_name == 'ls0':
                        #     value_0[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        # elif ap_name == 'ls10':
                        #     value_10[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        # elif ap_name == 'ls15':
                        #     value_15[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        # elif ap_name == 'ls20':
                        #     value_20[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        # elif ap_name == 'ls25':
                        #     value_25[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        # elif ap_name == 'ls50':
                        #     value_50[sid] = count / (len(all_id_funs)*len(all_test_instance_id))

                        if ap_name == 'ls0':
                             value_1[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        else:
                             value_2[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        
                        # print(count, end=' ')

                

                        # print(value_10[sid], value_15[sid], value_20[sid], value_25[sid], value_50[sid])
            
            
                    # cai, p = stats.friedmanchisquare(value_10, value_15, value_20, value_25, value_50)

                    # print("{}, {} results at dim{} is".format(precision_metric, ap_name, dim))
                    # print(cai, p)

                    # df = pd.DataFrame({
                    #     '10*D': value_10,
                    #     '15*D': value_15,
                    #     '20*D': value_20,
                    #     '25*D': value_25,
                    #     '50*D': value_50
                    # })

                    # df_melt = pd.melt(df)

                    if ap_name == 'ls0':
                        continue

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.set_xlabel('percentage')
                ax.set_ylabel('number of trial')
                # if precision_metric == 'sbs_or_over':
                #     ax.set_ylim(0.7, 1)
                # elif ap_name == 'ls2':
                #     ax.set_ylim(0.5, 1)


                # フリードマン検定
                # sns.boxplot(x='variable', y='value', data=df_melt, showfliers=False, ax=ax)
                # sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax=ax)

                # Friedman_path = './plt/Friedman'
                # os.makedirs(Friedman_path, exist_ok=True)
                # Friedman_file_path = os.path.join(Friedman_path, '{}_DIM{}_{}_p{:.4f}.png'.format(precision_metric, dim, ap_name, p))
                # plt.savefig(Friedman_file_path)

                
                # plt.clf()
                bins = np.linspace(min(value_1+value_2),max(value_1+value_2),10)
                plt.hist(value_1,bins,alpha=0.5,color="blue")
                plt.hist(value_2,bins,alpha=0.5,color="red")
                statistic, p = stats.mannwhitneyu(value_1, value_2)
                superior = sorted(value_1)[15] > sorted(value_2)[15]

                Wilcoxon_path = './plt/Wilcoxon'
                os.makedirs(Wilcoxon_path, exist_ok=True)
                Wilcoxon_file_path = os.path.join(Wilcoxon_path, '{}_DIM{}_sample{}D_p{:.4f}_{}.png'.format(precision_metric, dim, sample_multiplier, p, superior))
                plt.savefig(Wilcoxon_file_path)

                df_p.at['dim{}'.format(dim), 'sampling{}D'.format(sample_multiplier)] = p
                df_Median.at['dim{}'.format(dim), 'sampling{}D'.format(sample_multiplier)] = superior
    
        dict_p[precision_metric] = df_p
        dict_Median[precision_metric] = df_Median
    
    return dict_p, dict_Median



def aggregate_error(multiplier_list, ap_list, dim_list, dims, sampling_method, per_metric, feature_selector, bbob_suite = 'bbob'):
    
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'


    all_id_funs = range(1, 24+1)
    if bbob_suite == 'bbob_noisy':
         all_id_funs = range(101, 130+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    for precision_metric in ['selected_alg_error']:

        dict_error_0 = {}
        dict_error_other = {}

        # for dim in dim_list:

        for sample_multiplier in multiplier_list:
            
            if sample_multiplier == 10:
                ela_feature_classes = ela_feature_classes_10
            else:
                ela_feature_classes = ela_feature_classes_other
            

            # nbcを除外した実験のため変更
            ela_feature_classes = ela_feature_classes_10


            value_1 = [0]*len(all_sid)
            value_2 = [0]*len(all_sid)

            for ap_name in ap_list:

                if ap_name != 'ls0':
                    if ap_name != 'ls{}'.format(sample_multiplier):
                        continue

                for sid in all_sid:

                    count = 0

                    for pre_solver in [None]:
                        for selector in ['hiearchical_regression']:
                            for cross_valid_type in ['lopo_cv']:

                                for dim in dim_list:

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
                        
                    if ap_name == 'ls0':
                        value_1[sid] = count
                    else:
                        value_2[sid] = count
                    

                if ap_name == 'ls0':
                    continue


            dict_error_0[sample_multiplier] = sum(value_1)/len(value_1)
            dict_error_other[sample_multiplier] = sum(value_2)/len(value_2)

        print(r'%%%%%%%%%%%')
        if precision_metric == 'selected_alg_error':
            print(r'\subfloat[最良解と大域的最適解との誤差の合計の平均]{')
        print(r'\begin{tabular}{l|c|c}')
        print(r'\toprule')
        print(r' サンプリングサイズ & $\mathcal{A}_0$ & $\mathcal{A}_{sample}$\\')
        print(r'\midrule')

        ls0_superior_dict = {}
        # minimum_score = 1e10

        for sample_multiplier in [10, 15, 20, 25, 50]:
            value_0 = dict_error_0[sample_multiplier]
            value_other = dict_error_other[sample_multiplier]


            if value_0 > value_other:
                print('${}D$ & ${:.4f}$ & '.format(sample_multiplier, value_0) +r'$\mathbf{'+'{:.4f}'.format(value_other)+r'}$ \\')
            elif value_0 < value_other:
                print('${}D$ & '.format(sample_multiplier)+r'$\mathbf{'+'{:.4f}'.format(value_0) +r'}$' + ' & ${:.4f}$'.format(value_other)+r'\\')
            else:
                print('${}D$ & {:.4f}$ & ${:.4f}$'.format(sample_multiplier, value_0, value_other)+r'\\')

            # if minimum_score > value_0:
            #     minimum_score = value_0
            #     minimum_score_ap = 'ls0'
            #     minimum_score_multiplier = sample_multiplier
            # if minimum_score > value_other:
            #     minimum_score = value_other
            #     minimum_score_ap = 'ls0'
            #     minimum_score_multiplier = sample_multiplier
            
            ls0_superior_dict[sample_multiplier] = value_0 < value_other

        print(r'\bottomrule')
        print(r'\end{tabular}}\\')
        print(r'%%%%%%%%%%%')
    
    return ls0_superior_dict
    # return minimum_score_ap, minimum_score_multiplier


def aggregate_tablize():
    aggregate_dir_path = 'pp_aggregate'

    for metric in ['soo_percentage', 'v_p_percentage']:

        for ap_name in ['ls2','ls4','ls6','ls8','ls10','ls12','ls14','ls16','ls18']:
            # print('table with {},{}'.format(metric, ap_name))
            
            print(r'%%%%%%%%%%%')
            if metric == 'soo_percentage' and ap_name == 'ls2':
                print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合($\mathcal{A}_2$)]{')
            elif metric == 'soo_percentage' and ap_name == 'ls4':
                print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合($\mathcal{A}_4$)]{')
            elif metric == 'soo_percentage' and ap_name == 'ls6':
                print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合($\mathcal{A}_6$)]{')
            elif metric == 'soo_percentage' and ap_name == 'ls8':
                print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合($\mathcal{A}_8$)]{')
            elif metric == 'soo_percentage' and ap_name == 'ls10':
                print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合($\mathcal{A}_10$)]{')
            elif metric == 'soo_percentage' and ap_name == 'ls12':
                print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合($\mathcal{A}_12$)]{')
            elif metric == 'soo_percentage' and ap_name == 'ls14':
                print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合($\mathcal{A}_14$)]{')
            elif metric == 'soo_percentage' and ap_name == 'ls16':
                print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合($\mathcal{A}_16$)]{')
            elif metric == 'soo_percentage' and ap_name == 'ls18':
                print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合($\mathcal{A}_18$)]{')
            elif metric == 'v_p_percentage' and ap_name == 'ls2':
                print(r'\subfloat[選択されたアルゴリズムがVBSと一致した割合($\mathcal{A}_2$)]{')
            elif metric == 'v_p_percentage' and ap_name == 'ls4':
                print(r'\subfloat[選択されたアルゴリズムがVBSと一致した割合($\mathcal{A}_4$)]{')
            elif metric == 'v_p_percentage' and ap_name == 'ls6':
                print(r'\subfloat[選択されたアルゴリズムがVBSと一致した割合($\mathcal{A}_6$)]{')
            elif metric == 'v_p_percentage' and ap_name == 'ls8':
                print(r'\subfloat[選択されたアルゴリズムがVBSと一致した割合($\mathcal{A}_8$)]{')
            elif metric == 'v_p_percentage' and ap_name == 'ls10':
                print(r'\subfloat[選択されたアルゴリズムがVBSと一致した割合($\mathcal{A}_10$)]{')
            elif metric == 'v_p_percentage' and ap_name == 'ls12':
                print(r'\subfloat[選択されたアルゴリズムがVBSと一致した割合($\mathcal{A}_12$)]{')
            elif metric == 'v_p_percentage' and ap_name == 'ls14':
                print(r'\subfloat[選択されたアルゴリズムがVBSと一致した割合($\mathcal{A}_14$)]{')
            elif metric == 'v_p_percentage' and ap_name == 'ls16':
                print(r'\subfloat[選択されたアルゴリズムがVBSと一致した割合($\mathcal{A}_16$)]{')
            elif metric == 'v_p_percentage' and ap_name == 'ls18':
                print(r'\subfloat[選択されたアルゴリズムがVBSと一致した割合($\mathcal{A}_18$)]{')
            print(r'\begin{tabular}{lcccc}')
            print(r'\toprule')
            print(r'サンプリング数 & DIM2 & DIM3 & DIM5 & DIM10\\')
            print(r'\midrule')

            for sample_multiplier in [10, 15, 20, 25, 50]:
                for dim in [2, 3, 5, 10]:
                    file_dir = os.path.join(aggregate_dir_path, 'multiplier{}/DIM{}/{}_DIM{}.csv'.format(sample_multiplier, dim, ap_name, dim))

                    data_df = pd.read_csv(file_dir, index_col=0)


                    if dim == 2:
                        value_2 = data_df.at[ap_name, metric]
                    elif dim == 3:
                        value_3 = data_df.at[ap_name, metric]
                    elif dim == 5:
                        value_5 = data_df.at[ap_name, metric]
                    elif dim == 10:
                        value_10 = data_df.at[ap_name, metric]
                

                print('${}*D$ & {:.4f} & {:.4f} & {:.4f} & {:.4f}'.format(sample_multiplier, value_2, value_3, value_5, value_10)+r'\\')

            print(r'\bottomrule')
            print(r'\end{tabular}}\\')
            print(r'%%%%%%%%%%%')
                



def aggregate_colored_tablize(dict_p, dict_Median):

    # for precision_metric in ['sbs_or_over', 'vbs_selecting_precision', 'vbs_or_over']:
    # for precision_metric in ['vbs_selecting_precision', 'vbs_or_over']:
    for precision_metric in ['vbs_selecting_precision']:
            
        print(r'%%%%%%%%%%%')
        if precision_metric == 'sbs_or_over':
             print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合についてのp値]{')
        elif precision_metric == 'vbs_selecting_precision':
            print(r'\subfloat[選択されたアルゴリズムがVBSと一致した割合についてのp値]{')
        elif precision_metric == 'vbs_or_over':
            print(r'\subfloat[選択されたアルゴリズムがVBSと同じ性能を示した割合についてのp値]{')
        elif precision_metric == 'selected_alg_error':
            print(r'\subfloat[選択されたアルゴリズムの最良解と, 大域的最適解との誤差の平均についてのp値]{')
        # elif precision_metric == 'vbs_or_over':
        #     print(r'\subfloat[選択されたアルゴリズムがVBSと同じ性能を示した割合についてのp値]{')
        # elif precision_metric == 'vbs_or_over':
        #     print(r'\subfloat[選択されたアルゴリズムがVBSと同じ性能を示した割合についてのp値]{')
        print(r'\begin{tabularx}{60em}{|A|C|C|C|C|C|}')
        print(r'\hline')
        print(r'\rowcolor[rgb]{0.9, 0.9, 0.9} & sampling10D & sampling15D & sampling20D & sampling25D & sampling50D \\')
        print(r'\hline')

        for dim in [2, 3, 5, 10]:
            print('dim${}$ & '.format(dim), end='')
            for sample_multiplier in [10, 15, 20, 25, 50]:
                Median = dict_Median[precision_metric].at['dim{}'.format(dim), 'sampling{}D'.format(sample_multiplier)]
                p = dict_p[precision_metric].at['dim{}'.format(dim), 'sampling{}D'.format(sample_multiplier)]
                if p < 0.05:
                    if Median:
                        print(r'\cellcolor{c2}',end='')
                    else:
                        print(r'\cellcolor{c3}',end='')
                else:
                    print(r'\cellcolor{c1}',end='')
                
                if sample_multiplier == 50:
                    print('{:.4f} '.format(p), end='')
                else:
                    print('{:.4f} & '.format(p), end='')
            
            print(r'\\')

        print(r'\hline')
        print(r'\end{tabularx}}\\')
        print(r'%%%%%%%%%%%')


def sbs_tablize(ls0_superior_dict, multiplier_list, dim_list, sampling_method, dims, per_metric, feature_selector, bbob_suite = 'bbob'):
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'

    aggregate_dir_path = 'pp_aggregate'
    os.makedirs(aggregate_dir_path, exist_ok=True)

    all_id_funs = range(1, 24+1)
    if bbob_suite == 'bbob_noisy':
         all_id_funs = range(101, 130+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    df_sbs = pd.DataFrame(0, index=['dim2', 'dim3', 'dim5', 'dim10'], columns=['sampling10D', 'sampling15D', 'sampling20D', 'sampling25D', 'sampling50D'])

    for sample_multiplier in multiplier_list:
        if sample_multiplier == 10:
            ela_feature_classes = ela_feature_classes_10
        else:
            ela_feature_classes = ela_feature_classes_other


        # nbcを除外した実験のため変更
        ela_feature_classes = ela_feature_classes_10


        # df_ap = pd.DataFrame(0, index=['dim2', 'dim3', 'dim5', 'dim10'], columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24'])

        if ls0_superior_dict[sample_multiplier]:
            ap = 'ls0'
        else:
            ap = 'ls{}'.format(sample_multiplier)


        for dim in dim_list:

            for pre_solver in [None]:
                for selector in ['hiearchical_regression']:
                    for cross_valid_type in ['lopo_cv']:

                        sbs_or_over_summary = 0

                        for fun_id in all_id_funs:

                            for sid in all_sid:
                                dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                pp_as_result_dir_path = os.path.join('pp_as_results', '{}_{}_{}_{}_{}'.format(ap, selector, cross_valid_type, table_data_name, feature_selector))
                                
                                csv_file_path = os.path.join(pp_as_result_dir_path, 'f{}_DIM{}.csv'.format(fun_id, dim))
                                
                                sbs_or_over_count = 0

                                df_csv_file = pd.read_csv(csv_file_path, dtype={'selected_alg_error':float, 
                                                                                'sbs_error':float, 
                                                                                'vbs_alg_error':float,
                                                                                'sbs_or_over':bool,
                                                                                'vbs_selecting_precision':bool,
                                                                                'vbs_or_over':bool
                                                                                })

                                sbs_or_over_count = sum(df_csv_file['sbs_or_over'])

                                sbs_or_over_summary += sbs_or_over_count

                        df_sbs.at['dim{}'.format(dim), 'sampling{}D'.format(sample_multiplier)] = sbs_or_over_summary / (len(all_sid)*len(all_test_instance_id)*len(all_id_funs))
                
                        logger.info("Aggregation of %dD, %d was done.", sample_multiplier, dim)
    

    print(r'%%%%%%%%%%%')
    print(r'\subfloat[選択されたアルゴリズムがSBS以上の性能を示した割合]{')
    print(r'\begin{tabularx}{60em}{|A|C|C|C|C|C|}')
    print(r'\hline')
    print(r'\rowcolor[rgb]{0.9, 0.9, 0.9} & ', end='')
    for sample_multiplier in multiplier_list:
        if ls0_superior_dict[sample_multiplier]:
            print('sampling{}D'.format(sample_multiplier) + r'($\mathcal{A}_0$)', end='')
        else:
            print('sampling{}D'.format(sample_multiplier) + r'($\mathcal{A}_{' + '{}'.format(sample_multiplier) + r'}$)', end='')
        
        if sample_multiplier != multiplier_list[-1]:
            print(' & ', end='')
            
    print(r'\\')
    print(r'\hline')

    for dim in dim_list:
        print('dim${}$ & '.format(dim), end='')
        for sample_multiplier in multiplier_list:
            if df_sbs.loc['dim{}'.format(dim)].idxmax() == 'sampling{}D'.format(sample_multiplier):
                print(r'\cellcolor{c2}'+'{:.4f}'.format(df_sbs.at['dim{}'.format(dim), 'sampling{}D'.format(sample_multiplier)]), end='')
            else:
                print(r'\cellcolor{c1}'+'{:.4f}'.format(df_sbs.at['dim{}'.format(dim), 'sampling{}D'.format(sample_multiplier)]), end='')

            if sample_multiplier != multiplier_list[-1]:
                print(' & ', end='')
        
        print(r'\\')
        
    print(r'\hline')
    print(r'\end{tabularx}}\\')


# 先生にCSVを提出するためのコード.
def aggregate_tocsv(dim_list, multiplier_list, ap_list, feature_selector, sampling_method, per_metric, dims):
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'



    all_id_funs = range(1, 24+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    csv_path = 'aggregate_csv'
    os.makedirs(csv_path, exist_ok=True)


    for dim in dim_list:
        df_dim = pd.DataFrame(0, index=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24'], columns=['A0-10', 'A0-15', 'A0-20', 'A0-25', 'A0-50', 'A10', 'A15', 'A20', 'A25', 'A50'])

        for fun_id in all_id_funs:
            for ap_name in ap_list:
                for sample_multiplier in multiplier_list:

                    if sample_multiplier == 10:
                        ela_feature_classes = ela_feature_classes_10
                    else:
                        ela_feature_classes = ela_feature_classes_other

                    if ap_name != 'ls0':
                        if ap_name != 'ls{}'.format(sample_multiplier):
                            continue

                    for selector in ['hiearchical_regression']:
                        for cross_valid_type in ['lopo_cv']:

                            error_summary = 0

                            for sid in all_sid:

                                dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                pp_as_path = os.path.join('pp_as_results', '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))

                                df_csv_file = pd.read_csv(pp_as_path, dtype={'selected_alg_error':float, 
                                                                                'sbs_error':float, 
                                                                                'vbs_alg_error':float,
                                                                                'sbs_or_over':bool,
                                                                                'vbs_selecting_precision':bool,
                                                                                'vbs_or_over':bool
                                                                                })
                                

                                error_summary += sum(df_csv_file['selected_alg_error'])
                            

                            if ap_name == 'ls0':
                                df_dim.at['f{}'.format(fun_id), 'A0-{}'.format(sample_multiplier)] -= error_summary / len(all_sid)
                            elif ap_name == 'ls10':
                                df_dim.at['f{}'.format(fun_id), 'A10'] -= error_summary / len(all_sid)
                            elif ap_name == 'ls15':
                                df_dim.at['f{}'.format(fun_id), 'A15'] -= error_summary / len(all_sid)
                            elif ap_name == 'ls20':
                                df_dim.at['f{}'.format(fun_id), 'A20'] -= error_summary / len(all_sid)
                            elif ap_name == 'ls25':
                                df_dim.at['f{}'.format(fun_id), 'A25'] -= error_summary / len(all_sid)
                            elif ap_name == 'ls50':
                                df_dim.at['f{}'.format(fun_id), 'A50'] -= error_summary / len(all_sid)



        tocsv_path = os.path.join(csv_path, 'dim{}.csv'.format(dim))
        df_dim.to_csv(tocsv_path)


# サロゲートXと比較用
def aggregate_tocsv2(surrogate_multiplier, surrogate_sample, dim_list, multiplier_list, ap_list, feature_selector, sampling_method, per_metric, dims):
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'



    all_id_funs = range(1, 24+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    csv_path = 'aggregate_csv'
    os.makedirs(csv_path, exist_ok=True)

    df = pd.DataFrame(0, index=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24'], columns=['A10-0', 'A15-0', 'A20-0', 'A25-0', 'A50-0', 'A10-{}n'.format(surrogate_multiplier), 'A15-{}n'.format(surrogate_multiplier), 'A20-{}n'.format(surrogate_multiplier), 'A25-{}n'.format(surrogate_multiplier), 'A50-{}n'.format(surrogate_multiplier)])

    for directory in ['0', '{}n'.format(surrogate_multiplier)]:
        
        for dim in dim_list:
            
            for fun_id in all_id_funs:
                for ap_name in ap_list:
                    for sample_multiplier in multiplier_list:

                        if sample_multiplier == 10:
                            ela_feature_classes = ela_feature_classes_10
                        else:
                            ela_feature_classes = ela_feature_classes_other

                        if ap_name != 'ls0':
                            if ap_name != 'ls{}'.format(sample_multiplier):
                                continue

                        for selector in ['hiearchical_regression']:
                            for cross_valid_type in ['lopo_cv']:

                                error_summary = 0

                                for sid in all_sid:

                                    dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                    table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                    if directory == '0':
                                        pp_as_path = os.path.join('pp_as_results', directory, '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))
                                    else:
                                        pp_as_path = os.path.join('pp_as_results', directory+'_{}'.format(surrogate_sample), '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))

                                    df_csv_file = pd.read_csv(pp_as_path, dtype={'selected_alg_error':float, 
                                                                                    'sbs_error':float, 
                                                                                    'vbs_alg_error':float,
                                                                                    'sbs_or_over':bool,
                                                                                    'vbs_selecting_precision':bool,
                                                                                    'vbs_or_over':bool
                                                                                    })
                                    

                                    error_summary += sum(df_csv_file['selected_alg_error'])
                                

                                if ap_name == 'ls0':
                                    df.at['f{}'.format(fun_id), 'A0-{}'.format(sample_multiplier)] -= error_summary / len(all_sid)
                                elif ap_name == 'ls10':
                                    df.at['f{}'.format(fun_id), 'A10-{}'.format(directory)] -= error_summary / len(all_sid)
                                elif ap_name == 'ls15':
                                    df.at['f{}'.format(fun_id), 'A15-{}'.format(directory)] -= error_summary / len(all_sid)
                                elif ap_name == 'ls20':
                                    df.at['f{}'.format(fun_id), 'A20-{}'.format(directory)] -= error_summary / len(all_sid)
                                elif ap_name == 'ls25':
                                    df.at['f{}'.format(fun_id), 'A25-{}'.format(directory)] -= error_summary / len(all_sid)
                                elif ap_name == 'ls50':
                                    df.at['f{}'.format(fun_id), 'A50-{}'.format(directory)] -= error_summary / len(all_sid)
                                



            tocsv_path = os.path.join(csv_path, 'dim{}.csv'.format(dim))
            df.to_csv(tocsv_path)


# サロゲートXと比較用
def aggregate_tocsv3(surrogate_multiplier1, surrogate_sample1, surrogate_multiplier2, surrogate_sample2, dim_list, multiplier_list, ap_list, feature_selector, sampling_method, per_metric, dims):
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'



    all_id_funs = range(1, 24+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    csv_path = 'aggregate_csv'
    os.makedirs(csv_path, exist_ok=True)

    df = pd.DataFrame(0, index=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24'], 
                      columns=['A10-{}n_{}'.format(surrogate_multiplier1, surrogate_sample1), 'A15-{}n_{}'.format(surrogate_multiplier1, surrogate_sample1), 'A20-{}n_{}'.format(surrogate_multiplier1, surrogate_sample1), 'A25-{}n_{}'.format(surrogate_multiplier1, surrogate_sample1), 'A50-{}n_{}'.format(surrogate_multiplier1, surrogate_sample1), 
                               'A10-{}n_{}'.format(surrogate_multiplier2, surrogate_sample2), 'A15-{}n_{}'.format(surrogate_multiplier2, surrogate_sample2), 'A20-{}n_{}'.format(surrogate_multiplier2, surrogate_sample2), 'A25-{}n_{}'.format(surrogate_multiplier2, surrogate_sample2), 'A50-{}n_{}'.format(surrogate_multiplier2, surrogate_sample2)])

    for directory in ['{}n_{}'.format(surrogate_multiplier1, surrogate_sample1), '{}n_{}'.format(surrogate_multiplier2, surrogate_sample2)]:
        
        for dim in dim_list:
            
            for fun_id in all_id_funs:
                for ap_name in ap_list:
                    for sample_multiplier in multiplier_list:

                        if sample_multiplier == 10:
                            ela_feature_classes = ela_feature_classes_10
                        else:
                            ela_feature_classes = ela_feature_classes_other

                        if ap_name != 'ls0':
                            if ap_name != 'ls{}'.format(sample_multiplier):
                                continue

                        for selector in ['hiearchical_regression']:
                            for cross_valid_type in ['lopo_cv']:

                                error_summary = 0

                                for sid in all_sid:

                                    dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                    table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                    
                                    pp_as_path = os.path.join('pp_as_results', directory, '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))

                                    df_csv_file = pd.read_csv(pp_as_path, dtype={'selected_alg_error':float, 
                                                                                    'sbs_error':float, 
                                                                                    'vbs_alg_error':float,
                                                                                    'sbs_or_over':bool,
                                                                                    'vbs_selecting_precision':bool,
                                                                                    'vbs_or_over':bool
                                                                                    })
                                    

                                    error_summary += sum(df_csv_file['selected_alg_error'])
                                

                                if ap_name == 'ls0':
                                    df.at['f{}'.format(fun_id), 'A0-{}'.format(sample_multiplier)] -= error_summary / len(all_sid)
                                elif ap_name == 'ls10':
                                    df.at['f{}'.format(fun_id), 'A10-{}'.format(directory)] -= error_summary / len(all_sid)
                                elif ap_name == 'ls15':
                                    df.at['f{}'.format(fun_id), 'A15-{}'.format(directory)] -= error_summary / len(all_sid)
                                elif ap_name == 'ls20':
                                    df.at['f{}'.format(fun_id), 'A20-{}'.format(directory)] -= error_summary / len(all_sid)
                                elif ap_name == 'ls25':
                                    df.at['f{}'.format(fun_id), 'A25-{}'.format(directory)] -= error_summary / len(all_sid)
                                elif ap_name == 'ls50':
                                    df.at['f{}'.format(fun_id), 'A50-{}'.format(directory)] -= error_summary / len(all_sid)


            tocsv_path = os.path.join(csv_path, 'dim{}.csv'.format(dim))
            df.to_csv(tocsv_path)


def aggregate_tocsv_allsys(all_systems, dim_list, feature_selector, sampling_method, per_metric, dims, multiplier_list):
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'

    all_id_funs = range(1, 24+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    
    for sample_multiplier in multiplier_list:
        if sample_multiplier == 10:
            ela_feature_classes = ela_feature_classes_10
        else:
            ela_feature_classes = ela_feature_classes_other
    # for dim in dim_list:
        # if sample_multiplier == 10:
        #     ela_feature_classes = ela_feature_classes_10
        # else:
        #     ela_feature_classes = ela_feature_classes_other


        csv_path = os.path.join('./plotallsys_csv', 'multiplier{}'.format(sample_multiplier))
        # csv_path = os.path.join('./plotallsys_csv', 'dim{}'.format(dim))

        os.makedirs(csv_path, exist_ok=True)

        df = pd.DataFrame(0, index=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24'], 
                        columns=all_systems)
            
        for dim in dim_list:
            tocsv_path = os.path.join(csv_path, 'dim{}.csv'.format(dim))
        # for sample_multiplier in multiplier_list:
        #     tocsv_path = os.path.join(csv_path, 'multiplier{}.csv'.format(sample_multiplier))
            if os.path.exists(tocsv_path):
                continue

            if sample_multiplier == 10:
                ela_feature_classes = ela_feature_classes_10
            else:
                ela_feature_classes = ela_feature_classes_other

            for directory in all_systems:

                for fun_id in all_id_funs:

                    for selector in ['hiearchical_regression']:
                        for cross_valid_type in ['lopo_cv']:

                            error_summary = 0

                            for sid in all_sid:

                                dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                
                                pp_as_path = os.path.join('pp_as_results', directory, 'ls{}_{}_{}_{}_{}'.format(sample_multiplier, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))

                                df_csv_file = pd.read_csv(pp_as_path, dtype={'selected_alg_error':float, 
                                                                                'sbs_error':float, 
                                                                                'vbs_alg_error':float,
                                                                                'sbs_or_over':bool,
                                                                                'vbs_selecting_precision':bool,
                                                                                'vbs_or_over':bool
                                                                                })
                                

                                error_summary += sum(df_csv_file['selected_alg_error'])
                            
                            
                            df.at['f{}'.format(fun_id), '{}'.format(directory)] -= error_summary / len(all_sid)
                            print("f{}, {} is {}".format(fun_id, directory, df.at['f{}'.format(fun_id), '{}'.format(directory)]))

            df.to_csv(tocsv_path)

def aggregate_boxplot_vbs(dim_list, multiplier_list, ap_list, sampling_method, per_metric, dims, feature_selector, bbob_suite='bbob'):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set3')
    
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'



    all_id_funs = range(1, 24+1)
    if bbob_suite == 'bbob_noisy':
         all_id_funs = range(101, 130+1)

    all_test_instance_id = range(1, 5+1)

    

    all_sid = range(0, 31)

    # for precision_metric in ['sbs_or_over', 'vbs_selecting_precision', 'vbs_or_over']:
    for precision_metric in ['vbs_selecting_precision']:

        dictls0 = {}
        dictls10 = {}
        dictls15 = {}
        dictls20 = {}
        dictls25 = {}
        dictls50 = {}

        for dim in dim_list:

            value_0 = [0]*len(all_sid)
            value_10 = [0]*len(all_sid)
            value_15 = [0]*len(all_sid)
            value_20 = [0]*len(all_sid)
            value_25 = [0]*len(all_sid)
            value_50 = [0]*len(all_sid)



            for sample_multiplier in multiplier_list:

                
                if sample_multiplier == 10:
                    ela_feature_classes = ela_feature_classes_10
                else:
                    ela_feature_classes = ela_feature_classes_other

                
                for ap_name in ap_list:

                    if ap_name != 'ls0':
                        if ap_name != 'ls{}'.format(sample_multiplier):
                            continue
                    else:
                        if sample_multiplier != 25:
                            continue


                    for sid in all_sid:

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



                        if ap_name == 'ls0':
                            value_0[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif ap_name == 'ls10':
                            value_10[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif ap_name == 'ls15':
                            value_15[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif ap_name == 'ls20':
                            value_20[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif ap_name == 'ls25':
                            value_25[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif ap_name == 'ls50':
                            value_50[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        
                        


            dictls0['{}'.format(dim)] = value_0
            dictls10['{}'.format(dim)] = value_10
            dictls15['{}'.format(dim)] = value_15
            dictls20['{}'.format(dim)] = value_20
            dictls25['{}'.format(dim)] = value_25
            dictls50['{}'.format(dim)] = value_50



        df_0 = pd.DataFrame(dictls0)
        df_10 = pd.DataFrame(dictls10)
        df_15 = pd.DataFrame(dictls15)
        df_20 = pd.DataFrame(dictls20)
        df_25 = pd.DataFrame(dictls25)
        df_50 = pd.DataFrame(dictls50)
        

        df_0_melt = pd.melt(df_0)
        df_0_melt['portfolio'] = r'$\mathcal{A}_{0}$'
        df_10_melt = pd.melt(df_10)
        df_10_melt['portfolio'] = r'$\mathcal{A}_{10}$'
        df_15_melt = pd.melt(df_15)
        df_15_melt['portfolio'] = r'$\mathcal{A}_{15}$'
        df_20_melt = pd.melt(df_20)
        df_20_melt['portfolio'] = r'$\mathcal{A}_{20}$'
        df_25_melt = pd.melt(df_25)
        df_25_melt['portfolio'] = r'$\mathcal{A}_{25}$'
        df_50_melt = pd.melt(df_50)
        df_50_melt['portfolio'] = r'$\mathcal{A}_{50}$'

        df = pd.concat([df_0_melt, df_10_melt, df_15_melt, df_20_melt, df_25_melt, df_50_melt], axis=0)


        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111)
        sns.boxplot(x='variable', y='value', data=df, hue='portfolio', palette='Dark2', ax=ax)


        ax.set_xlabel('Dimension', fontsize=55)
        ax.set_ylabel('Percentage of VBS selected in 31runs', fontsize=55)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02,), borderaxespad=0, fontsize=33, ncol=3)
        ax.tick_params(labelsize=55)

        

        boxplot_path = './plt/boxplot'
        os.makedirs(boxplot_path, exist_ok=True)
        boxplot_file_path = os.path.join(boxplot_path, '{}.png'.format(precision_metric))
        plt.savefig(boxplot_file_path)

        boxplot_file_path = os.path.join(boxplot_path, '{}.pdf'.format(precision_metric))
        pp = PdfPages(boxplot_file_path)
        pp.savefig(fig)
        pp.close()


# サロゲートとの比較用
def aggregate_boxplot_vbs2(dim_list, multiplier_list, ap_list, sampling_method, per_metric, dims, feature_selector, bbob_suite='bbob'):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set3')
    
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'
    

    sample_list = ['0', '100n_lhs', '200n_lhs', '300n_lhs', '100n_random', '300n_random', '300n_random']


    all_id_funs = range(1, 24+1)
    if bbob_suite == 'bbob_noisy':
         all_id_funs = range(101, 130+1)

    all_test_instance_id = range(1, 5+1)

    

    all_sid = range(0, 31)

    for ap_name in ap_list:
        target = 'ls'
        idx = ap_name.find(target)
        ap_option = ap_name[idx+len(target):]
        
        # for precision_metric in ['sbs_or_over', 'vbs_selecting_precision', 'vbs_or_over']:
        for precision_metric in ['vbs_selecting_precision']:

            dict0 = {}
            dict100n_lhs = {}
            dict200n_lhs = {}
            dict300n_lhs = {}
            dict100n_random = {}
            dict200n_random = {}
            dict300n_random = {}

            for dim in dim_list:

                value_0 = [0]*len(all_sid)
                value_100n_lhs = [0]*len(all_sid)
                value_200n_lhs = [0]*len(all_sid)
                value_300n_lhs = [0]*len(all_sid)
                value_100n_random = [0]*len(all_sid)
                value_200n_random = [0]*len(all_sid)
                value_300n_random = [0]*len(all_sid)

                for sample_multiplier in multiplier_list:

                    
                    if sample_multiplier == 10:
                        ela_feature_classes = ela_feature_classes_10
                    else:
                        ela_feature_classes = ela_feature_classes_other

                    
                    for sample_option in sample_list:


                        # if ap_name != 'ls0':
                        #     if ap_name != 'ls{}'.format(sample_multiplier):
                        #         continue
                        # else:
                        #     if sample_multiplier != 25:
                        #         continue


                        for sid in all_sid:

                            count = 0

                            # for pre_solver in [None, 'slsqp_multiplier50', 'smac_multiplier50']:
                            for pre_solver in [None]:
                                for selector in ['hiearchical_regression']:
                                    # for cross_valid_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
                                    for cross_valid_type in ['lopo_cv']:



                                        for fun_id in all_id_funs:

                                            dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                            table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                            csv_file_path = os.path.join('pp_as_results', '{}'.format(sample_option), '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))


                                            df_csv_file = pd.read_csv(csv_file_path, dtype={'selected_alg_error':float, 
                                                                                            'sbs_error':float, 
                                                                                            'vbs_alg_error':float,
                                                                                            'sbs_or_over':bool,
                                                                                            'vbs_selecting_precision':bool,
                                                                                            'vbs_or_over':bool
                                                                                            })


                                            count += sum(df_csv_file[precision_metric])



                            if sample_option == '0':
                                value_0[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            elif sample_option == '100n_lhs':
                                value_100n_lhs[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            elif sample_option == '200n_lhs':
                                value_200n_lhs[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            elif sample_option == '300n_lhs':
                                value_300n_lhs[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            elif sample_option == '100n_random':
                                value_100n_random[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            elif sample_option == '200n_random':
                                value_200n_random[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            elif sample_option == '300n_random':
                                value_300n_random[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            
                            


                dict0['{}'.format(dim)] = value_0
                dict100n_lhs['{}'.format(dim)] = value_100n_lhs
                dict200n_lhs['{}'.format(dim)] = value_200n_lhs
                dict300n_lhs['{}'.format(dim)] = value_300n_lhs
                dict100n_random['{}'.format(dim)] = value_100n_random
                dict200n_random['{}'.format(dim)] = value_200n_random
                dict300n_random['{}'.format(dim)] = value_300n_random



            df_0 = pd.DataFrame(dict0)
            df_100n_lhs = pd.DataFrame(dict100n_lhs)
            df_200n_lhs = pd.DataFrame(dict200n_lhs)
            df_300n_lhs = pd.DataFrame(dict300n_lhs)
            df_100n_random = pd.DataFrame(dict100n_random)
            df_200n_random = pd.DataFrame(dict200n_random)
            df_300n_random = pd.DataFrame(dict300n_random)
            

            df_0_melt = pd.melt(df_0)
            df_0_melt['option'] = r'no surrogate'
            df_100n_lhs_melt = pd.melt(df_100n_lhs)
            df_100n_lhs_melt['option'] = r'$100n$\_lhs'
            df_200n_lhs_melt = pd.melt(df_200n_lhs)
            df_200n_lhs_melt['option'] = r'$200n$\_lhs'
            df_300n_lhs_melt = pd.melt(df_300n_lhs)
            df_300n_lhs_melt['option'] = r'$300n$\_lhs'
            df_100n_random_melt = pd.melt(df_100n_random)
            df_100n_random_melt['option'] = r'$100n$\_random'
            df_200n_random_melt = pd.melt(df_200n_random)
            df_200n_random_melt['option'] = r'$200n$\_random'
            df_300n_random_melt = pd.melt(df_300n_random)
            df_300n_random_melt['option'] = r'$300n$\_random'

            df = pd.concat([df_0_melt, df_100n_lhs_melt, df_200n_lhs_melt, df_300n_lhs_melt, df_100n_random_melt, df_200n_random_melt, df_300n_random_melt], axis=0)


            fig = plt.figure(figsize=(20, 15))
            ax = fig.add_subplot(111)
            sns.boxplot(x='variable', y='value', data=df, hue='option', palette='Dark2', ax=ax)


            ax.set_xlabel('Dimension', fontsize=55)
            ax.set_ylabel('Percentage of VBS selected in 31runs with '+r'$\mathcal{A}_{'+ap_option+'}$', fontsize=55)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02,), borderaxespad=0, fontsize=33, ncol=3)
            ax.tick_params(labelsize=55)

            

            boxplot_path = './plt/boxplot'
            os.makedirs(boxplot_path, exist_ok=True)
            boxplot_file_path = os.path.join(boxplot_path, '{}_{}.png'.format(precision_metric, ap_option))
            plt.savefig(boxplot_file_path)

            boxplot_file_path = os.path.join(boxplot_path, '{}_{}.png'.format(precision_metric, ap_option))
            pp = PdfPages(boxplot_file_path)
            pp.savefig(fig)
            pp.close()


# サロゲートとの比較用
def aggregate_boxplot_vbs2(dim_list, multiplier_list, ap_list, sampling_method, per_metric, dims, feature_selector, bbob_suite='bbob'):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set3')
    
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'
    

    sample_list = ['0', '100n_lhs', '200n_lhs', '100n_lhs_new', '200n_lhs_new']


    all_id_funs = range(1, 24+1)
    if bbob_suite == 'bbob_noisy':
         all_id_funs = range(101, 130+1)

    all_test_instance_id = range(1, 5+1)


    ### 注 解集合を拡張して得られた特徴量+拡張せずに得られた特徴量で計算したもの. そうでない場合はnoneにする.
    comp = 'comp'


    all_sid = range(0, 31)

    for ap_name in ap_list:
        target = 'ls'
        idx = ap_name.find(target)
        ap_option = ap_name[idx+len(target):]
        
        # for precision_metric in ['sbs_or_over', 'vbs_selecting_precision', 'vbs_or_over']:
        for precision_metric in ['vbs_selecting_precision']:

            dict0 = {}
            dict100n_lhs = {}
            dict200n_lhs = {}
            dict100n_lhs_new = {}
            dict200n_lhs_new = {}

            for dim in dim_list:

                value_0 = [0]*len(all_sid)
                value_100n_lhs = [0]*len(all_sid)
                value_200n_lhs = [0]*len(all_sid)
                value_100n_lhs_new = [0]*len(all_sid)
                value_200n_lhs_new = [0]*len(all_sid)

                for sample_multiplier in multiplier_list:

                    
                    if sample_multiplier == 10:
                        ela_feature_classes = ela_feature_classes_10
                    else:
                        ela_feature_classes = ela_feature_classes_other

                    
                    for sample_option in sample_list:


                        # if ap_name != 'ls0':
                        #     if ap_name != 'ls{}'.format(sample_multiplier):
                        #         continue
                        # else:
                        #     if sample_multiplier != 25:
                        #         continue


                        for sid in all_sid:

                            count = 0

                            # for pre_solver in [None, 'slsqp_multiplier50', 'smac_multiplier50']:
                            for pre_solver in [None]:
                                for selector in ['hiearchical_regression']:
                                    # for cross_valid_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
                                    for cross_valid_type in ['lopo_cv']:



                                        for fun_id in all_id_funs:

                                            dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                            table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                            if (comp == 'comp') and (sample_option != '0'):
                                                option = '0and{}'.format(sample_option)
                                            else:
                                                option = sample_option

                                            csv_file_path = os.path.join('pp_as_results', '{}'.format(option), '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))


                                            df_csv_file = pd.read_csv(csv_file_path, dtype={'selected_alg_error':float, 
                                                                                            'sbs_error':float, 
                                                                                            'vbs_alg_error':float,
                                                                                            'sbs_or_over':bool,
                                                                                            'vbs_selecting_precision':bool,
                                                                                            'vbs_or_over':bool
                                                                                            })


                                            count += sum(df_csv_file[precision_metric])



                            if sample_option == '0':
                                value_0[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            elif sample_option == '100n_lhs':
                                value_100n_lhs[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            elif sample_option == '200n_lhs':
                                value_200n_lhs[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            elif sample_option == '100n_lhs_new':
                                value_100n_lhs_new[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            elif sample_option == '200n_lhs_new':
                                value_200n_lhs_new[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                            
                            

                dict0['{}'.format(dim)] = value_0
                dict100n_lhs['{}'.format(dim)] = value_100n_lhs
                dict200n_lhs['{}'.format(dim)] = value_200n_lhs
                dict100n_lhs_new['{}'.format(dim)] = value_100n_lhs_new
                dict200n_lhs_new['{}'.format(dim)] = value_200n_lhs_new



            df_0 = pd.DataFrame(dict0)
            df_100n_lhs = pd.DataFrame(dict100n_lhs)
            df_200n_lhs = pd.DataFrame(dict200n_lhs)
            df_100n_lhs_new = pd.DataFrame(dict100n_lhs_new)
            df_200n_lhs_new = pd.DataFrame(dict200n_lhs_new)
            

            df_0_melt = pd.melt(df_0)
            df_0_melt['option'] = r'no surrogate'
            df_100n_lhs_melt = pd.melt(df_100n_lhs)
            df_100n_lhs_melt['option'] = r'$100n$\_lhs'
            df_200n_lhs_melt = pd.melt(df_200n_lhs)
            df_200n_lhs_melt['option'] = r'$200n$\_lhs'
            df_100n_lhs_new_melt = pd.melt(df_100n_lhs_new)
            df_100n_lhs_new_melt['option'] = r'$100n$\_lhs\_new'
            df_200n_lhs_new_melt = pd.melt(df_200n_lhs_new)
            df_200n_lhs_new_melt['option'] = r'$200n$\_lhs\_new'


            df = pd.concat([df_0_melt, df_100n_lhs_melt, df_200n_lhs_melt, df_100n_lhs_new_melt, df_200n_lhs_new_melt], axis=0)


            fig = plt.figure(figsize=(20, 15))
            ax = fig.add_subplot(111)
            sns.boxplot(x='variable', y='value', data=df, hue='option', palette='Dark2', ax=ax)


            ax.set_xlabel('Dimension', fontsize=55)
            ax.set_ylabel('Percentage of VBS selected in 31runs with '+r'$\mathcal{A}_{'+ap_option+'}$', fontsize=55)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02,), borderaxespad=0, fontsize=33, ncol=3)
            ax.tick_params(labelsize=55)

            boxplot_path = './plt/boxplot'
            os.makedirs(boxplot_path, exist_ok=True)
            boxplot_file_path = os.path.join(boxplot_path, '{}_{}.png'.format(precision_metric, ap_option))
            plt.savefig(boxplot_file_path)

            boxplot_file_path = os.path.join(boxplot_path, '{}_{}.png'.format(precision_metric, ap_option))
            pp = PdfPages(boxplot_file_path)
            pp.savefig(fig)
            pp.close()


### もともとはポートフォリオごとに画像を生成、画像の中身は次元数ごとに分けられたいくつかの設定でのVBS精度の箱ひげ図。
### ポートフォリオとsample_multiplierの不一致によりエラーが出てる。
### 今回はとりあえず、次元数を分けずに各ポートフォリオ（sample_multiplierごとに）分けた各設定でのVBS精度の箱ひげ図を出力する。→出力される画像は1枚。


# サロゲートとの比較用
def aggregate_boxplot_vbs3(dim_list, multiplier_list, ap_list, sampling_method, per_metric, dims, feature_selector, bbob_suite='bbob'):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set3')
    
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'
    

    sample_list = ['0', '100n_lhs', '200n_lhs', '100n_lhs_new', '200n_lhs_new']


    all_id_funs = range(1, 24+1)
    if bbob_suite == 'bbob_noisy':
         all_id_funs = range(101, 130+1)

    all_test_instance_id = range(1, 5+1)


    ### 注 解集合を拡張して得られた特徴量+拡張せずに得られた特徴量で計算したもの. そうでない場合はnoneにする.
    comp = 'comp'


    all_sid = range(0, 31)

    
    
    
        
    # for precision_metric in ['sbs_or_over', 'vbs_selecting_precision', 'vbs_or_over']:
    for precision_metric in ['vbs_selecting_precision']:

        dict0 = {}
        dict100n_lhs = {}
        dict200n_lhs = {}
        dict100n_lhs_new = {}
        dict200n_lhs_new = {}

        for sample_multiplier in multiplier_list:

            if sample_multiplier == 10:
                ela_feature_classes = ela_feature_classes_10
            else:
                ela_feature_classes = ela_feature_classes_other

            
            for dim in dim_list:

                value_0 = [0]*len(all_sid)
                value_100n_lhs = [0]*len(all_sid)
                value_200n_lhs = [0]*len(all_sid)
                value_100n_lhs_new = [0]*len(all_sid)
                value_200n_lhs_new = [0]*len(all_sid)

                    
                for sample_option in sample_list:


                    # if ap_name != 'ls0':
                    #     if ap_name != 'ls{}'.format(sample_multiplier):
                    #         continue
                    # else:
                    #     if sample_multiplier != 25:
                    #         continue


                    for sid in all_sid:

                        count = 0

                        # for pre_solver in [None, 'slsqp_multiplier50', 'smac_multiplier50']:
                        for pre_solver in [None]:
                            for selector in ['hiearchical_regression']:
                                # for cross_valid_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
                                for cross_valid_type in ['lopo_cv']:


                                    for fun_id in all_id_funs:

                                        dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                        table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                        if (comp == 'comp') and (sample_option != '0'):
                                            option = '0and{}'.format(sample_option)
                                        else:
                                            option = sample_option

                                        csv_file_path = os.path.join('pp_as_results', '{}'.format(option), 'ls{}_{}_{}_{}_{}'.format(sample_multiplier, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))


                                        df_csv_file = pd.read_csv(csv_file_path, dtype={'selected_alg_error':float, 
                                                                                        'sbs_error':float, 
                                                                                        'vbs_alg_error':float,
                                                                                        'sbs_or_over':bool,
                                                                                        'vbs_selecting_precision':bool,
                                                                                        'vbs_or_over':bool
                                                                                        })


                                        count += sum(df_csv_file[precision_metric])



                        if sample_option == '0':
                            value_0[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif sample_option == '100n_lhs':
                            value_100n_lhs[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif sample_option == '200n_lhs':
                            value_200n_lhs[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif sample_option == '100n_lhs_new':
                            value_100n_lhs_new[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        elif sample_option == '200n_lhs_new':
                            value_200n_lhs_new[sid] = count / (len(all_id_funs)*len(all_test_instance_id))
                        
                        

            dict0['{}n'.format(sample_multiplier)] = value_0
            dict100n_lhs['{}n'.format(sample_multiplier)] = value_100n_lhs
            dict200n_lhs['{}n'.format(sample_multiplier)] = value_200n_lhs
            dict100n_lhs_new['{}n'.format(sample_multiplier)] = value_100n_lhs_new
            dict200n_lhs_new['{}n'.format(sample_multiplier)] = value_200n_lhs_new


        df_0 = pd.DataFrame(dict0)
        df_100n_lhs = pd.DataFrame(dict100n_lhs)
        df_200n_lhs = pd.DataFrame(dict200n_lhs)
        df_100n_lhs_new = pd.DataFrame(dict100n_lhs_new)
        df_200n_lhs_new = pd.DataFrame(dict200n_lhs_new)
        

        df_0_melt = pd.melt(df_0)
        df_0_melt['option'] = r'no surrogate'
        df_100n_lhs_melt = pd.melt(df_100n_lhs)
        df_100n_lhs_melt['option'] = r'$100n$\_lhs'
        df_200n_lhs_melt = pd.melt(df_200n_lhs)
        df_200n_lhs_melt['option'] = r'$200n$\_lhs'
        df_100n_lhs_new_melt = pd.melt(df_100n_lhs_new)
        df_100n_lhs_new_melt['option'] = r'$100n$\_lhs\_new'
        df_200n_lhs_new_melt = pd.melt(df_200n_lhs_new)
        df_200n_lhs_new_melt['option'] = r'$200n$\_lhs\_new'


        df = pd.concat([df_0_melt, df_100n_lhs_melt, df_200n_lhs_melt, df_100n_lhs_new_melt, df_200n_lhs_new_melt], axis=0)


        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111)
        sns.boxplot(x='variable', y='value', data=df, hue='option', palette='Dark2', ax=ax)


        ax.set_xlabel(r'sampling size $s$', fontsize=55)
        ax.set_ylabel('Percentage of VBS selected', fontsize=55)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02,), borderaxespad=0, fontsize=33, ncol=3)
        ax.tick_params(labelsize=55)

        boxplot_path = './plt/boxplot'
        os.makedirs(boxplot_path, exist_ok=True)
        boxplot_file_path = os.path.join(boxplot_path, '{}.pdf'.format(precision_metric))
        plt.savefig(boxplot_file_path)

        boxplot_file_path = os.path.join(boxplot_path, '{}.pdf'.format(precision_metric))
        pp = PdfPages(boxplot_file_path)
        pp.savefig(fig)
        pp.close()





def aggregate_error_fun(dim_list, multiplier_list, ap_list, feature_selector, sampling_method, per_metric, dims):
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'



    all_id_funs = range(1, 24+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    csv_path = 'aggregate_csv'
    os.makedirs(csv_path, exist_ok=True)


    for dim in dim_list:
        df_dim = pd.DataFrame(0, index=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24'], columns=['A0-10', 'A0-15', 'A0-20', 'A0-25', 'A0-50', 'A10', 'A15', 'A20', 'A25', 'A50'])

        for fun_id in all_id_funs:
            for ap_name in ap_list:
                for sample_multiplier in multiplier_list:

                    if sample_multiplier == 10:
                        ela_feature_classes = ela_feature_classes_10
                    else:
                        ela_feature_classes = ela_feature_classes_other

                    if ap_name != 'ls0':
                        if ap_name != 'ls{}'.format(sample_multiplier):
                            continue

                    for selector in ['hiearchical_regression']:
                        for cross_valid_type in ['lopo_cv']:

                            error_summary = 0

                            for sid in all_sid:

                                dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                pp_as_path = os.path.join('pp_as_results', '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))

                                df_csv_file = pd.read_csv(pp_as_path, dtype={'selected_alg_error':float, 
                                                                                'sbs_error':float, 
                                                                                'vbs_alg_error':float,
                                                                                'sbs_or_over':bool,
                                                                                'vbs_selecting_precision':bool,
                                                                                'vbs_or_over':bool
                                                                                })
                                

                                error_summary += sum(df_csv_file['selected_alg_error'])
                            

                            if ap_name == 'ls0':
                                df_dim.at['f{}'.format(fun_id), 'A0-{}'.format(sample_multiplier)] += error_summary / len(all_sid)
                            elif ap_name == 'ls10':
                                df_dim.at['f{}'.format(fun_id), 'A10'] += error_summary / len(all_sid)
                            elif ap_name == 'ls15':
                                df_dim.at['f{}'.format(fun_id), 'A15'] += error_summary / len(all_sid)
                            elif ap_name == 'ls20':
                                df_dim.at['f{}'.format(fun_id), 'A20'] += error_summary / len(all_sid)
                            elif ap_name == 'ls25':
                                df_dim.at['f{}'.format(fun_id), 'A25'] += error_summary / len(all_sid)
                            elif ap_name == 'ls50':
                                df_dim.at['f{}'.format(fun_id), 'A50'] += error_summary / len(all_sid)

        print(r'%%%%%%%%%%%')
        print(r'\subfloat[各ポートフォリオを用いたシステムの各問題における誤差値の平均(n=' + '{}'.format(dim) + r')]{')
        print(r'\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|}')
        print(r'\toprule')
        print(r' n & \mathcal{A}_0(10n) & \mathcal{A}_0(15n) & \mathcal{A}_0(20n) & \mathcal{A}_0(25n) & \mathcal{A}_0(50n) & \mathcal{A}_10 & \mathcal{A}_15 & \mathcal{A}_20 & \mathcal{A}_25 & \mathcal{A}_50\\')
        print(r'midrule')
        for fun_id in all_id_funs:
            print('{} & '.format(dim), end='')
            for ap_name in ap_list:
                for sample_multiplier in multiplier_list:
                    if sample_multiplier == multiplier_list[-1]:
                        if ap_name != 'ls0':
                            if ap_name != 'ls{}'.format(sample_multiplier):
                                continue
                            else:
                                print('${:.4f}$ & '.format(df_dim.at['f{}'.format(fun_id), 'A{}'.format(sample_multiplier)]),end='')
                        else:
                            print('${:.4f}$ & '.format(df_dim.at['f{}'.format(fun_id), 'A0-{}'.format(sample_multiplier)]),end='')
                    else:
                        if ap_name != 'ls0':
                            if ap_name != 'ls{}'.format(sample_multiplier):
                                continue
                            else:
                                print('${:.4f}$'.format(df_dim.at['f{}'.format(fun_id), 'A{}'.format(sample_multiplier)]) + r'\\')
                        else:
                            print('${:.4f}$'.format(df_dim.at['f{}'.format(fun_id), 'A0-{}'.format(sample_multiplier)]) + r'\\')

        print(r'\bottomrule')
        print(r'\end{tabular}}\\')

    

def significant_vsSBS(dim_list, multiplier_list, ap_list, sampling_method, per_metric, dims, feature_selector):
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'

    all_id_funs = range(1, 24+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    print(r'%%%%%%%%%%%')
    print(r'\begin{tabular}{cccccccc}')
    print(r'\toprule')
    print(r' & $n=2$ & $n=3$ & $n=5$ & $n=10$\\ ')
    print(r'\midrule')

    for ap_name in ap_list:
        for sample_multiplier in multiplier_list:

            if ap_name != 'ls0':
                if ap_name != 'ls{}'.format(sample_multiplier):
                    continue
            else:
                if sample_multiplier != 25:
                    continue
            
            if sample_multiplier == 10:
                ela_feature_classes = ela_feature_classes_10
            else:
                ela_feature_classes = ela_feature_classes_other

            if ap_name == 'ls0':
                print(r'$\set{A}_{0}, s=25n$ & ',end='')
            elif ap_name == 'ls10':
                print(r'$\set{A}_{10}, s=10n$ & ',end='')
            elif ap_name == 'ls15':
                print(r'$\set{A}_{15}, s=15n$ & ',end='')
            elif ap_name == 'ls20':
                print(r'$\set{A}_{20}, s=20n$ & ',end='')
            elif ap_name == 'ls25':
                print(r'$\set{A}_{25}, s=25n$ & ',end='')
            elif ap_name == 'ls50':
                print(r'$\set{A}_{50}, s=50n$ & ',end='')
            

            for dim in dim_list:

                dict_significant = {'plus':0, 'minus':0, 'approx':0}

                for fun_id in all_id_funs:

                    value_sys = [0]*len(all_sid)
                    value_sbs = [0]*len(all_sid)

                    for selector in ['hiearchical_regression']:
                        for cross_valid_type in ['lopo_cv']:

                            for sid in all_sid:

                                dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                                table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims

                                pp_as_path = os.path.join('pp_as_results', '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector), 'f{}_DIM{}.csv'.format(fun_id, dim))

                                df_csv_file = pd.read_csv(pp_as_path, dtype={'selected_alg_error':float, 
                                                                                'sbs_error':float, 
                                                                                'vbs_alg_error':float,
                                                                                'sbs_or_over':bool,
                                                                                'vbs_selecting_precision':bool,
                                                                                'vbs_or_over':bool
                                                                                })

                                
                                value_sys[sid] += sum(df_csv_file['selected_alg_error']) / len(all_test_instance_id)
                                
                                value_sbs[sid] += sum(df_csv_file['sbs_error']) / len(all_test_instance_id)


                    statistic, p = stats.mannwhitneyu(value_sys, value_sbs)
                    superior = sorted(value_sys)[15] < sorted(value_sbs)[15]
                    close = np.isclose(sorted(value_sys)[15], sorted(value_sbs)[15])
                    if p < 0.05:
                        if close:
                            dict_significant['approx'] += 1
                        elif superior:
                            dict_significant['plus'] += 1
                        else:
                            dict_significant['minus'] += 1
                    else:
                        dict_significant['approx'] += 1
                
                print('${}/{}/{}$'.format(dict_significant['plus'],dict_significant['minus'], dict_significant['approx']), end='')

                if dim == dim_list[-1]:
                    print(r'\\')
                else:
                    print(' & ', end='')
    
    print(r'\bottomrule')
    print('\end{tabular}')



@click.command()
# @click.option('--surrogate_number', '-sur', required=False, default=2000, type=int, help='Number of surrogated X')
@click.option('--surrogate_multiplier1', '-surm1', required=True, default=100, type=int, help='Multiplier of surrogated X')
@click.option('--surrogate_sample1', '-surs1', required=True, default='random', type=str, help='the way of surrogate')
@click.option('--surrogate_multiplier2', '-surm2', required=False, default=0, type=int, help='Multiplier of surrogated X')
@click.option('--surrogate_sample2', '-surs2', required=False, default='no', type=str, help='the way of surrogate')
def main(surrogate_multiplier1, surrogate_sample1,surrogate_multiplier2, surrogate_sample2):
    # ela_feature_classes = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    # ela_feature_classes = 'basic_ela_distr_pca_limo_ic_disp_ela_meta'
    dims = 'dims2_3_5_10'
    sampling_method = 'lhs'
    # sample_multiplier = 50
    # per_metric = 'RMSE'
    per_metric = 'accuracy'
    feature_selector = 'none'
    n_features_to_select = 0
    
    multiplier_list = [10, 15, 20, 25, 50]
    # ap_list = ['ls0', 'ls10', 'ls15', 'ls20', 'ls25', 'ls50']
    ap_list = ['ls10', 'ls15', 'ls20', 'ls25', 'ls50']
    dim_list = [2, 3, 5, 10]
    # all_systems = ['0', '100n_lhs', '200n_lhs', '300n_lhs', '400n_lhs', '500n_lhs', '1000n_lhs', '100n_random', '200n_random', '300n_random']
    all_systems = ['0', '100n_lhs', '200n_lhs', '300n_lhs', '400n_lhs', '500n_lhs', '1000n_lhs']
    
    
    # precision_datarframize(multiplier_list, ap_list, dim_list, dims, sampling_method, per_metric, feature_selector, n_features_to_select)


    # dict_p, dict_Median = Friedman(multiplier_list, ap_list, dim_list, dims, sampling_method, per_metric, feature_selector, n_features_to_select)


    # aggregate_tablize()

    # aggregate_colored_tablize(dict_p, dict_Median)

    # dict_p, dict_Median, dict_error_mean_0, dict_error_mean_sampling = aggregate_error(multiplier_list, ap_list, dim_list, dims, sampling_method, per_metric, feature_selector, n_features_to_select)
    # ls0_superior_dict = aggregate_error(multiplier_list, ap_list, dim_list, dims, sampling_method, per_metric, feature_selector)

    # aggregate_colored_tablize(dict_p, dict_Median)

    # sbs_tablize(ls0_superior_dict, multiplier_list, dim_list, sampling_method, dims, per_metric, feature_selector)


    # 先生にCSVを提出するためのコード. 
    # aggregate_tocsv(dim_list, multiplier_list, ap_list, feature_selector, sampling_method, per_metric, dims)

    # if surrogate_multiplier2 == 0 and surrogate_sample2 == 'no':
    #     # サロゲートXと比較用コード.
    #     aggregate_tocsv2(surrogate_multiplier1, surrogate_sample1, dim_list, multiplier_list, ap_list, feature_selector, sampling_method, per_metric, dims)
    # else:
    #     # サロゲートXと比較用コード.2
    #     aggregate_tocsv3(surrogate_multiplier1, surrogate_sample1, surrogate_multiplier2, surrogate_sample2, dim_list, multiplier_list, ap_list, feature_selector, sampling_method, per_metric, dims)

    # 全てのシステムの比較用コード.
    # aggregate_tocsv_allsys(all_systems, dim_list, feature_selector, sampling_method, per_metric, dims, multiplier_list)


    # VBS精度を箱ひげ図化する関数.
    # aggregate_boxplot_vbs(dim_list, multiplier_list, ap_list, sampling_method, per_metric, dims, feature_selector)

    # aggregate_boxplot_vbs2(dim_list, multiplier_list, ap_list, sampling_method, per_metric, dims, feature_selector)
    aggregate_boxplot_vbs3(dim_list, multiplier_list, ap_list, sampling_method, per_metric, dims, feature_selector)
    
    # 関数ごとの誤差の平均を表化する関数.
    # aggregate_error_fun()


    # 各ポートフォリオの各次元数について, 選択されたアルゴリズムによる解精度がSBSによる解精度に対して有意差があるかどうかを集計し表化する関数.
    # significant_vsSBS(dim_list, multiplier_list, ap_list, sampling_method, per_metric, dims, feature_selector)





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

    main()