"""
Postprocessing results of algorithm selection.
"""
import os
import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.ticker as mtick
import seaborn as sns
import csv
import logging
import statistics as stat
import subprocess
from scipy.stats import rankdata

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True

mat_markers = ['o',  '^', 's', 'd', 'p', 'h', '8', '*', 'X', 'D']

    
# def controltest(pp_res_dir_path, res_file_path, n_methods, dim):
def controltest(pp_res_dir_path, res_file_path, n_methods, sample_multiplier):
    os.makedirs('./tmp', exist_ok=True)

    with open('./tmp/tmp1.tex', "w") as fh:
        subprocess.run(['java', 'controlTest.Friedman', res_file_path], stdout=fh)        

    head_line = 16 + n_methods
    with open('./tmp/tmp2.tex', "w") as fh:
        subprocess.run(['head', '-n', '{}'.format(head_line), './tmp/tmp1.tex'], stdout=fh)
        
    with open('./tmp/tmp3.tex', "w") as fh:
        subprocess.run(['tail', '-n', '{}'.format(n_methods), './tmp/tmp2.tex'], stdout=fh)

    score_list = []        
    with open('./tmp/tmp3.tex', "r") as fh:
        file_data = fh.readlines()
        for sel_id, line in enumerate(file_data):
            tmp = line.rstrip()
            i = tmp.find('&')
            tmp[i+1:]
            score = tmp[i+1:].replace('\\', '')
            score_list.append(float(score)) 

    for i, score in enumerate(score_list):        
        score_file_path = os.path.join(pp_res_dir_path, 'multiplier{}_{}th_method.csv'.format(sample_multiplier, i))
        with open(score_file_path, 'w') as fh:
            fh.write(str(score))

    score_arr = np.array(score_list)
    sorted_ids = np.argsort(score_arr)

    # print(score_list)
    # print(score_arr)
    # print(sorted_ids)

    return sorted_ids
    

# def plot_all_sys(fig_file_path, all_dims, all_systems, all_dim_symbols, sorted_sys_dict, dims_dict, pp_res_dir_path):
def plot_all_sys(fig_file_path, all_multipliers, all_systems, all_multipliers_symbols, sorted_sys_dict, multipliers_dict, pp_res_dir_path):
    fig = plt.figure(figsize=(9, 3))
    ax = plt.subplot(111)
    ax.set_rasterization_zorder(1)

    for i, system in enumerate(all_systems):
        average_ranking_list = []

        # for dim in all_dims:
        #     system_index = sorted_sys_dict[str(dims_dict[dim])].index(system)
        #     system_file_path = os.path.join(pp_res_dir_path, 'dim{}_{}th_method.csv'.format(dims_dict[dim], system_index))

        #     with open(system_file_path, 'r') as fh:
        #         average_ranking_list.append(float(fh.read()))

        # ax.plot(all_dims, average_ranking_list, lw=2, marker=mat_markers[i], markersize=10, markerfacecolor="None", markeredgewidth=2, label='${}$'.format(system.replace('_', '-')))
        
        for multiplier in all_multipliers:
            system_index = sorted_sys_dict[str(multipliers_dict[multiplier])].index(system)
            system_file_path = os.path.join(pp_res_dir_path, 'multiplier{}_{}th_method.csv'.format(multipliers_dict[multiplier], system_index))

            with open(system_file_path, 'r') as fh:
                average_ranking_list.append(float(fh.read()))

        ax.plot(all_multipliers, average_ranking_list, lw=2, marker=mat_markers[i], markersize=10, markerfacecolor="None", markeredgewidth=2, label='${}$'.format(system.replace('_', '-')))


    plt.grid(True, which="both", ls="-", color='0.85')    
        
    plt.ylim(ymin=3.5, ymax=7)
    ax.invert_yaxis()
    plt.yticks(fontsize=27)

    # ax.set_xticklabels(all_dims, rotation='vertical', fontsize=27)
    ax.set_xticklabels(all_multipliers, rotation='vertical', fontsize=27)

    # plt.xlabel("dimmentions", fontsize=30)
    plt.xlabel("multiplier numbers", fontsize=30)

    plt.ylabel("Average Ranking", fontsize=30)        
    plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.4), ncol=3, fontsize=10, columnspacing=0.3)    
    plt.savefig(fig_file_path, bbox_inches='tight')
    plt.close()
        
if __name__ == '__main__':    
    bbob_suite = 'bbob'
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)
    dim_list = [2, 3, 5, 10]
    feature_selector = 'none'

    ela_feature_classes = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    dims = 'dims2_3_5_10'

    per_metric = 'accuracy'
    sampling_method = 'lhs'
    sample_multiplier_list = [10, 15, 20, 25, 50]
    n_methods = 10

    # dims_dict = {'n=2':2, 'n=3':3, 'n=5':5, 'n=10':10}
    # all_dims = ['n=2', 'n=3', 'n=5', 'n=10']
    multipliers_dict = {'s=10n':10, 's=15n':15, 's=20n':20, 's=25n':25, 's=50n':50}
    all_multipliers = ['s=10n', 's=15n', 's=20n', 's=25n', 's=50n']

    all_systems = ['0', '100n_lhs', '200n_lhs', '300n_lhs', '400n_lhs', '500n_lhs', '1000n_lhs', '100n_random', '200n_random', '300n_random']

    # for sample_multiplier in sample_multiplier_list:
    for dim in dim_list:
        # dir_sampling_method = '{}_multiplier{}_sid_{}'.format(sampling_method, sample_multiplier, per_metric)
        # table_data_name = dir_sampling_method + '_' + ela_feature_classes + '_' + dims

        # csv_path = os.path.join('./plotallsys_csv', 'multiplier{}'.format(sample_multiplier))
        csv_path = os.path.join('./plotallsys_csv', 'dim{}'.format(dim))

        pp_res_dir_path = './friedman_results_pls'
        os.makedirs(pp_res_dir_path, exist_ok=True)
        
        sorted_sys_dict = {}
        # for dim in dim_list:
        #     res_file_path = os.path.join(csv_path, 'dim{}.csv'.format(dim))    
            
        #     sorted_id = controltest(pp_res_dir_path, res_file_path, n_methods, dim)
        #     sorted_allsys = []
        #     for id in sorted_id:
        #         sorted_allsys.append(all_systems[id])
        #     sorted_sys_dict["{}".format(dim)] = sorted_allsys
        for sample_multiplier in sample_multiplier_list:
            res_file_path = os.path.join(csv_path, 'multiplier{}.csv'.format(sample_multiplier))    
            
            sorted_id = controltest(pp_res_dir_path, res_file_path, n_methods, sample_multiplier)
            sorted_allsys = []
            for id in sorted_id:
                sorted_allsys.append(all_systems[id])
            sorted_sys_dict["{}".format(sample_multiplier)] = sorted_allsys

        # all_dim_symbols = []
        # for dim in all_dims:
        #     all_dim_symbols.append(dims_dict[dim])
        all_multipliers_symbols = []
        for multiplier in all_multipliers:
            all_multipliers_symbols.append(multipliers_dict[multiplier])        
        
        for selector in ['hiearchical_regression']:
            fig_dir_path = os.path.join('pp_figs')
            os.makedirs(fig_dir_path, exist_ok=True)
            # fig_file_path = os.path.join(fig_dir_path, 'multiplier{}.pdf'.format(sample_multiplier))
            fig_file_path = os.path.join(fig_dir_path, 'dim{}.pdf'.format(dim))

            # plot_all_sys(fig_file_path, all_dims, all_systems, all_dim_symbols, sorted_sys_dict, dims_dict, pp_res_dir_path)
            plot_all_sys(fig_file_path, all_multipliers, all_systems, all_multipliers_symbols, sorted_sys_dict, multipliers_dict, pp_res_dir_path)
            logger.info("A figure was generated: %s", fig_file_path)
