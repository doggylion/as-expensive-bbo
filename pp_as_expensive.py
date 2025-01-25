"""
Postprocessing results of algorithm selection.
"""
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
from fopt_info import bbob_fopt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(filename='{}.log'.format(__file__), level=logging.DEBUG)



def pp_selected_algs(ap_dir_path, as_res_dir_path, sample_dir_path, sbs_dir_path, bbob_suite, dim, fun_id, test_instance_ids, target_pow, pre_solver, per_metric='RMSE'):
    pp_dir_path = 'pp_' + as_res_dir_path
    if pre_solver != None:
        pp_dir_path += '_' + pre_solver

    os.makedirs(pp_dir_path, exist_ok=True)


    # fitness_dir_path = os.path.join(pp_dir_path, 'fitness_error_comparison')
    # os.makedirs(fitness_dir_path, exist_ok=True)


    df_fitness_error = pd.DataFrame(
        columns=[
            'iid', 
            'selected_alg', 
            'selected_alg_error', 
            'sbs_error', 
            'vbs_alg', 
            'vbs_alg_error',
            'sbs_or_over',
            'vbs_selecting_precision',
            'vbs_or_over'
            ])

    for iid in test_instance_ids:
        alg_name_path = os.path.join(as_res_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(fun_id, dim, iid))
        selected_alg = np.loadtxt(alg_name_path, delimiter=",", comments="#", dtype=str)

        selected_alg_path = os.path.join(ap_dir_path, 'accuracy/{}_f{}_DIM{}.csv'.format(selected_alg, fun_id, dim))
        s_alg_error = (pd.read_csv(selected_alg_path, names=['id', 'error'])['error'])[iid-1]


        sbs_error = (pd.read_csv(sbs_dir_path, names=['id', 'error'])['error'])[iid-1]


        vbs_name_path = os.path.join(ap_dir_path, 'vbs_{}/f{}_DIM{}.csv'.format(per_metric, fun_id, dim))
        vbs_alg = np.loadtxt(vbs_name_path, delimiter=",", comments="#", dtype=str)

        vbs_alg = vbs_alg.tolist()
        
        if isinstance(vbs_alg, str):
            vbs_path = os.path.join(ap_dir_path, 'accuracy/{}_f{}_DIM{}.csv'.format(vbs_alg, fun_id, dim))
        else:
            vbs_path = os.path.join(ap_dir_path, 'accuracy/{}_f{}_DIM{}.csv'.format(vbs_alg[0], fun_id, dim))
        vbs_error = (pd.read_csv(vbs_path, names=['id', 'error'])['error'])[iid-1]


        if isinstance(vbs_alg, str):
            dict_fitness_error = {'iid':int(iid), 
                                'selected_alg':selected_alg, 
                                'selected_alg_error': s_alg_error, 
                                'sbs_error': sbs_error, 
                                'vbs_alg': vbs_alg, 
                                'vbs_alg_error': vbs_error,
                                'sbs_or_over':bool(float(s_alg_error) <= float(sbs_error)),
                                'vbs_selecting_precision':(selected_alg == vbs_alg),
                                'vbs_or_over':bool(float(s_alg_error) <= float(vbs_error))
                                }
        else:
            dict_fitness_error = {'iid':int(iid), 
                                'selected_alg':selected_alg, 
                                'selected_alg_error': s_alg_error, 
                                'sbs_error': sbs_error, 
                                'vbs_alg': vbs_alg[0], 
                                'vbs_alg_error': vbs_error,
                                'sbs_or_over':bool(float(s_alg_error) <= float(sbs_error)),
                                'vbs_selecting_precision':(selected_alg in vbs_alg),
                                'vbs_or_over':bool(float(s_alg_error) <= float(vbs_error))
                                }

        # df_fitness_error = df_fitness_error.append(dict_fitness_error, ignore_index=True, sort=False)
        df_add = pd.DataFrame(dict_fitness_error, index=[0])
        df_fitness_error = pd.concat([df_fitness_error, df_add], ignore_index=True, sort=False)

    comparison_dir_path = os.path.join(pp_dir_path, 'f{}_DIM{}.csv'.format(fun_id, dim))
    df_fitness_error.to_csv(comparison_dir_path, index=False)

    

def run(surrogate_multiplier, ap_name, per_metric, option, as_res_dir_path, pre_solver=None, bbob_suite='bbob'):
    dims = [2, 3, 5, 10]
    #dims = [10]    
    test_instance_ids = range(1, 5+1)
    target_pow = '-2.0'

    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)  
    
    

    # The bbob_fopt_data directory contains a file that provides the objective value f(x^*) of the optimal solution x^* for each function instance. f(x^*) is used in the postprocessing the performance of a sampler    
    if os.path.exists('./{}_fopt_data'.format(bbob_suite)) == False:
        bbob_fopt()
    
    as_config_file_path = os.path.join(as_res_dir_path, 'config.json')
    with open(as_config_file_path) as fh:
        config_dict = json.load(fh)

    

    # copy the metric value values of the SBS in the algorithm portfolio
    for dim in dims:
        # pp_sbs_dir_path = 'pp_as_results/sbs_{}_{}_DIM{}/rel{}'.format(config_dict['ap_name'], config_dict['per_metric'], dim, config_dict['per_metric'])
        pp_sbs_dir_path = 'pp_as_results/{}/sbs_{}_{}_DIM{}/{}'.format(option, config_dict['ap_name'], config_dict['per_metric'], dim, config_dict['per_metric'])
        if os.path.isdir(pp_sbs_dir_path) == False:
            os.makedirs(pp_sbs_dir_path, exist_ok=True)                    
            sbs_file_path = os.path.join('./alg_portfolio', config_dict['ap_name'], 'sbs_{}/sbs_DIM{}.csv'.format(config_dict['per_metric'], dim))
            with open(sbs_file_path, 'r') as fh:
                sbs_name = fh.read()

            for fun_id in all_fun_ids:
                # rel_metric_path = os.path.join('./alg_portfolio', config_dict['ap_name'], 'rel'+config_dict['per_metric'], '{}_f{}_DIM{}_liid0.csv'.format(sbs_name, fun_id, dim))
                metric_path = os.path.join('./alg_portfolio/{}/{}/{}_f{}_DIM{}.csv'.format(ap_name, per_metric, sbs_name, fun_id, dim))
                copied_metric_path = os.path.join(pp_sbs_dir_path, 'f{}_DIM{}.csv'.format(fun_id, dim))
                shutil.copyfile(metric_path, copied_metric_path)
                
    # Calculate the metric value of the algorithms selected by the algorithm selection system
    for dim in dims:
        for fun_id in all_fun_ids:
            sbs_dir_path = os.path.join('pp_as_results/{}/sbs_{}_{}_DIM{}/{}/f{}_DIM{}.csv'.format(option, config_dict['ap_name'], config_dict['per_metric'], dim, config_dict['per_metric'], fun_id, dim))
            pp_selected_algs(config_dict['ap_dir_path'], as_res_dir_path, config_dict['sample_dir_path'], sbs_dir_path, bbob_suite, dim, fun_id, test_instance_ids, target_pow, pre_solver, per_metric=config_dict['per_metric'])
     

@click.command()
# @click.option('--surrogate_number', '-sur', required=False, default=2000, type=int, help='Number of surrogated X')
@click.option('--surrogate_multiplier', '-surm', required=True, default=100, type=int, help='Multiplier of surrogated X')
@click.option('--surrogate_sample', '-surs', required=True, default='random', type=str, help='the way of surrogate')
@click.option('--comp_option', '-comp', required=True, default='default', type=str, help='the way of surrogate')
@click.option('--separation', '-spr', required=True, default='none', type=str, help='separate or not')
def main(surrogate_multiplier, surrogate_sample, comp_option, separation):
    # ela_feature_classes = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'
    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    dims = 'dims2_3_5_10'
    sampling_method = 'lhs'
    # sample_multiplier = 50
    per_metric = 'accuracy'
    feature_selector = 'none'
    n_features_to_select = 0
    

    for sample_multiplier in [10, 15, 20, 25, 50]:
        # for ap_name in ['kt', 'dlvat', 'jped', 'bmtp', 'mk', 'ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']:

        if sample_multiplier == 10:
            ela_feature_classes = ela_feature_classes_10
        else:
            ela_feature_classes = ela_feature_classes_other


        # for ap_name in ['ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']:
        # for ap_name in ['ls0', 'ls10', 'ls15', 'ls20', 'ls25', 'ls50']:
        for ap_name in ['ls10', 'ls15', 'ls20', 'ls25', 'ls50']:
            if ap_name != 'ls0':
                if ap_name != 'ls{}'.format(sample_multiplier):
                    continue

            # for pre_solver in [None, 'slsqp_multiplier50', 'smac_multiplier50']:
            for pre_solver in [None]:
                for sid in range(0, 31):
                    dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                    table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims
                    # for selector in ['multiclass_classification', 'hiearchical_regression', 'clustering', 'pairwise_regression', 'pairwise_classification']:
                    for selector in ['hiearchical_regression']:
                        # for cross_valid_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
                        for cross_valid_type in ['lopo_cv']:
                            if separation == 'none':
                                option = '{}n_{}'.format(surrogate_multiplier, surrogate_sample)
                            else:
                                option = '{}n_{}_{}'.format(surrogate_multiplier, surrogate_sample, separation)

                            if comp_option == 'comp':
                                option = '0and{}'.format(option)
                                
                            as_result_dir_path = os.path.join('as_results/{}/{}_{}_{}_{}_{}'.format(option, ap_name, selector, cross_valid_type, table_data_name, feature_selector))

                            if feature_selector != 'none':
                                as_result_dir_path += '_nfs{}'.format(n_features_to_select)    
                            run(surrogate_multiplier, ap_name, per_metric, option, as_result_dir_path, pre_solver=pre_solver)
                            logger.info("Postprocessing of %s was done.", as_result_dir_path)


if __name__ == '__main__':
    main()