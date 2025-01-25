import pandas as pd
import os
import bbobbenchmarks
import fgeneric
import numpy as np
import click
import statistics

def BBOB_calculate(df_csv_file, fun_id, instance_id, bbob_suite='bbob', output_folder='true_surrogeted_sample'):
        datapath = 'tmp'
        opts = dict(algid='PUT ALGORITHM NAME',
                comments='PUT MORE DETAILED INFORMATION, PARAMETER SETTINGS ETC')
        f = fgeneric.LoggingFunction(datapath, **opts)
    
        f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=instance_id))

        # print(f.evalfun(df_csv_file))

        return f.evalfun(df_csv_file)


# def dataframize_mse(dim_list, surrogate_multiplier_list, multiplier_list):
#     all_id_funs = range(1, 24+1)

#     all_test_instance_id = range(1, 5+1)

#     all_sid = range(0, 31)

#     sampling = 'lhs'

#     output_dir_path = 'mse_output'
#     os.makedirs(output_dir_path, exist_ok=True)

#     for sample_multiplier in multiplier_list:
#         multiplier_dir_path = os.path.join(output_dir_path, 'multiplier{}'.format(sample_multiplier))
#         os.makedirs(multiplier_dir_path, exist_ok=True)

#         for dim in dim_list:

#             path = os.path.join(output_dir_path, 'multiplier{}_dim{}.text'.format(sample_multiplier, dim))

#             output_file_path = os.path.join(output_dir_path, 'multiplier{}/dim{}.csv'.format(sample_multiplier, dim))

#             is_first_time = True

#             df = pd.DataFrame(0, index=surrogate_multiplier_list, 
#                                             columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24'])

#             for sampling_option in ['lhs', 'random']:
#                 if os.path.isfile(output_file_path):
#                     continue

#                 df_multiplier = pd.DataFrame(0, index=surrogate_multiplier_list, 
#                                             columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24'])
                
#                 for surrogate_multiplier in surrogate_multiplier_list:
#                 # for surrogate_multiplier in ['100n']:
#                     if sampling_option == 'random':
#                         print('check1')
#                         if surrogate_multiplier != '100n':
#                             if surrogate_multiplier != '200n':
#                                 if surrogate_multiplier != '300n':
#                                     print('check2')
#                                     df_multiplier = df_multiplier.drop(surrogate_multiplier)
#                                     continue

#                     for fun_id in all_id_funs:

#                         sample_path = os.path.join('sample_data', '{}_{}'.format(surrogate_multiplier, sampling_option))

#                         score = 0
#                         count = 0

#                         for sid in all_sid:
#                             sample_dir = os.path.join(sample_path, '{}_multiplier{}_sid{}'.format(sampling, sample_multiplier, sid))

#                             for instance_id in all_test_instance_id:
#                                 file_path = os.path.join(sample_dir, 'x_f_data_bbob_f{}_DIM{}_i{}.csv'.format(fun_id, dim, instance_id))

#                                 df_csv_file = pd.read_csv(file_path, header=None).iloc[dim*sample_multiplier:, 1:]
#                                 surrogated_obj = pd.read_csv(file_path, header=None).iloc[dim*sample_multiplier:, 0].to_list()

#                                 obj = BBOB_calculate(df_csv_file.to_numpy().tolist(), fun_id, instance_id)

#                                 for i in range((int(surrogate_multiplier.rsplit('n')[0])-sample_multiplier)*dim):
#                                     score += (obj[i] - surrogated_obj[i]) ** 2
#                                     count += 1
                            
#                             print("sample={}, dim={}, surrogate={}, option={}, fun={}, sid={} is done.".format(sample_multiplier, dim, surrogate_multiplier, sampling_option, fun_id, sid))

#                         df_multiplier.at[surrogate_multiplier, 'f{}'.format(fun_id)] += score / count

#                 df_multiplier = df_multiplier.add_suffix('_{}'.format(sampling_option), axis='index')

#                 print(df_multiplier)

#                 if is_first_time:
#                     df = df_multiplier
#                     is_first_time = False
#                 else:
#                     df = pd.concat([df, df_multiplier])
            
#             if not os.path.isfile(output_file_path):
#                 df.to_csv(output_file_path)
    
#         # print(df)

#             result_file = pd.read_csv(output_file_path, header=0, index_col=0)

#             print(result_file)

#             with open(path, mode='w') as f:
#                 print(r'%%%%%%%%%%%', file=f)
#                 print(r'\subfloat[各システムの各問題における平均二乗誤差($n='+'{}'.format(dim)+r'$)]{', file=f)
#                 print(r'\begin{tabular}{60em}{lcccccccccccccccccccccccc}', file=f)
#                 print(r'\rowcolor[rgb]{0.9, 0.9, 0.9} & ', end='', file=f)
#                 print(r'\toprule', file=f)

#                 print(r' & ')
#                 for fun_id in all_id_funs:    
#                     print(r'$f_{'+'{}'.format(fun_id)+r'}$', end='', file=f)
                    
#                     if fun_id != all_id_funs[-1]:
#                         print(' & ', end='', file=f)
#                     else:
#                         print(r'\\', file=f)

#                 print(r'\midrule', file=f)

                
#                 for surrogate_multiplier in surrogate_multiplier_list:
                        
#                     for sampling_option in ['lhs', 'random']:
#                         if sampling_option == 'random':
#                             if surrogate_multiplier != '100n':
#                                 if surrogate_multiplier != '200n':
#                                     if surrogate_multiplier != '300n':
#                                         continue

#                         print(r'$\mathcal{A}_'+'{}'.format(sample_multiplier)+r'$, $s^{\prime}='+'{}'.format(surrogate_multiplier)+r'$', end=' & ', file=f)

#                         for fun_id in all_id_funs:
#                             print('{}'.format(result_file.at['{}_{}'.format(surrogate_multiplier, sampling_option), 'f{}'.format(fun_id)]), end='', file=f)

#                             if fun_id != all_id_funs[-1]:
#                                 print(' & ', end='', file=f)                
#                             else:
#                                 print(r'\\', file=f)

#                 print(r'\bottomrule', file=f)
#                 print(r'\end{tabular}}\\', file=f)
#                 print(r'%%%%%%%%%%%', file=f)


def dataframize_mse(dim_list, surrogate_multiplier_list, multiplier_list, mse_or_r2, separation):
    all_id_funs = range(1, 24+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    sampling = 'lhs'

    if mse_or_r2 == 'mse':
        output_dir_path = 'mse_output'
    elif mse_or_r2 == 'r2':
        output_dir_path = 'r2_output'
    os.makedirs(output_dir_path, exist_ok=True)

    
    for fun_id in all_id_funs:
        fun_dir_path = os.path.join(output_dir_path, 'f{}'.format(fun_id))
        os.makedirs(fun_dir_path, exist_ok=True)

        for dim in dim_list:

            path = os.path.join(output_dir_path, 'f{}_dim{}.text'.format(fun_id, dim))

            output_file_path = os.path.join(output_dir_path, 'f{}/dim{}.csv'.format(fun_id, dim))

            is_first_time = True

            df = pd.DataFrame(0, index=surrogate_multiplier_list, columns=multiplier_list)

            for separation_option in separation:

                for sampling_option in ['lhs', 'random']:
                    if os.path.isfile(output_file_path):
                        continue

                    df_multiplier = pd.DataFrame(0, index=surrogate_multiplier_list, columns=multiplier_list)
                    
                    for surrogate_multiplier in surrogate_multiplier_list:
                    # for surrogate_multiplier in ['100n']:
                        if sampling_option == 'random':
                            print('check1')
                            if separation_option == 'default':
                                if surrogate_multiplier != '100n':
                                    if surrogate_multiplier != '200n':
                                        if surrogate_multiplier != '300n':
                                            print('check2')
                                            df_multiplier = df_multiplier.drop(surrogate_multiplier)
                                            continue
                            else: 
                                df_multiplier = df_multiplier.drop(surrogate_multiplier)
                                continue
                        else:
                            if separation_option == 'new':
                                if surrogate_multiplier != '100n':
                                    df_multiplier = df_multiplier.drop(surrogate_multiplier)
                                    continue

                        for sample_multiplier in multiplier_list:

                            if separation_option == 'default':
                                sample_path = os.path.join('sample_data', '{}_{}'.format(surrogate_multiplier, sampling_option))
                            else:
                                sample_path = os.path.join('sample_data', '{}_{}_{}'.format(surrogate_multiplier, sampling_option, separation_option))   

                            score = 0
                            count = 0
                            total = 0

                            obj_list = []

                            for sid in all_sid:
                                sample_dir = os.path.join(sample_path, '{}_multiplier{}_sid{}'.format(sampling, sample_multiplier, sid))

                                for instance_id in all_test_instance_id:
                                    file_path = os.path.join(sample_dir, 'x_f_data_bbob_f{}_DIM{}_i{}.csv'.format(fun_id, dim, instance_id))

                                    df_csv_file = pd.read_csv(file_path, header=None).iloc[dim*sample_multiplier:, 1:]
                                    surrogated_obj = pd.read_csv(file_path, header=None).iloc[dim*sample_multiplier:, 0].to_list()

                                    obj = BBOB_calculate(df_csv_file.to_numpy().tolist(), fun_id, instance_id)

                                    for i in range((int(surrogate_multiplier.rsplit('n')[0])-sample_multiplier)*dim):
                                        score += (obj[i] - surrogated_obj[i]) ** 2
                                        count += 1
                                        obj_list.append(obj[i])
                                
                                print("separation={}, fun={}, dim={}, option={}, surrogate={}, sample={}, sid={} is done.".format(separation_option, fun_id, dim, sampling_option, surrogate_multiplier, sample_multiplier, sid))

                            if mse_or_r2 == 'mse':
                                df_multiplier.at[surrogate_multiplier, sample_multiplier] += score / count
                            elif mse_or_r2 == 'r2':
                                mean_of_score = statistics.mean(obj_list)
                                for object in obj_list:
                                    total += (object - mean_of_score) ** 2
                                df_multiplier.at[surrogate_multiplier, sample_multiplier] += 1 - (score / total)


                    df_multiplier = df_multiplier.add_suffix('_{}_{}'.format(sampling_option, separation_option), axis='index')

                    print(df_multiplier)

                    if is_first_time:
                        df = df_multiplier
                        is_first_time = False
                    else:
                        df = pd.concat([df, df_multiplier])
                
            if not os.path.isfile(output_file_path):
                df.to_csv(output_file_path)
    
        # print(df)

            result_file = pd.read_csv(output_file_path, header=0, index_col=0)

            print(result_file)

            with open(path, mode='w') as f:
                print(r'%%%%%%%%%%%', file=f)
                print(r'\subfloat[$f_{'+'{}'.format(fun_id)+r'}, '+'n={}'.format(dim)+r'$]{', file=f)
                print(r'\begin{tabular}{lccccc}', file=f)
                print(r'\rowcolor[rgb]{0.9, 0.9, 0.9} & ', file=f)
                print(r'\toprule', file=f)

                print(r' & ', end='', file=f)
                # for fun_id in all_id_funs:
                for sample_multiplier in multiplier_list:
                    # print(r'$f_{'+'{}'.format(fun_id)+r'}$', end='', file=f)
                    print(r'$\mathcal{A}_{'+'{}'.format(sample_multiplier)+r'}', end='', file=f)
                    
                    if sample_multiplier != multiplier_list[-1]:
                        print(' & ', end='', file=f)
                    else:
                        print(r'\\', file=f)

                print(r'\midrule', file=f)

                for separation_option in separation:
                    for surrogate_multiplier in surrogate_multiplier_list:
                            
                        for sampling_option in ['lhs', 'random']:
                            if sampling_option == 'random':
                                if separation_option == 'default':
                                    if surrogate_multiplier != '100n':
                                        if surrogate_multiplier != '200n':
                                            if surrogate_multiplier != '300n':
                                                continue
                                else: 
                                    continue
                            else:
                                if separation_option == 'new':
                                    if surrogate_multiplier != '100n':
                                        continue

                            print(r'$s^{\prime}='+'{}'.format(surrogate_multiplier)+r'$ '+'{}'.format(sampling_option), end=' & ', file=f)

                            for sample_multiplier in multiplier_list:
                                print('{:,.4f}'.format(result_file.at['{}_{}_{}'.format(surrogate_multiplier, sampling_option, separation_option), '{}'.format(sample_multiplier)]), end='', file=f)

                                if sample_multiplier != multiplier_list[-1]:
                                    print(' & ', end='', file=f)                
                                else:
                                    print(r'\\', file=f)

                print(r'\bottomrule', file=f)
                print(r'\end{tabular}}\\', file=f)
                print(r'%%%%%%%%%%%', file=f)
        


# 通常+サロゲートでVBS精度をデータフレーム化
def dataframize_vbs_precision(dim_list, surrogate_multiplier_list, multiplier_list):
    all_id_funs = range(1, 24+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    sampling = 'lhs'
    selector = 'hiearchical_regression'
    cross_valid_type = 'lopo_cv'
    per_metric = 'accuracy'
    dims = 'dims2_3_5_10'
    feature_selector = 'none'

    ela_feature_classes_other = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    ela_feature_classes_10 = 'basic_ela_distr_pca_limo_ic_disp_ela_level_ela_meta'

    output_dir_path = 'mse_output'
    os.makedirs(output_dir_path, exist_ok=True)

    is_first_time = True

    print(r'\caption{\small 各システム、$s^{\prime}$の組み合わせにおけるVBSの選択精度}')

    for dim in dim_list:

        df_dim = pd.DataFrame(0, index=surrogate_multiplier_list, columns=multiplier_list, )

        for sample_multiplier in multiplier_list:
            if sample_multiplier == 10:
                ela_feature_classes = ela_feature_classes_10
            else:
                ela_feature_classes = ela_feature_classes_other
            
            for surrogate_multiplier in surrogate_multiplier_list:

                score = 0
                count = 0

                for fun_id in all_id_funs:

                    vbs_file_path = os.path.join('alg_portfolio/ls{}/vbs_accuracy/f{}_DIM{}.csv'.format(sample_multiplier, fun_id, dim))
                    vbs_df = pd.read_csv(vbs_file_path, header=None)

                    # for sampling_option in ['lhs', 'random']:
                    for sampling_option in ['lhs']:

                        for sid in all_sid:
                            dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling, sample_multiplier, sid)
                            table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims
                            result_dir_path = os.path.join('as_results/{}_{}/ls{}_{}_{}_{}_{}'.format(surrogate_multiplier, sampling_option, sample_multiplier, selector, cross_valid_type, table_data_name, feature_selector))
                            

                            for instance_id in all_test_instance_id:

                                result_file_path = os.path.join(result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(fun_id, dim, instance_id))

                                selected_alg = pd.read_csv(result_file_path, header=None).iat[0,0]

                                if selected_alg in vbs_df.values:
                                    score += 1
                                count += 1

                df_dim.at[surrogate_multiplier, sample_multiplier] += score / count

        # print('VBS precision in dim{} is'.format(dim))
        # print(df_dim)

        
        print(r'%%%%%%%%%%%')
        print(r'\subfloat[$n='+'{}'.format(dim)+r'$]{')
        
        print(r'\begin{tabular}{l',end='')
        for sample_multiplier in multiplier_list:
            print(r'c', end='')
        print(r'}')

        # print(r'\rowcolor[rgb]{0.9, 0.9, 0.9}')
        print(r'\toprule')

        print(r' & ', end='')
        for sample_multiplier in multiplier_list:
            print(r'$\mathcal{A}_{'+'{}'.format(sample_multiplier)+r'} $', end='')
            
            if sample_multiplier != multiplier_list[-1]:
                print(' & ', end='')
            else:
                print(r'\\')

        print(r'\midrule')

        for surrogate_multiplier in surrogate_multiplier_list:
            print(r'$s^{\prime}='+'{}'.format(surrogate_multiplier)+r'$', end=' & ')

            for sample_multiplier in multiplier_list:
                print('{:,.4f}'.format(df_dim.at[surrogate_multiplier, sample_multiplier]), end='')

                if sample_multiplier != multiplier_list[-1]:
                    print(' & ', end='')
                else:
                    print(r'\\')

        print(r'\bottomrule')
        print(r'\end{tabular}}\\')
        print(r'%%%%%%%%%%%')


# MSE
# 次元ごとに分けてdfをcsvファイル化, dfはサロゲートサイズ(+sampling option)×関数
@click.command()
# @click.option('--surrogate_number', '-sur', required=False, default=2000, type=int, help='Number of surrogated X')
@click.option('--mse_or_r2', '-mser2', required=True, default='mse', type=str, help='mse or R2 option')
def main(mse_or_r2):
    dims = 'dims2_3_5_10'
    sampling_method = 'lhs'
    # sample_multiplier = 50
    # per_metric = 'RMSE'
    per_metric = 'accuracy'
    feature_selector = 'none'
    n_features_to_select = 0
    
    
    dim_list = [2, 3, 5, 10]
    surrogate_multiplier_list = ['100n', '200n', '300n', '400n', '500n', '1000n']
    multiplier_list = [10, 15, 20, 25, 50]

    separation_option = ['default', 'new']

    # BBOB_calculate()

    dataframize_mse(dim_list, surrogate_multiplier_list, multiplier_list, mse_or_r2, separation_option)

    # dataframize_vbs_precision(dim_list, surrogate_multiplier_list, multiplier_list)


if __name__ == '__main__':
    main()