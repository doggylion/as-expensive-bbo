#!/usr/bin/env python
import subprocess
import click

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
    # per_metric = 'RMSE'
    sampling_method = 'lhs'
    # sample_multiplier = 10
    feature_selector = 'none'
    n_features_to_select = 0
    
    # for ap_name in ['kt', 'dlvat', 'jped', 'bmtp', 'mk', 'ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']:
    # for ap_name in ['ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']:
    # for ap_name in ['ls0', 'ls10', 'ls15', 'ls20', 'ls25', 'ls50']:
    for ap_name in ['ls10', 'ls15', 'ls20', 'ls25', 'ls50']:
        for sample_multiplier in [10, 15, 20, 25, 50]:
            if ap_name != 'ls0':
                if ap_name != 'ls{}'.format(sample_multiplier):
                    continue

            for sid in range(0, 31):
                dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                # for selector in ['multiclass_classification', 'hiearchical_regression', 'pairwise_classification', 'pairwise_regression', 'clustering']:
                for selector in ['hiearchical_regression']:
                    # for cross_valid_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
                    for cross_valid_type in ['lopo_cv']:
                        for per_metric in ['accuracy']:
                            s_arg1 = 'arg1=--ap_name {}'.format(ap_name)
                            s_arg2 = 'arg2=--dir_sampling_method {}'.format(dir_sampling_method)
                            s_arg3 = 'arg3=--as_run_id {}'.format(sid)
                            if sample_multiplier == 10:
                                s_arg4 = 'arg4=--ela_feature_classes {}'.format(ela_feature_classes_10)
                            else:
                                s_arg4 = 'arg4=--ela_feature_classes {}'.format(ela_feature_classes_other)

                            s_arg5 = 'arg5=--dims {}'.format(dims)
                            s_arg6 = 'arg6=--selector {}'.format(selector)
                            s_arg7 = 'arg7=--cross_valid_type {}'.format(cross_valid_type)
                            s_arg8 = 'arg8=--feature_selector {}'.format(feature_selector)
                            s_arg9 = 'arg9=--n_features_to_select {}'.format(n_features_to_select)
                            s_arg10 = 'arg10=--per_metric {}'.format(per_metric)

                            # s_arg13 = 'arg13=--surrogate_number {}'.format(surrogate_number)
                            s_arg13 = 'arg13=--surrogate_multiplier {}'.format(surrogate_multiplier)
                            s_arg14 = 'arg14=--surrogate_sample {}'.format(surrogate_sample)
                            s_arg15 = 'arg15=--comp_option {}'.format(comp_option)
                            s_arg16 = 'arg16=--separation {}'.format(separation)

                            if cross_valid_type  == 'loio_cv':
                                for dim in [2, 3, 5, 10]:
                                    for left_instance_id in [1, 2, 3, 4, 5]:
                                        s_arg11 = 'arg11=--dim {}'.format(dim)
                                        s_arg12 = 'arg12=--left_instance_id {}'.format(left_instance_id)
                                        s_args = ','.join([s_arg1, s_arg2, s_arg3, s_arg4, s_arg5, s_arg6, s_arg7, s_arg8, s_arg9, s_arg10, s_arg11, s_arg12, s_arg13, s_arg14, s_arg15, s_arg16])         
                                        subprocess.run(['qsub', '-l', 'walltime=72:00:00', '-v', s_args, 'job_as.sh'])
                            elif cross_valid_type  == 'lopo_cv':
                                for dim in [2, 3, 5, 10]:
                                    for left_fun_id in range(1, 24+1):
                                        s_arg11 = 'arg11=--dim {}'.format(dim)
                                        s_arg12 = 'arg12=--left_fun_id {}'.format(left_fun_id)
                                        s_args = ','.join([s_arg1, s_arg2, s_arg3, s_arg4, s_arg5, s_arg6, s_arg7, s_arg8, s_arg9, s_arg10, s_arg11, s_arg12, s_arg13, s_arg14, s_arg15, s_arg16])         
                                        subprocess.run(['qsub', '-l', 'walltime=72:00:00', '-v', s_args, 'job_as.sh'])
                            elif cross_valid_type  == 'lopoad_cv':
                                for left_dim in [2, 3, 5, 10]:
                                    for left_fun_id in range(1, 24+1):
                                        s_arg11 = 'arg11=--left_dim {}'.format(left_dim)
                                        s_arg12 = 'arg12=--left_fun_id {}'.format(left_fun_id)
                                        s_args = ','.join([s_arg1, s_arg2, s_arg3, s_arg4, s_arg5, s_arg6, s_arg7, s_arg8, s_arg9, s_arg10, s_arg11, s_arg12, s_arg13, s_arg14, s_arg15, s_arg16])
                                        subprocess.run(['qsub', '-l', 'walltime=72:00:00', '-v', s_args, 'job_as.sh'])

if __name__ == '__main__':
    main()