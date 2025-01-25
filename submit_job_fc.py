#!/usr/bin/env python
import subprocess
import click

@click.command()
# @click.option('--surrogate_number', '-sur', required=False, default=2000, type=int, help='Number of surrogated X')
@click.option('--surrogate_multiplier', '-surm', required=True, default=100, type=int, help='Multiplier of surrogated X')
@click.option('--sample_multiplier', '-sample', required=True, default=10, type=int, help='Number of surrogated X')
@click.option('--surrogate_sample', '-surs', required=True, default='random', type=str, help='the way of surrogate')
@click.option('--separation', '-spr', required=True, default='none', type=str, help='separate or not')
def main(surrogate_multiplier,sample_multiplier, surrogate_sample, separation):
    sampling_method = 'lhs'
    # sample_multiplier = 50
    
    for sid in range(0, 31):
        # for sample_multiplier in range(1,11):
        # for sample_multiplier in [10]:
        dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)                
        for ela_feature_class in ['basic', 'ela_distr', 'pca', 'limo', 'ic', 'disp', 'nbc', 'ela_level', 'ela_meta']:
            for dim in [2, 3, 5, 10]:
                for fun_id in range(1, 24+1):
                    s_arg1 = 'arg1=--dir_sampling_method {}'.format(dir_sampling_method)
                    s_arg2 = 'arg2=--ela_feature_class {}'.format(ela_feature_class)
                    s_arg3 = 'arg3=--dim {}'.format(dim)
                    s_arg4 = 'arg4=--fun_id {}'.format(fun_id)
                    s_arg5 = 'arg5=--sample_path sample_data'
                    s_arg6 = 'arg6=--surrogate_multiplier {}'.format(surrogate_multiplier)
                    s_arg7 = 'arg7=--surrogate_sample {}'.format(surrogate_sample)
                    s_arg8 = 'arg8=--separation {}'.format(separation)
                    s_args = ','.join([s_arg1, s_arg2, s_arg3, s_arg4, s_arg5, s_arg6, s_arg7, s_arg8])
                    subprocess.run(['qsub', '-l', 'walltime=72:00:00', '-v', s_args, 'job_fc.sh']) 

if __name__ == '__main__':
    main()