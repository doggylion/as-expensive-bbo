import os
import click
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

#行の表示数の上限を撤廃
pd.set_option('display.max_rows', None)
#列の表示数の上限を撤廃
pd.set_option('display.max_columns', None)

@click.command()
# @click.option('--surrogate_number', '-sur', required=False, default=2000, type=int, help='Number of surrogated X')
@click.option('--surrogate_multiplier', '-surm', required=True, default=100, type=int, help='Multiplier of surrogated X')
@click.option('--surrogate_sample', '-surs', required=True, default='random', type=str, help='the way of surrogate')
@click.option('--separation', '-spr', required=True, default='none', type=str, help='separate or not')
def main(surrogate_multiplier, surrogate_sample, separation):
    sampling_method = 'lhs'
    
    comp_option = 'comp'

    bbob_suite = 'bbob'
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)

    all_dims = [2, 3, 5, 10]

    all_sample_multipliers = [10, 15, 20, 25, 50]

    all_instance_ids = [1, 2, 3, 4, 5]

    all_sid = range(31)

    if comp_option == 'comp':
        if separation == 'none':
            option = '{}n_{}'.format(surrogate_multiplier, surrogate_sample)
        else:
            option = '{}n_{}_{}'.format(surrogate_multiplier, surrogate_sample, separation)


    for dim in all_dims:

        for sample_multiplier in all_sample_multipliers:
            ap_name = 'ls{}'.format(sample_multiplier)
            ap_dir_path = os.path.join('./alg_portfolio', ap_name)
            ap_config_file_path = os.path.join(ap_dir_path, 'ap_config.csv')
            ap_algs = np.loadtxt(ap_config_file_path, delimiter=",", comments="#", dtype=np.str)
            
            df_multiplier = df_alg = pd.DataFrame(columns=['arg', 'labels', 'times'])

            for alg in ap_algs:
                df_alg = pd.DataFrame(columns=['arg', 'labels', 'times'])
                    
                for fun_id in all_fun_ids:

                    for instance_id in all_instance_ids:

                        for sid in all_sid:
                            dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
                            file_path = os.path.join('feature_impotance', option, 'DIM{}/f{}/{}/{}.csv'.format(dim, fun_id, alg, dir_sampling_method, instance_id))

                            df = pd.read_csv(file_path, index_col=0)

                            logger.info("Read: dim{}, multiplier{}, {}, f{}, iid{}, sid{}".format(dim, sample_multiplier, alg, fun_id, instance_id, sid))

                            if len(df_alg) == 0:
                                df_alg = pd.concat([df_alg, df])
                                df_alg['times'][:] = 1
                                print("check1")
                            else:
                                for index1, row1 in df.iterrows():
                                    check = True
                                    for index2, row2 in df_alg.iterrows():
                                        if row1['labels'] == row2['labels']:
                                            df_alg.at[index2, 'arg'] += row1['arg']
                                            df_alg.at[index2, 'times'] += 1
                                            check = False
                                    
                                    if check:
                                        row1['times'] = 1
                                        df_alg.loc[len(df_alg)] = row1.values
                                        print("check2")



                if len(df_multiplier) == 0:
                    df_multiplier = pd.concat([df_multiplier, df_alg])
                else:
                    for index1, row1 in df_alg.iterrows():
                        check = True
                        for index2, row2 in df_multiplier.iterrows():
                            if row1['labels'] == row2['labels']:
                                row2['arg'] += row1['arg']
                                row2['times'] += row1['times']
                                check = False
                                    
                        if check:
                            df_multiplier.loc[len(df_multiplier)] = row1.values

                
                df_alg_rank = df_alg.copy()

                for i in range(len(df_alg_rank)):
                    df_alg_rank.at[i, 'arg'] = df_alg_rank.at[i, 'arg'] / (df_alg_rank.at[i, 'times'])

                print(df_alg_rank)
                df_alg_rank.sort_values('arg', inplace=True)
                print(df_alg_rank)

                folder_path = os.path.join('feature_impotance_result', option, 'DIM{}/ls{}'.format(dim, sample_multiplier))
                os.makedirs(folder_path, exist_ok=True)
                text_path = os.path.join(folder_path, '{}.txt'.format(alg))

                f = open(text_path, 'w')

                f.write(r'%%%%%%%%%%%')
                f.write('\n')
                f.write(r'\subfloat[$s^{\prime}='+'{}'.format(option)+r'$]{')
                f.write('\n')
                f.write(r'\begin{tabular}{lc}')
                f.write('\n')
                f.write(r'\toprule')
                f.write('\n')
                f.write(r'Feature & Rank\\')
                f.write('\n')
                f.write(r'\midrule')
                f.write('\n')

                # for i in range(len(df_alg_rank)):
                for index, row in df_alg_rank.iterrows():
                    f.write('{}'.format(row['labels'].replace('_', '-'))+r' & $'+'{:,.2f}'.format(row['arg'])+r'$\\')
                    f.write('\n')

                f.write(r'\bottomrule')
                f.write('\n')
                f.write(r'\end{tabular}}\\')
                f.write('\n')
                f.write(r'%%%%%%%%%%%')

                logger.info("Done: {}, {}, {}".format(dim, sample_multiplier, alg))

            df_multiplier_rank = df_multiplier.copy()

            for i in range(len(df_multiplier)):
                df_multiplier_rank.at[i, 'arg'] = df_multiplier_rank.at[i, 'arg'] / (df_multiplier_rank.at[i, 'times'])
            
            df_multiplier_rank.sort_values('arg', inplace=True)

            folder_path = os.path.join('feature_impotance_result', option, 'DIM{}'.format(dim))
            os.makedirs(folder_path, exist_ok=True)
            text_path = os.path.join(folder_path, '{}.txt'.format(sample_multiplier))


            f = open(text_path, 'w')

            f.write(r'%%%%%%%%%%%')
            f.write('\n')
            f.write(r'\subfloat[$s^{\prime}='+'{}'.format(option)+r'$]{')
            f.write('\n')
            f.write(r'\begin{tabular}{lc}')
            f.write('\n')
            f.write(r'\toprule')
            f.write('\n')
            f.write(r'Feature & Rank\\')
            f.write('\n')
            f.write(r'\midrule')
            f.write('\n')

            # for i in range(len(df_alg_rank)):
            for index, row in df_multiplier_rank.iterrows():
                    f.write('{}'.format(row['labels'].replace('_', '-'))+r' & $'+'{:,.2f}'.format(row['arg'])+r'$\\')
                    f.write('\n')

            f.write(r'\bottomrule')
            f.write('\n')
            f.write(r'\end{tabular}}\\')
            f.write('\n')
            f.write(r'%%%%%%%%%%%')


if __name__ == '__main__':
    main()
