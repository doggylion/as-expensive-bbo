import matplotlib.pyplot as plt
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def plot_n2(dim_list, surrogate_multiplier_list, sample_multiplier_list):
    all_id_funs = range(1, 24+1)

    all_test_instance_id = range(1, 5+1)

    all_sid = range(0, 31)

    sampling = 'lhs'

    separation = ['new']
    
    output_dir_path = 'plot_scatter_n2'
    os.makedirs(output_dir_path, exist_ok=True)

    
    

    for separation_option in separation:
        for sampling_option in ['lhs']: 
            for surrogate_multiplier in surrogate_multiplier_list:
                sample_path = os.path.join('sample_data', '{}_{}_{}'.format(surrogate_multiplier, sampling_option, separation_option))  

                for sample_multiplier in sample_multiplier_list:
                    for sid in all_sid:
                    
                        sample_dir = os.path.join(sample_path, '{}_multiplier{}_sid{}'.format(sampling, sample_multiplier, sid))
                            
                        for dim in dim_list:
                            for fun_id in all_id_funs:
                                for instance_id in all_test_instance_id:

                                    file_path = os.path.join(sample_dir, 'x_f_data_bbob_f{}_DIM{}_i{}.csv'.format(fun_id, dim, instance_id))

                                    fig_file_path = os.path.join(output_dir_path, 'sample{}_sid{}_f{}_DIM{}_i{}.pdf'.format(sample_multiplier, sid, fun_id, dim, instance_id))
                            
                                    df_csv_file = pd.read_csv(file_path, header=None)

                                    x1_s = df_csv_file.loc[0:sample_multiplier*dim, 1].values.tolist()
                                    x2_s = df_csv_file.loc[0:sample_multiplier*dim, 2].values.tolist()

                                    x1_sd = df_csv_file.loc[sample_multiplier*dim:, 1].values.tolist()
                                    x2_sd = df_csv_file.loc[sample_multiplier*dim:, 2].values.tolist()

                                    fig = plt.figure()

                                    ax = fig.add_subplot(1,1,1)

                                    ax.scatter(x1_s, x2_s, c='red')
                                    ax.scatter(x1_sd, x2_sd, c='blue')

                                    ax.set_title('sample={}, sid={}, f{}, n={}, instance={}'.format(sample_multiplier, sid, fun_id, dim, instance_id))
                                    ax.set_xlabel('x1')
                                    ax.set_ylabel('x2')

                                    ax.set_xlim(left=-5, right=5)
                                    ax.set_ylim(bottom=-5, top=5)

                                    ax.grid(True)

                                    plt.savefig(fig_file_path, bbox_inches='tight')
                                    plt.close()

                                    logger.info("A figure was generated: %s", fig_file_path)

    return 0


# 新手法の2次元の場合の散布図を生成, サロゲートを適用するs'-s個と実際に評価したs個を色分け
if __name__ == '__main__':    
    bbob_suite = 'bbob'
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)
    dim_list = [2]
    feature_selector = 'none'

    ela_feature_classes = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    dims = 'dims2_3_5_10'

    per_metric = 'accuracy'
    sampling_method = 'lhs'
    sample_multiplier_list = [10, 15, 20, 25, 50]
    surrogate_multiplier_list = ['100n']

    plot_n2(dim_list, surrogate_multiplier_list, sample_multiplier_list)