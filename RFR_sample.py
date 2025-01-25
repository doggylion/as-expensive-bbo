from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import os
import click
from pyDOE import lhs
import numpy as np
import math
import scipy.spatial.distance as distance

import bbobbenchmarks
import fgeneric

# ./sample_data/lhs_multiplier{10,15,20,25,50}_sid{0~30}を読み込んでデータスプリット
# あらかじめ決めた個数分[-5,5]の範囲で乱数生成, 座標として記録, RFRで回帰予測
# ./surrogated_sample_data/に保存

def BBOB_calculate(df_csv_file, fun_id, instance_id, bbob_suite='bbob', output_folder='true_surrogeted_sample'):
        datapath = 'tmp'
        opts = dict(algid='PUT ALGORITHM NAME',
                comments='PUT MORE DETAILED INFORMATION, PARAMETER SETTINGS ETC')
        f = fgeneric.LoggingFunction(datapath, **opts)
    
        f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=instance_id))

        # print(f.evalfun(df_csv_file))

        return f.evalfun(df_csv_file)

def default_RFR_sample(surrogate_multiplier, surrogate_sample):
    sample_path = os.path.join('./sample_data/0')
    s_sample_path = os.path.join('./sample_data/{}n_{}'.format(surrogate_multiplier, surrogate_sample))
    os.makedirs(s_sample_path, exist_ok=True)

    for multiplier in [10,15,20,25,50]:
        for sid in range(31):
            sample_dir = os.path.join(sample_path, 'lhs_multiplier{}_sid{}'.format(multiplier, sid))
            s_sample_dir = os.path.join(s_sample_path, 'lhs_multiplier{}_sid{}'.format(multiplier, sid))
            os.makedirs(s_sample_dir, exist_ok=True)

            for function in range(1,24+1):
                for dim in [2,3,5,10]:
                    surrogate_number = surrogate_multiplier*dim
                    for instance in range(1,5+1):
                        sample_data_path = os.path.join(sample_dir,'x_f_data_bbob_f{}_DIM{}_i{}.csv'.format(function,dim,instance))
                        s_sample_data_path = os.path.join(s_sample_dir,'x_f_data_bbob_f{}_DIM{}_i{}.csv'.format(function,dim,instance))

                        df = pd.read_csv(sample_data_path,header=None)

                        y = df.iloc[0:multiplier*dim, 0]
                        X = df.iloc[0:multiplier*dim, 1:dim+1]

                        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                        model = RFR(random_state=sid)

                        # model.fit(X_train,y_train)
                        model.fit(X, y)

                        if surrogate_sample == 'random':
                            addicted_X = pd.DataFrame(0, index=range(multiplier*dim,surrogate_number), columns=range(1,dim+1))

                            for addicted_index in range(surrogate_number-multiplier*dim):
                                for i in range(dim):
                                    addicted_X.iat[addicted_index, i] += random.uniform(-5, 5)

                        elif surrogate_sample == 'lhs':
                            sample = lhs(dim, (surrogate_multiplier-multiplier)*dim, criterion='center')

                            lbound = np.full(dim, -5.)
                            ubound = np.full(dim, 5.)
                            sample = (ubound - lbound) * sample + lbound

                            addicted_X = pd.DataFrame(data=sample, index=range(multiplier*dim,surrogate_number), columns=range(1,dim+1))

                            # print(sample,'\n',type(sample))

                        addicted_y = model.predict(addicted_X)
                            
                        df_addicted_y = pd.DataFrame(data=addicted_y, index=range(multiplier*dim,surrogate_number), columns=[0])

                        df_addicted = pd.concat([df_addicted_y,addicted_X], axis=1)

                        # print(df,'\n',df_addicted)

                        surrogated_df = pd.concat([df,df_addicted], axis=0)

                        print(surrogated_df)

                        surrogated_df.to_csv(s_sample_data_path, header=False, index=False)

                        print(pd.read_csv(s_sample_data_path, header=None))


# 仮名関数, 決められた方式でs'だけサンプリングした後sを分割して評価, 残りのs'-sを機械学習で回帰予測　他は同様
def RFR_sample(surrogate_multiplier, surrogate_sample, multiplier_list):
    s_sample_path = os.path.join('./sample_data/{}n_{}_new'.format(surrogate_multiplier, surrogate_sample)) #仮名ディレクトリ
    os.makedirs(s_sample_path, exist_ok=True)

    for multiplier in multiplier_list:
        for sid in range(31):
            # sample_dir = os.path.join(sample_path, 'lhs_multiplier{}_sid{}'.format(multiplier, sid))
            s_sample_dir = os.path.join(s_sample_path, 'lhs_multiplier{}_sid{}'.format(multiplier, sid)) #後で変更
            os.makedirs(s_sample_dir, exist_ok=True)

            for function in range(1,24+1):
                for dim in [2,3,5,10]:
                    surrogate_number = surrogate_multiplier*dim
                    for instance in range(1,5+1):
                        # sample_data_path = os.path.join(sample_dir,'x_f_data_bbob_f{}_DIM{}_i{}.csv'.format(function,dim,instance))
                        s_sample_data_path = os.path.join(s_sample_dir,'x_f_data_bbob_f{}_DIM{}_i{}.csv'.format(function,dim,instance))

                        if surrogate_sample == 'random':
                            X = pd.DataFrame(0, index=range(surrogate_number), columns=range(1,dim+1))

                            for index in range(surrogate_number):
                                for i in range(dim):
                                    X.iat[index, i] += random.uniform(-5, 5)

                        elif surrogate_sample == 'lhs':
                            sample = lhs(dim, surrogate_multiplier*dim, criterion='center')

                            lbound = np.full(dim, -5.)
                            ubound = np.full(dim, 5.)
                            sample = (ubound - lbound) * sample + lbound

                            X = pd.DataFrame(data=sample, index=range(surrogate_number), columns=range(1,dim+1))

                            # print(sample,'\n',type(sample))
                        
                        # print(X)

                        # df = pd.read_csv(sample_data_path,header=None)
                        sparce_df, addicted_X = finding_sparse_x(X, multiplier, dim, function, instance, surrogate_number)

                        sparce_y = sparce_df.iloc[0:multiplier*dim, 0]
                        sparce_X = sparce_df.iloc[0:multiplier*dim, 1:dim+1]

                        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                        model = RFR(random_state=sid)

                        # model.fit(X_train,y_train)
                        model.fit(sparce_X, sparce_y)

                        # if surrogate_sample == 'random':
                        #     addicted_X = pd.DataFrame(0, index=range(multiplier*dim,surrogate_number), columns=range(1,dim+1))

                        #     for addicted_index in range(surrogate_number-multiplier*dim):
                        #         for i in range(dim):
                        #             addicted_X.iat[addicted_index, i] += random.uniform(-5, 5)

                        # elif surrogate_sample == 'lhs':
                        #     sample = lhs(dim, (surrogate_multiplier-multiplier)*dim, criterion='center')

                        #     lbound = np.full(dim, -5.)
                        #     ubound = np.full(dim, 5.)
                        #     sample = (ubound - lbound) * sample + lbound

                        #     addicted_X = pd.DataFrame(data=sample, index=range(multiplier*dim,surrogate_number), columns=range(1,dim+1))

                        #     # print(sample,'\n',type(sample))

                        addicted_y = model.predict(addicted_X)
                            
                        df_addicted_y = pd.DataFrame(data=addicted_y, index=range(multiplier*dim,surrogate_number), columns=[0])

                        df_addicted = pd.concat([df_addicted_y,addicted_X], axis=1)

                        # print(df,'\n',df_addicted)

                        surrogated_df = pd.concat([sparce_df,df_addicted], axis=0)

                        print(surrogated_df)

                        surrogated_df.to_csv(s_sample_data_path, header=False, index=False)

                        # print(pd.read_csv(s_sample_data_path, header=None))

# 入力されたX(サイズはs')から疎なs個の解を分割し評価, 出力は評価済みのs個の解のdfと分割されたs'-s個の要素のdf
def finding_sparse_x(X, multiplier, dim, fun_id, instance_id, surrogate_number):
    Sparse_x_list = []

    sparce_df = pd.DataFrame(columns=range(1,dim+1))
    
    # Greedy
    first_sparce = random.randint(0, len(X)-1)
    sparce_df = pd.concat([sparce_df, pd.DataFrame([X.iloc[first_sparce]])])
    X = X.drop([first_sparce])

    # distance_array = distance.squareform(distance.pdist(X_nparray))

    while len(sparce_df) < multiplier*dim:
        # print('X=\n{}\n{}'.format(X.to_numpy(), type(X.to_numpy())))
        # print('sparce_df=\n{}\n{}'.format(sparce_df.to_numpy(), type(sparce_df.to_numpy())))
        distance_array = distance.cdist(X.to_numpy(), sparce_df.to_numpy())
        # print(distance_array)
        distance_list = []
        for index in range(len(X)):
            # print(row)
            # if index in Sparse_x_list:
            #     distance_list.append(0)
            #     continue

            distance_list.append(np.sum(distance_array[index]))

        # Sparse_x_list.append(np.argmax(distance_list))

        sparce_index = np.argmax(distance_list)
        
        sparce_df = pd.concat([sparce_df, pd.DataFrame([X.iloc[sparce_index]])])
        X = X.drop(X.index[sparce_index])

    # for index, row in X.iterrows():
    #     if index in Sparse_x_list:
    #         sparce_df = pd.concat([sparce_df, pd.DataFrame([row])])
    #     else:
    #         addicted_X = pd.concat([addicted_X, pd.DataFrame([row])])
    
    f = BBOB_calculate(sparce_df, fun_id, instance_id, bbob_suite='bbob', output_folder='true_surrogeted_sample')
    f_sparse = pd.Series(f, name=0)

    # sparce_df = sparce_df.reindex(columns=range(dim+1))

    # print('sparce_df=\n{}\n{}'.format(sparce_df,type(sparce_df)))
    # print('f_sparse=\n{}\n{}'.format(f_sparse,type(f_sparse)))
    
    sparce_df = pd.concat([sparce_df.reset_index(drop=True), pd.DataFrame(f_sparse)], axis=1).reindex(columns=range(dim+1))

    X.index = range(multiplier*dim, surrogate_number)

    # print('sparce_df=\n{}\n{}'.format(sparce_df,type(sparce_df)))
    # print('addicted_X=\n{}\n{}'.format(addicted_X,type(addicted_X)))
    return sparce_df, X

@click.command()
# @click.option('--surrogate_number', '-sur', required=False, default=2000, type=int, help='Number of surrogated X')
@click.option('--surrogate_multiplier', '-surm', required=True, default=100, type=int, help='Multiplier of surrogated X')
@click.option('--surrogate_sample', '-surs', required=True, default='random', type=str, help='the way of surrogate')
@click.option('--separation_option', '-spop', required=True, default='default', type=str, help='the way of separating s^prime')
# def main(surrogate_number):
def main(surrogate_multiplier, surrogate_sample, separation_option):
    multiplier_list = [10, 15, 20, 25, 50]

    if separation_option == 'default':
        default_RFR_sample(surrogate_multiplier, surrogate_sample)
    elif separation_option == 'new': #仮名
        RFR_sample(surrogate_multiplier, surrogate_sample, multiplier_list) #仮名

if __name__  == "__main__":
    main()