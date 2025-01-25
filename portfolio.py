
import numpy as np
import os
import logging
import pandas as pd
import random
import copy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


method = 'greedy'


dimension = [2, 3, 5, 10]
sampling = [0, 10, 15, 20, 25, 50]
R_size = 200
F = list(range(24))


# def find_ICARUS(bbob_file_path, A_bst, dim, F):

#     t_F = []

#     for function in F:
        
#         t_f = {}
             
#         df = pd.read_csv(bbob_file_path+'/f{}_DIM{}.csv'.format(function+1, dim), usecols=['alg', 'RMSE'])

#         for i in range(K):
#             d = {"{}".format(df.iat[i, 0]): df.iat[i, 1]}
#             if (i == 0)or(df.iat[i, 1] == df.iat[i-1, 1]):
#                 A_bst.append(df.iat[i, 0])
            
#             t_f = {**t_f,**d}
        
#         t_F.append(t_f)

#         logger.info("Data of function{} with dim{} have been imported".format(function, dim))

#     return set(A_bst), t_F


def find_top3(bbob_file_path, A_bst, dim, F):
    for function in F:
             
        df = pd.read_csv(bbob_file_path+'/f{}_DIM{}.csv'.format(function+1, dim), usecols=['alg', 'RMSE'])

        flag = 1
        i = 0
        while flag == 1:
            if (i < 3)or(df.iat[i, 1] == df.iat[0, 1]):
                A_bst.append(df.iat[i, 0])
            else: 
                flag = 0
            i += 1

        # logger.info("Data of function{} with dim{} have been imported".format(function, dim))

    return set(A_bst)


def aggregate(sampling_multiplier):

    t_F = []

    R = []

    
    for dim in dimension:
        for function in F:
            
            t_f = {}
                
            df = pd.read_csv('pp_bbob_exdata/ranking_feval100D/sampling{}D/dim{}/f{}_DIM{}.csv'.format(sampling_multiplier, dim, function+1, dim), usecols=['alg', 'accuracy'])

            for i in range(R_size):
                d = {"{}".format(df.iat[i, 0]): i}
                # d = {"{}".format(df.iat[i, 0]): df.iat[i, 1]}
                if df.iat[i, 0] not in R:
                    R.append(df.iat[i, 0])
                
                t_f = {**t_f,**d}
            
            t_F.append(t_f)

            # logger.info("Data of function{} with dim{} have been imported".format(function, dim))

    return R, t_F


def greedy(k, sampling_multiplier):
    np.set_printoptions(threshold=np.inf)

    # for param in [10, 20]:
    A = []
    A_bst = []
    R = []
    t_F = []

    R, t_F = aggregate(sampling_multiplier)
    A = initialize(R, k)

    score_A = score(A, t_F)
    min_score_A = score_A

    print("Initial score of A = {}".format(score_A))
    i = 0
    while i != 3:
        for ad in R:
            for a in A:
                A_new = copy.deepcopy(A)
                if(ad not in A):
                    A_pro = copy.deepcopy(A)
                    A_pro.remove(a)
                    A_pro.append(ad)
                    score_A_pro = score(A_pro, t_F)
                    if score_A > score_A_pro:
                        A = copy.deepcopy(A_pro)
                        score_A = score_A_pro
                        


        # print(score_A)

        if min_score_A == score_A:
            i += 1
        elif min_score_A > score_A:
            min_score_A = score_A
            i = 0

    print("Algorithm portfolio is constructed with k = {}, score = {}".format(k, score_A))
            
    return A


def initialize(R, k):
    random.seed(k)

    A = []

    for i in range(k):
        A.append(R[random.randint(0, len(R)-1)])
        # A.append(R[i])
        
    return A


def score(A, t_F):
    m_A = 0
    c = 1 / (R_size*len(dimension)*len(F))

    for i, t_f in enumerate(t_F):
        score_A = 1
        score_a = 1

        for a in A:
            if a in t_f:
                score_a = c * t_f[a]
            
                if score_A > score_a:
                    score_A = score_a
                # print(a, t_f[a], score_A)

        m_A += score_A
        
    return m_A


def icarus():
    np.set_printoptions(threshold=np.inf)

    for param in [10, 20]:
        for dim in [2, 3, 5, 10]:

            A = []
            A_bst = []
            F_uslv = F
            q = len(F_uslv)


            logger.info('==================== Configurate the algorithm portfolio from performance with param={},D={} ===================='.format(param, dim))
            # 2行目
            bbob_file_path = './pp_bbob_exdata/ranking_accuracy_{}D/dim_{}'.format(param, dim)
            A_bst, t_F = find_ICARUS(bbob_file_path, A_bst, dim, F)

            print("A_bst = ", A_bst)
            print("t_F = ", t_F)

            while q > 0:
                B = np.zeros((len(A_bst), len(A_bst)))
                C = np.zeros((len(A_bst), len(A_bst)))
                U = []

                # logger.info('==================== C matrix ====================')
                for function in F_uslv:
                    print("function = ", function)
                    for i, ai in enumerate(A_bst):
                        print("i = {}\n".format(i))
                        if ai in t_F[function]:
                            for j, aj in enumerate(A_bst):
                                print("j = ", j)
                                if aj in t_F[function]:
                                    print("ai = ",t_F[function].get(ai), "aj = ", t_F[function].get(aj))
                                    if float(t_F[function].get(ai)) > float(t_F[function].get(aj)):
                                        C[i][j] += 1
                            
                                else:
                                    C[i][j] += 1
                            
                print("C = ", C)

                # logger.info('==================== B matrix ====================')
                for i, ai in enumerate(A_bst):
                    for j, aj in enumerate(A_bst):
                        if C[i][j] == C[j][i]:
                            B[i][j] += 1
                print("B = ", B)
    
                B = C + (B @ B)

                print("B + BB = ", B)

                v = input("続行するには何かキーを押してください > ")

                for i, ai in enumerate(A_bst):
                    flag = 1
                    for j, aj in enumerate(A_bst):
                        if B[i][j] != 1:
                            flag = 0
                    
                    if flag == 1:
                        U.append(ai)
                
                # print("U = ", U)
                
                new_F_uslv = []
                for function in F_uslv:
                    for i, ai in enumerate(A_bst):
                        if ai not in t_F[function]:
                            new_F_uslv.append(function)
                            break
                
                F_uslv = new_F_uslv
                A = A + U
                q = len(F_uslv)

                print(q)
            
            print("the algorithm portfolio with param={},D={}".format(param, dim), A)


def top3():
    np.set_printoptions(threshold=np.inf)

    for param in [10, 20]:
        for dim in [2, 3, 5, 10]:

            A = []
            A_bst = []


            # logger.info('==================== Configurate the algorithm set from performance with param={},D={} ===================='.format(param, dim))

            bbob_file_path = './pp_bbob_exdata/ranking_accuracy_{}D/dim_{}'.format(param, dim)
            A_bst = find_top3(bbob_file_path, A_bst, dim, F)

            # print("A_bst_dim{} = ".format(dim), A_bst)

            if dim == 2:
                A_2 = A_bst
            elif dim == 3:
                A_3 = A_bst
            elif dim == 5:
                A_5 = A_bst
            else:
                A_10 = A_bst

        A = A_2 & A_3 & A_5 & A_10
            
        print("the algorithm portfolio with param={} ({} Algorithms)\n".format(param, len(A)), A)





def run():
    method = 'greedy'
    
    # Size of algorithm portfolio with greedy search
    for k in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
        if method == 'ICARUS':
            icarus()
        elif method == 'top3':
            top3()
        elif method == 'greedy':
            A = greedy(k)
            print(A)


if __name__ == '__main__':
    run()