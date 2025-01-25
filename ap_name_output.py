import os
from csv import reader

def run():
    ap_dir = './alg_portfolio'
    # ap_list = ['ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']
    ap_list = ['ls0', 'ls10', 'ls15', 'ls20', 'ls25', 'ls50']


    print(r'\begin{tabular}{ll}')
    print(r'\toprule')
    print(r'ポートフォリオ & 探索アルゴリズム\\')
    print(r'\midrule')

    for i, ap_name in enumerate(ap_list):
        # if ap_name == 'ls2':
        #     print(r'$\mathcal{A}_2$ ', end='& ')
        # elif ap_name == 'ls4':
        #     print(r'$\mathcal{A}_4$ ', end='& ')
        # elif ap_name == 'ls6':
        #     print(r'$\mathcal{A}_6$ ', end='& ')
        # elif ap_name == 'ls8':
        #     print(r'$\mathcal{A}_8$ ', end='& ')
        # elif ap_name == 'ls10':
        #     print(r'$\mathcal{A}_{10}$ ', end='& ')
        # elif ap_name == 'ls12':
        #     print(r'$\mathcal{A}_{12}$ ', end='& ')
        # elif ap_name == 'ls14':
        #     print(r'$\mathcal{A}_{14}$ ', end='& ')
        # elif ap_name == 'ls16':
        #     print(r'$\mathcal{A}_{16}$ ', end='& ')
        # elif ap_name == 'ls18':
        #     print(r'$\mathcal{A}_{18}$ ', end='& ')
        if ap_name == 'ls0':
            print(r'$\mathcal{A}_0$ ', end='& ')
        elif ap_name == 'ls10':
            print(r'$\mathcal{A}_{10}$ ', end='& ')
        elif ap_name == 'ls15':
            print(r'$\mathcal{A}_{15}$ ', end='& ')
        elif ap_name == 'ls20':
            print(r'$\mathcal{A}_{20}$ ', end='& ')
        elif ap_name == 'ls25':
            print(r'$\mathcal{A}_{25}$ ', end='& ')
        elif ap_name == 'ls50':
            print(r'$\mathcal{A}_{50}$ ', end='& ')



        file_dir = os.path.join(ap_dir, '{}/ap_config.csv'.format(ap_name))   

        with open(file_dir, 'r') as csv_file:
            csv_reader = reader(csv_file)
            alg_list = list(csv_reader)[0]

            for j, alg in enumerate(alg_list):
                if j != len(alg_list)-1:
                    print(alg,end=', ')
                    if j % 2 == 1:
                        print(r'\\ & ', end='')
                else:
                    print(alg, end=r'\\'+'\n')
                
                
            
        if i != len(ap_list)-1:
            print(r'\midrule')
        else:
            print(r'\bottomrule')
        
    print(r'\end{tabular}}\\')



if __name__ == '__main__':
    run()