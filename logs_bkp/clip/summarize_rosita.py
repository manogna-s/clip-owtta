import os
import pandas as pd
import glob
import json

# Path to the directory containing your CSV files
ID_datsets = ['cifar10OOD', 'cifar100OOD', 'VisdaOOD', 'ImagenetROOD']
OOD_datasets = ['MNIST', 'SVHN', 'cifar10', 'cifar100', 'Tiny']

# for id_data in ID_datsets:
#     for ood_data in OOD_datasets:
#         log_dir = f'/home/manogna/TTA/clip-owtta/logs/maple/{id_data}/{ood_data}'
#         if os.path.exists(log_dir):
#             csv_files = sorted(glob.glob(f'{log_dir}/*/result_metrics/*.csv'), reverse=True)

#             # Initialize an empty DataFrame to store the combined data
#             combined_df = []

#             # Loop through each CSV file and append its data to the combined DataFrame
#             for file in csv_files:
#                 if not ('rosita' in file or 'zsclip' in file or 'ft_pl_txt' in file): continue
#                 if not 'maxlogit' in file: continue
                
#                 df = pd.read_csv(file)
                
#                 # Append the DataFrame to the combined DataFrame
#                 combined_df.append(df)
#             combined_df = pd.concat(combined_df, ignore_index=True)
#             # combined_df.to_csv(f'{log_dir}/summary_metrics.csv', index=False)
#             print(f'\n\nID dataset:{id_data}, OOD dataset:{ood_data}')
#             print(combined_df)
            
            


for id_data in ID_datsets:
    for ood_data in OOD_datasets:
        log_dir = f'/home/manogna/TTA/clip-owtta/logs_bkp/clip/{id_data}/{ood_data}'
        if os.path.exists(log_dir):
            combined_df = []
            for method in ['zsclip', 'promptalign_aug', 'ft_pl', 'rosita_v12_pl', 'rosita_v12_simclr', 'rosita_v12', 'rosita_v10_pl', 'rosita_v10']:
                # csv_files = sorted(glob.glob(f'{log_dir}/{method}/result_metrics/*.csv'), reverse=True)

                # Initialize an empty DataFrame to store the combined data

                # Loop through each CSV file and append its data to the combined DataFrame
                # for file in csv_files:
                    # print(file)
                    # if ('rosita_v0' in file or 'zsclip' in file or 'ft_pl_txt' in file): 

                    #     if not 'maxlogit' in file: continue
                if method in ['zsclip', 'promptalign_aug']:    
                    file_path = f'{log_dir}/{method}_txt/result_metrics/maxlogit.csv'
                    # df = pd.read_csv(f'{log_dir}/{method}_txt/result_metrics/maxlogit.csv')
                else:      
                    file_path = f'{log_dir}/{method}_txt/result_metrics/maxlogit_plthresh07.csv'  
                    # df = pd.read_csv(f'{log_dir}/{method}_txt/result_metrics/maxlogit_plthresh07.csv')
                if not os.path.exists(file_path): continue
                df = pd.read_csv(file_path)
                
                # Append the DataFrame to the combined DataFrame
                combined_df.append(df)
            combined_df = pd.concat(combined_df, ignore_index=True)
            # combined_df.to_csv(f'{log_dir}/summary_metrics.csv', index=False)
            print(f'\n\nID dataset:{id_data}, OOD dataset:{ood_data}')
            print(combined_df)
