import os
import pandas as pd
import glob
import json

# Path to the directory containing your CSV files
ID_datsets = ['cifar10OOD', 'cifar100OOD', 'VisdaOOD', 'ImagenetROOD']
OOD_datasets = ['MNIST', 'SVHN', 'cifar10', 'cifar100', 'Tiny']

for id_data in ID_datsets:
    for ood_data in OOD_datasets:
        log_dir = f'/home/manogna/TTA/clip-owtta/logs_bkp/clip/{id_data}/{ood_data}'
        if os.path.exists(log_dir):
            csv_files = sorted(glob.glob(f'{log_dir}/*/result_metrics/*.csv'), reverse=True)

            # Initialize an empty DataFrame to store the combined data
            combined_df = []

            # Loop through each CSV file and append its data to the combined DataFrame
            for file in csv_files:
                df = pd.read_csv(file)
                
                # Append the DataFrame to the combined DataFrame
                combined_df.append(df)
            combined_df = pd.concat(combined_df, ignore_index=True)
            combined_df.to_csv(f'{log_dir}/summary_metrics.csv', index=False)
            print(f'\n\nID dataset:{id_data}, OOD dataset:{ood_data}')
            print(combined_df)
