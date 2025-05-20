# Modified version of dataset_gen_afba.py focusing on active fire detection
import argparse
import pandas as pd
import os
from satimg_dataset_processor.satimg_dataset_processor import AFBADatasetProcessor, AFTestDatasetProcessor

# Training set rois
dfs = []
for year in ['2017', '2018', '2019', '2020']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# Test set rois
dfs_test = []
for year in ['2021']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df_test = pd.read_csv(filename)
    dfs_test.append(df_test)
df_test = pd.concat(dfs_test, ignore_index=True)

val_ids = ['20568194', '20701026','20562846','20700973','24462610', '24462788', '24462753', '24103571', '21998313', '21751303', '22141596', '21999381', '22712904']

df = df.sort_values(by=['Id'])
df['Id'] = df['Id'].astype(str)
train_df = df[~df.Id.isin(val_ids)]
val_df = df[df.Id.isin(val_ids)]

train_ids = train_df['Id'].values.astype(str)
val_ids = val_df['Id'].values.astype(str)

df_test = df_test.sort_values(by=['Id'])
test_ids = df_test['Id'].values.astype(str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Active Fire Detection dataset.')
    parser.add_argument('-mode', type=str, help='Train/Val/Test')
    parser.add_argument('-ts', type=int, help='Length of TS')
    parser.add_argument('-it', type=int, help='Interval')
    parser.add_argument('-data_path', type=str, default='/path/to/data', help='Path to data directory')
    parser.add_argument('-save_path', type=str, default='dataset', help='Path to save dataset')
    args = parser.parse_args()
    
    ts_length = args.ts
    interval = args.it
    modes = args.mode
    data_path = args.data_path
    save_path = args.save_path
    
    # Active Fire Detection only
    usecase = 'af'
    
    if modes == 'train':
        locations = train_ids
    elif modes == 'val':
        locations = val_ids
    else:
        # Test locations for active fire detection
        locations = ['elephant_hill_fire', 'eagle_bluff_fire', 'double_creek_fire','sparks_lake_fire', 'lytton_fire', 
                    'chuckegg_creek_fire', 'swedish_fire', 'sydney_fire', 'thomas_fire', 'tubbs_fire', 
                    'carr_fire', 'camp_fire', 'creek_fire', 'blue_ridge_fire', 'dixie_fire', 'mosquito_fire', 'calfcanyon_fire']
    
    satimg_processor = AFBADatasetProcessor()
    if modes == 'train' or modes == 'val':
        # Generate dataset for training or validation
        satimg_processor.dataset_generator_seqtoseq(
            mode=modes, 
            usecase=usecase, 
            data_path=data_path, 
            locations=locations, 
            visualize=False, 
            file_name=usecase+'_'+modes+'_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy',
            label_name=usecase+'_'+modes+'_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy',
            save_path=os.path.join(save_path, 'dataset_'+modes), 
            ts_length=ts_length, 
            interval=interval, 
            image_size=(256, 256)
        )
    else:  
        # Generate dataset for testing
        for id in locations:
            af_test_processor = AFTestDatasetProcessor()
            af_test_processor.af_test_dataset_generator(
                id, 
                save_path=os.path.join(save_path, 'dataset_test'), 
                file_name='af_' + id + '_img.npy'
            )
            af_test_processor.af_seq_tokenizing_and_test_slicing(
                location=id, 
                modes=modes, 
                ts_length=ts_length, 
                interval=interval, 
                usecase=usecase, 
                root_path=save_path, 
                save_path=save_path
            )
    
    print(f"Dataset generation complete for {modes} mode with ts_length={ts_length} and interval={interval}")