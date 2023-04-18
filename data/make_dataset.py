import os
import argparse
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, required=True, metavar="FILE", help='Data directory path')
parser.add_argument('--out_path', type=str, required=True, help='Save csv file path')
parser.add_argument('--train', type=float, default=0.7, help='Train Dataset ratio')
parser.add_argument('--valid', type=float, default=0.2, help='Valid Dataset ratio')
parser.add_argument('--test', type=float, default=0.1, help='Test Dataset ratio')

def main():
    args = parser.parse_args()
    image_folders = os.listdir(os.dir_path, 'image')
    label_folders = os.listdir(os.dir_path, 'label')
    image_lists = [os.path.join(os.dir_path, 'image', folder, name) for name in os.listdir(os.path.join(os.dir_path, 'image', folder)) for folder in image_folders]
    label_lists = [os.path.join(os.dir_path, 'label', folder, name) for name in os.listdir(os.path.join(os.dir_path, 'image', folder)) for folder in image_folders]
    image_lists = np.array(image_lists)
    label_lists = np.array(label_lists)

    file_ranges = np.arange(0, (len(image_lists)))
    train = (np.random.choice(len(file_ranges), int(len(file_ranges)*args.train)))
    valid = (np.random.choice(list(set(file_ranges)-set(train)), int(len(file_ranges)*args.valid)))
    test = (np.random.choice(list(set(file_ranges)-set(train)-set(valid)), int(len(file_ranges)*args.test)))

    train_data = {'image': image_lists[train], 'label': label_lists[train]}
    valid_data = {'image': image_lists[valid], 'label': label_lists[valid]}
    test_data = {'image': image_lists[test], 'label': label_lists[test]}
    df = pd.DataFrame.from_dict(train_data)
    df.to_csv(os.path.join(args.out_path, 'Train_data.csv'))
    df = pd.DataFrame.from_dict(test_data)
    df.to_csv(os.path.join(args.out_path, 'Valid_data.csv'))
    df = pd.DataFrame.from_dict(valid_data)
    df.to_csv(os.path.join(args.out_path, 'Test_data.csv'))
    
if __name__ == '__main__':
    main()