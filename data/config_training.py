config = {'train_data_path': ['D:/luna16/data_subset/subset0/',
                              'D:/luna16/data_subset/subset1/',
                              'D:/luna16/data_subset/subset2/',
                              'D:/luna16/data_subset/subset3/',
                              'D:/luna16/data_subset/subset4/',
                              'D:/luna16/data_subset/subset5/',
                              'D:/luna16/data_subset/subset6/',
                              'D:/luna16/data_subset/subset7/',
                              'D:/luna16/data_subset/subset8/'],
          'val_data_path': ['D:/luna16/data_subset/subset9/'],
          'test_data_path': ['D:/luna16/data_subset/subset9/'],

          'train_preprocess_result_path': 'D:/luna16/preprocess/',
          # contains numpy for the data and label, which is generated by prepare.py
          'val_preprocess_result_path': 'D:/luna16/preprocess/',
          # make sure copy all the numpy into one folder after prepare.py
          'test_preprocess_result_path': 'D:/luna16/preprocess/',

          'train_annos_path': 'D:/luna16/annotations.csv',
          'val_annos_path': 'D:/luna16/annotations.csv',
          'test_annos_path': 'D:/luna16/annotations.csv',

          'black_list': [],

          'preprocessing_backend': 'python',

          'luna_segment': 'D:/luna16/seg-lungs-LUNA16/',
          # download from https://luna16.grand-challenge.org/data/
          'preprocess_result_path': 'D:/luna16/preprocess/',
          'luna_data': 'D:/luna16/data_subset/',
          'luna_label': 'D:/luna16/annotations.csv'
          }