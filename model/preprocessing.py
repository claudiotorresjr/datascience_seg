import pandas as pd

from model import utils

class Preprocessing(object):
    """
        Base class to handle preprocessing
    """

    RAW_TRAIN_PATH = "model/dataset/train-tweets.csv"
    RAW_TEST_PATH = "model/dataset/test-tweets.csv"

    CLEAN_TRAIN_PATH = "model/dataset/train-tweets-clean.csv"
    CLEAN_TEST_PATH = "model/dataset/test-tweets-clean.csv"

    def start_process(self):
        """
            Start the preprocessing process
        """
    
        #open the dataframe using pandas
        print("Opening train dataset...")
        train_df = pd.read_csv(self.RAW_TRAIN_PATH, delimiter= ',', header = 0)
        #drop in 'Unnamed: 7' (all NaN)
        train_df = train_df.drop('Unnamed: 7', axis=1)
        #drop in 'location' (some weird locations and not standardized! ex. could be ("NY", "New York", "Upper East Side", Etc!))
        train_df = train_df.drop('location', axis=1)
        #remove south dakota label
        train_df = train_df.loc[train_df['Type'] != 'South Dakota']
        train_df.reset_index()

        print("Opening test dataset...")
        test_df = pd.read_csv(self.RAW_TEST_PATH)
        #drop in 'Id' (no need. dataframe already have one :D)
        test_df = test_df.drop('Id', axis=1)
        #drop in 'location' (some weird locations and not standardized! ex. could be ("NY", "New York", "Upper East Side", Etc!))
        test_df = test_df.drop('location', axis=1)

        #set the float print format
        pd.set_option('float_format', '{:.4f}'.format)

        print(f"\nlabels count for train dataset:")
        self._labels_count(train_df)

        #some columns have NaN values. Replace all with de column median
        print("\nNANs in train")
        print('===================')
        train_df = self._replace_nam_median(train_df)
        print("NANs in test")
        print('===================')
        test_df = self._replace_nam_median(test_df)

        print(f"\nCalculating all features from train dataset:")
        print(f"{' '.join(self.MODEL_FEATURES)}")
        train_df = utils.create_features_columns(train_df, True)

        print(f"\nCalculating all features from test dataset:")
        print(f"{' '.join(self.MODEL_FEATURES)}")
        test_df = utils.create_features_columns(test_df)

        #save datasets
        train_df.to_csv(self.CLEAN_TRAIN_PATH, sep = ',', header=True, index=False)
        test_df.to_csv(self.CLEAN_TEST_PATH, sep = ',', header=True, index=False)
    
    def _replace_nam_median(self, df):
        """
            Replace all NaN values with the column values median

            :param df: dataframe

            :return: dataframe
        """
    
        print('-------------------')
        print(df.isna().sum())
        df['following'].fillna(df['following'].median(), inplace=True)
        df['followers'].fillna(df['followers'].median(), inplace=True)
        df['actions'].fillna(df['actions'].median(), inplace=True)
        df['is_retweet'].fillna(0, inplace=True)

        return df

    def _labels_count(self, df):
        """
            Count the total of span and non-spam

            :param df: dataframe
        """

        spam_count = len(df.loc[df['Type'] == 'Spam'])
        non_spam_count = len(df.loc[df['Type'] == 'Quality'])
        total = spam_count + non_spam_count

        print("Spam count: {} ({:.2f}%)".format(spam_count, spam_count/total*100))
        print("Quality count: {} ({:.2f}%)".format(non_spam_count, spam_count/total*100))
