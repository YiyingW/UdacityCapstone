import pandas as pd 
from datetime import datetime 
import numpy as np
from dateutil.parser import parse
import pickle
from sklearn import preprocessing



def convert_PurchDate(dataset):
	'''
	Input: a dataframe has an attribute called PurchDate
	Output: convert the date into two columns: month and day. drop PurchDate column.
	'''
	PurchDate_list = list(dataset.PurchDate)
	PurchDate_datetime_list = [parse(time) for time in PurchDate_list]
	month_list = [date.month for date in PurchDate_datetime_list]
	day_list = [date.day for date in PurchDate_datetime_list]
	dataset['month'] = month_list
	dataset['day'] = day_list
	dataset.drop(['PurchDate'], inplace=True, axis=1)
	return dataset
def unmodified_categ_count_freq(column_list, dataset):
	'''
	This function will create a dictionary of name_count_frequency table for select columns based on train dataset.
	NaN values will be converted to a category called 'NOT AVAIL'
	'''
	trainset = dataset[dataset['tst']==0].copy(deep=True)
	testset = dataset[dataset['tst']==1].copy(deep=True)

	name_count_freq = {}
	for column_name in column_list:
		name = column_name+'_count_freq'
		trainset[column_name].fillna('NOT AVAIL', inplace=True)
		testset[column_name].fillna('NOT AVAIL', inplace=True)
		if column_name == 'Transmission':
			trainset[column_name].replace('Manual', 'MANUAL', inplace=True)
			testset[column_name].replace('Manual', 'MANUAL', inplace=True)
		elif column_name == 'Make':
			trainset[column_name].replace('TOYOTA SCION', 'SCION', inplace=True)
			testset[column_name].replace('TOYOTA SCION', 'SCION', inplace=True)

		count = trainset.ix[:, ['IsBadBuy', column_name]].groupby(column_name).count()
		freq = trainset.ix[:, ['IsBadBuy', column_name]].groupby(column_name).mean()
		count.rename(columns={'IsBadBuy': 'count'}, inplace=True)
		freq.rename(columns={'IsBadBuy': 'freq'}, inplace=True)
		merged = pd.merge(count, freq, left_index=True, right_index=True)
		merged.reset_index(level=0, inplace=True)
		name_count_freq[name] = merged

	only_OH = ['Nationality', 'IsOnlineSale']
	for column_name in only_OH:
		trainset[column_name].fillna('NOT AVAIL', inplace=True)
		testset[column_name].fillna('NOT AVAIL', inplace=True)

	dataset = pd.concat([trainset, testset])

	return (name_count_freq, dataset)

def create_modified_freq_tables(name_count_freq):
	'''
	This function is to change the frequencies for the categories that have very low count. 
	'''
	current = name_count_freq['Make_count_freq']
	current.loc[current['count'] <= 629, 'freq'] = 0.14
	current.drop(['count'], axis=1, inplace=True )
	name_count_freq['Make_count_freq'] = current
	
	current = name_count_freq['Model_count_freq']
	current.loc[current['count'] <= 8, 'freq'] = 0.12
	current.drop(['count'], axis=1, inplace=True )
	name_count_freq['Model_count_freq'] = current


	current = name_count_freq['Trim_count_freq']
	current.loc[current['count'] <= 36, 'freq'] = 0.15
	current.drop(['count'], axis=1, inplace=True )
	name_count_freq['Trim_count_freq'] = current


	current = name_count_freq['SubModel_count_freq']
	current.loc[current['count'] <= 7, 'freq'] = 0.1
	current.drop(['count'], axis=1, inplace=True )
	name_count_freq['SubModel_count_freq'] = current


	current = name_count_freq['VNZIP1_count_freq']
	current.loc[current['count'] <= 176, 'freq'] = 0.11
	current.drop(['count'], axis=1, inplace=True )
	name_count_freq['VNZIP_count_freq'] = current
	
	other = ['Auction','Color', 'WheelType','Transmission','Size', 'TopThreeAmericanName', 'PRIMEUNIT', \
							'AUCGUART','VNST', 'month', 'day', 'VehicleAge']

	for unchange in other:
		name = unchange + '_count_freq'
		name_count_freq[name].drop(['count'], axis=1, inplace=True)

	modified_freq = name_count_freq
	return modified_freq

def merge_freq(column_list, modified_freq, dataset):
	'''
	This functinon is to merge the created frequency column to the dataset as additional attributes. 
	'''
	for column_name in column_list:
		name = column_name + '_count_freq'
		freq_column = modified_freq[name]
		dataset = dataset.merge(freq_column, on=column_name, how='left')

	return dataset

def make_dummies(column_list, dataset):
	'''
	This function is to make dummy variables for the columns in column_list.
	'''
	for column_name in column_list:
		dummies = pd.get_dummies(dataset[column_name], prefix=column_name)
		dataset = dataset.join(dummies)

	return dataset




# Load in test dataset 
test_raw = pd.read_csv('data/test.csv')
test_raw['IsBadBuy'] = -1
test_raw['tst'] = 1

# Load in train dataset 
train_raw = pd.read_csv('data/training.csv')
train_raw['tst'] = 0  

# concatenate train_raw with test_raw
dataset = pd.concat([train_raw, test_raw])

to_drop = ['VehYear', 'RefId', 'WheelTypeID', 'BYRNO'] 

dataset = dataset.drop(to_drop, axis=1)



dataset = convert_PurchDate(dataset)
to_freq = ['Auction', 'Make', 'Model', 'SubModel','Color', 'WheelType', 'Trim','Transmission','Size', 'TopThreeAmericanName', 'PRIMEUNIT', \
							'AUCGUART', 'VNZIP1', 'VNST', 'month', 'day', 'VehicleAge']



# pickle_out = open('temp.pickle', 'wb')
# pickle.dump(dataset, pickle_out)
# pickle_out.close()

# pickle_in = open('temp.pickle', 'rb')
# dataset = pickle.load(pickle_in)

name_count_freq = unmodified_categ_count_freq(to_freq, dataset)[0]
dataset = unmodified_categ_count_freq(to_freq, dataset)[1]

modified_freq = create_modified_freq_tables(name_count_freq)




dataset = merge_freq(to_freq, modified_freq, dataset)



to_one_hot = ['Nationality', 'Make', 'Color', 'Transmission', 'VehicleAge', 'WheelType', 'Size', 'TopThreeAmericanName', 'PRIMEUNIT',\
				'AUCGUART', 'VNST', 'IsOnlineSale', 'month', 'day', 'Auction']


dataset = make_dummies(to_one_hot, dataset)


to_drop = ['Nationality', 'Make', 'Model','Trim','SubModel','Color', 'Transmission', 'VehicleAge', 'WheelType', 'Size', 'TopThreeAmericanName', 'PRIMEUNIT',\
				'AUCGUART', 'VNZIP1', 'VNST', 'IsOnlineSale', 'month', 'day', 'Auction']

dataset.drop(to_drop, axis=1, inplace=True)

# drop the four CleanPrice columns 
clean_price_to_drop = ['MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailCleanPrice']

dataset.drop(clean_price_to_drop, axis=1, inplace=True)

numeric_cols = ['MMRAcquisitionAuctionAveragePrice',  'MMRAcquisitionRetailAveragePrice', \
		'MMRCurrentAuctionAveragePrice', 'MMRCurrentRetailAveragePrice', 'VehBCost',\
		'WarrantyCost', 'VehOdo']

dataset.fillna(-99999, inplace=True)


dataset_num = dataset.ix[:, numeric_cols].copy(deep=True)
num = np.array(dataset_num)
num = preprocessing.scale(num)

dataset[numeric_cols] = num


pickle_out = open('preprocessed_dataset.pickle', 'wb')
pickle.dump(dataset, pickle_out)
pickle_out.close()









