# dataset name 
# dataset = 'Adressa-1w'
# dataset = 'yelp'
dataset = 'amazon_book'
assert dataset in ['ml-1m', 'amazon_book', 'yelp', 'pinterest-20', 'Adressa-1w']

# model name 
# model = 'NeuMF-end'
model = 'GMF'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
# root_path = '../data-4-valid/'
# main_path = '../data_clean_train_neg_mapped/'
# main_path = '../data-4-valid/data_coteaching/'
# main_path = '../data-4-valid/data_noisy/'
# main_path = '../data_clean_train_neg/'
# root_path = '../yelp/3/'
# main_path = '../yelp/3/data_coteaching/'
root_path = '../amazon/'
main_path = '../amazon/data_coteaching/'

train_rating = main_path + '{}.train.rating'.format(dataset)
valid_rating = main_path + '{}.valid.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)
user_neg = root_path + '{}.user_neg'.format(dataset)

model_path = './amazon_book/data_coteaching/'
# model_path = './yelp/3/data_coteaching/'
# model_path = './data_clean_train_neg_mapped_models/'
# model_path = './data-4-valid/data_coteaching_2/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
