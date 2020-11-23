# dataset name 
# dataset = 'adressa'
# dataset = 'yelp'
dataset = 'amazon_book'
assert dataset in ['amazon_book', 'yelp', 'adressa']

# model name 
# model = 'NeuMF-end'
model = 'GMF'
assert model in ['GMF', 'NeuMF-end']

# data paths
# root_path = '../adressa/'
# main_path = '../adressa/data/'
# root_path = '../yelp/'
# main_path = '../yelp/data/'
root_path = '../amazon_book/'
main_path = '../amazon_book/data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
valid_rating = main_path + '{}.valid.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)
user_neg = root_path + '{}.user_neg'.format(dataset)

# save path
model_path = './models/amazon_book/'
# model_path = './models/yelp/'
# model_path = './models/adressa/'

GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
