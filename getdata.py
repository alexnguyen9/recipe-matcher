import pandas as pd
import itertools
import numpy as np
import pickle
import nltk
import re

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

# convert plural to singular
wnl = WordNetLemmatizer()
def singular(x):
    return [wnl.lemmatize(s) for s in x]

# get only nouns and adjectives (for some reason it kept removing 'pepper' and 'chicken')
def get_nouns_and_adj(l):
    return [m[0] for m in itertools.chain.from_iterable([nltk.pos_tag(nltk.word_tokenize(x)) for x in l]) if (m[1] in ['NN','JJ','NNS','NNP','NNPS'] or m[0] in ['pepper','chicken'])]

# change certain compound words to singular words
def replace_foods(s):
    return s.replace('brown sugar','brownsugar')\
        .replace('chili powder','chilipowder')\
        .replace('garlic powder','garlicpowder')\
        .replace('all purpose','allpurpose')\
        .replace('fish sauce','fishsauce') \
        .replace('soy sauce','soysauce') \
        .replace('sesame oil','sesameoil') \
        .replace('semi sweet','semisweet') \
        .replace('cream cheese','creamcheese') \
        .replace('olive oil','oliveoil') \
        .replace('baking soda','bakingsoda') \
        .replace('sour cream','sourcream') \
        .replace('bell pepper','bellpepper') \
        .replace('jell o','jello') \
        .replace('soy milk','soymilk') \
        .replace('bell pepper','bellpepper') \
        .replace('bellpepper','bellpepper') \
        .replace('baking power', 'bakingpowder') \
        .replace('baking soda', 'bakingsoda') \
        .replace('green bean', 'greenbean') \
        .replace('corn syrup', 'cornsyrup') \
        .replace('food coloring', 'foodcoloring') \
        .replace('chicken broth', 'chickenstock') \
        .replace('chicken stock', 'chickenstock') \
        .replace('beef stock', 'beefstock') \
        .replace('vegetable stock', 'vegetablestock') \
        .replace('vegetable broth', 'vegetablestock') \
        .replace('beef broth', 'beefstock') \
        .replace('soy milk', 'soymilk') \
        .replace('almond milk', 'almondmilk') \
        .replace('lemon juice', 'lemonjuice') \
        .replace('lime juice', 'limejuice') \
        .replace('soy milk', 'soymilk') \
        .replace('black pepper', 'pepper') \
        .replace('sour cream', 'sourcream') \
        .replace('ice cream', 'icecream') \
        .replace('green onion', 'greenonion') \
        .replace('crab meat', 'crabmeat') \
        .replace('barbeque','bbq') \
        .replace('barbecue','bbq') \
        .replace('creme','cream')   \
        .replace('cheddar','cheddar cheese')   \
        .replace('parmesan','parmesan cheese')  \
        .replace('parmigiano','parmesan cheese')  \
        .replace('feta','feta cheese')   \
        .replace('gouda','gouda cheese')   \
        .replace('spaghetti','spaghetti pasta')   \
        .replace('linguine','linguine pasta')   \
        .replace('penne','penne pasta')   \
        .replace('fettucine','fettucine pasta') \
        .replace('macaroni','macaroni pasta') \
        .replace('rigatoni','rigatoni pasta') \
        .replace('chilli','chili') \
        .replace('chile','chili') \
        .replace('chiles','chili') \
        .replace('chilies','chili') \
        .replace('white sugar', 'sugar') \
        .replace('pepper flake', 'pepperflake') \
        .replace('coconut milk', 'coconutmilk') \
        .replace('coconut oil', 'coconutoil') \
        .replace('ice cube', 'ice')

# remove meausure words, preparation adjectives
def remove_words(x):
    remove = ['quart','liter','ml','teaspoon','pt','tablespoon','cup','ounce','fluid','gallon','pint',
          'pound','slice','sheet','pound','gram','ml','stick','bulb','inch','pinch','large',
         'small','light','sprig','quarter','half','whole','handful','good','best','fresh',
         'package','can','packed','stem','medium','piece','stalk','finishing','bottle','container',
         'clove','ear','fine','quality','coarse','bunch','wedge','flat','ground','lb','c','tbs',
         'thin','wide','refridgerator','equipment','standard','b','unprocessed','en', 'round',
          'optional','tsp','warm','cold','chopped','boiling','kitchen','length','lengthwise','smallish',
          'quick','dry','wet','new','few','many','splash','drop','topping','pure','regular','oz',
          'jar','envelope','extra','generous','hard','old','little','different','low','fat','gluten','free',
          'raw','square','foil','special','store','hard','soft','frozen','bag','recipe','decadent','spiral',
          'mini','simple','cooked','dark','packet','pre','box','unsalted','firm','other','tb','thread',
          'strand','strip','thick','restaurant','accompaniment','kg','lbs','ripe','boneless','range','zesty',
          'sodium','lowfat','original','tbsp','fl','peel','available','dash','nonstick','adjustable',
          'natural','zest','preheat','head','refridgerated','such','uncooked','canned','size','skinless',
          'frying','baby','size','artisan','organic','canned','sliced','cooled','chilled','part',
          'peeled','bottled','unpeeled','crunchy','pt','litre','additional','addition','wrapped','sweetened']
    return [y for y in x if y not in remove]


# clean a list of ingredients separated with a comma
def clean_string(i):
    return replace_foods(' '.join(remove_words(singular(re.sub(r'[^a-zA-Z0-9/,/]','',i.lower()).split(',')))))




if __name__ == '__main__':

	print("Reading JSON Files...")
	data_bbc = pd.read_json("data/bbccouk-recipes.json",lines=True)
	#data_cookstr = pd.read_json("data/cookstr-recipes.json",lines=True)
	data_epi = pd.read_json("data/epicurious-recipes.json",lines=True)
	data_ar = pd.read_json("data/allrecipes-recipes.json",lines=True)

	# fix epicurious food dataframe
	data_epi.rename(columns = {'hed':'title','prepSteps':'instructions'},inplace=True)
	data_epi.url = data_epi.url.apply(lambda x: 'www.epicurious.com' + x)

	# combine all the food recipes and get only the relevant columns (title, ingredients, instructions, url)
	combined = pd.concat([data_bbc,data_epi,data_ar],join='inner',ignore_index=True)
	

	# remove this food item since I think there is a bug in the scraper
	combined = combined[combined.title != 'Johnsonville® Three Cheese Italian Style Chicken Sausage Skillet Pizza']
	combined.dropna(inplace=True)

	# get recipes with at least 3 or more ingredients
	combined = combined[combined.ingredients.apply(len) > 2]

	# turn the list into a string
	combined.instructions = combined.instructions.apply(lambda x:' '.join(x))

	# reset index
	combined.reset_index(inplace=True,drop=True)

	print("Cleaning ingredients...")
	# clean the recipes
	recipe = combined.ingredients.apply(lambda x: [re.sub(",.*$", "", y).lower() for y in x]) # remove everything after a comma and make lower case
	recipe = recipe.apply(lambda x: [re.sub('é','e', y)  for y in x]) # change accented 'e'
	recipe = recipe.apply(lambda x: [re.sub('î','i', y)  for y in x]) # change accented 'i'
	recipe = recipe.apply(lambda x: [re.sub(r'[^\x00-\x7f]',r' ', y)  for y in x]) # remove accented characters
	recipe = recipe.apply(lambda x: [re.sub(" with.*$", "", y) for y in x]) # everything after a 'with'
	recipe = recipe.apply(lambda x: [re.sub('\([^()]*\)', "", y) for y in x]) # everything in parenthesis
	recipe = recipe.apply(lambda x: [re.sub(r'\W+'," ", y) for y in x]) # only alphanumeric characters

	recipe = recipe.apply(get_nouns_and_adj) # get only nouns and adjectives
	recipe = recipe.apply(singular) # convert to singular words
	recipe = recipe.apply(remove_words) # remove irrelatvent words
	recipe = recipe.apply(lambda x: replace_foods(' '.join(x))) # turn into a string

	print("Vectorizing ingredients...")
	# define the count vectorizer
	vc = CountVectorizer(stop_words='english',min_df=60,binary=True)

	# this is the document term matrix from our recipes
	X = vc.fit_transform(recipe.values)

	print("Pickling...")
	#export as pickle files for flask app
	pickle.dump(X.toarray().astype(bool), open('food_matrix.pkl','wb')) # save as boolean array to save space
	pickle.dump(combined, open('main_data.pkl','wb'))
	pickle.dump(vc, open('transformer.pkl','wb'))
	
	