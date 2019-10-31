## Recipe Matcher

I deployed this project as a flask on pythonanywhere.com, you can check it out [here](http://recipefinder.pythonanywhere.com/)

You can also find more information on this project [here](https://alexnguyen9.github.io/project/recipematcher/)

The scraped recipe data comes from this [reddit post](https://www.reddit.com/r/datasets/comments/94awca/thousands_of_recipes_from_epicurious_bbc/) I found a while back.  Specifically the data is found [here](https://archive.org/download/recipes-en-201706/).  I used  the following  four json files for this project:
 * epicurious-recipes.json 
* bbccouk-recipes.json
* allrecipes-recipes.json
* cookstr-recipes.json 

I downloaded them from the link and unzipped them into the data folder.




This project was done on Python 3.7.  You can install necessary packages with the `requirements.txt` file.  This project involved two parts:
1. Compiling the datasets and creating a bag of words model with the recipes
2.  Deploying the app to flask

## Step 1
`getdata.py` combines the 4 json files into a pandas dataframe.  The file cleans through the ingredients of all the recipes and creates a document term matrix for the recipes using sklearn's CountVectorizer.

The file outputs the pickle files:
`transformer.pkl` is the countvectorizer that was fitted from the ingredients from the dataset
`food_matrix.pkl` is the recipe term matrix from the countvectorizer
`main_data.pkl` is the dataframe that contains the recipe title, ingredients, and instructions

## Step 2
The flask requires the 3 previous pickle files first. You can then deploy the flask app locally by running:
 `python app.py`
