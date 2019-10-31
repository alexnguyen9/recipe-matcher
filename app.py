import pandas as pd
import pickle
from flask import Flask, request, render_template, url_for, session
from getdata import clean_string
import numpy as np
from scipy.spatial.distance import cdist

app = Flask(__name__)

# session key
app.secret_key = 'dljsaklqk24e21cjn!Ew@@dsa5'


food_matrix = pickle.load(open('food_matrix.pkl','rb'))
transformer = pickle.load(open('countvectorizer.pkl','rb'))
food_data = pickle.load(open('main_data.pkl','rb'))

@app.route('/')
def home():
    return render_template('webpage.html', inputstring="")

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    session["counter"] = 0
    
    string_to_clean = str(request.form.get("ingredients"))
    string_to_convert = clean_string(string_to_clean)
    output = string_to_convert
    
	
    session["inputstring"] = string_to_clean	
    y = transformer.transform([output])
	
    m = cdist(y.toarray()[0].reshape(1,-1), food_matrix, metric='cosine')
    global index 
    index = np.argsort(m[0]).tolist()
    

    
    instructions = food_data.instructions[index[session["counter"]]]
    title = food_data.title[index[session["counter"]]]
    ingredients = food_data.ingredients[index[session["counter"]]]
    source = food_data.url[index[session["counter"]]]

    return render_template('display_recipe.html', output=output,recipe_title=title,recipe_instructions=instructions,recipe_ingredients=ingredients, source = source,page=session["counter"],inputstring=session["inputstring"])


@app.route('/next',methods=['POST','GET'])
def next():
    
    if request.form["Submit"] == 'next':
        session["counter"] += 1
    else:
        session["counter"] -= 1
    
    
    
    instructions = food_data.instructions[index[session["counter"]]]
    title = food_data.title[index[session["counter"]]]
    ingredients = food_data.ingredients[index[session["counter"]]]
    source = food_data.url[index[session["counter"]]]

    return render_template('display_recipe.html', recipe_title=title,recipe_instructions=instructions,recipe_ingredients=ingredients, source = source, page=session["counter"],inputstring=session["inputstring"])


if __name__ == "__main__":
    app.run(debug=True)