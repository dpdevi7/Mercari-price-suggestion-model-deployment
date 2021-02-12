from flask import Flask, jsonify, request
from flask import redirect, render_template, url_for
import itertools
import regex as re
import unidecode
import subprocess
import os
import numpy as np
import pickle
import flask
import tensorflow
from tensorflow import keras
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.layers import Embedding, Input, GRU, LSTM, Dense, BatchNormalization, Dropout, RNN, Flatten, GlobalAveragePooling1D, concatenate, PReLU, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model

# https://www.tutorialspoint.com/flask

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['FILE_UPLOAD'] = "upload" ## /home/d/Desktop/Case Study 1/Deployment/

def myModel():

    tensorflow.keras.backend.clear_session()

    embd_dim = 10


    input_1 = keras.layers.Input(shape=(13, ), name="name_input")
    input_2     = keras.layers.Input(shape=(160, ), name="desc_input")
    input_3     = keras.layers.Input(shape=(1, ), name="brand_input")
    input_4     = keras.layers.Input(shape=(3, ), name="cat_input")
    input_5     = keras.layers.Input(shape=(1, ), name="cat1_input")
    input_6      = keras.layers.Input(shape=(1,), name="cat2_input")
    input_7    = keras.layers.Input(shape=(1,), name="cat3_input")
    input_8    = keras.layers.Input(shape=(1,), name="ship_input")
    input_9    = keras.layers.Input(shape=(1,), name="cond_input")


    input__layers = [
        
        ('name_embd', len(name_vocab), 13, input_1),
        
        ('desc_embd',len(desc_vocab), 160, input_2),

        ('brand_embd', len(brand_vocab), 1, input_3),

        ('cat_embd', len(cat_vocab), 3, input_4),
        
        ('cat1_embd', len(cat1_vocab), 1, input_5),

        ('cat2_embd', len(cat2_vocab), 1, input_6),
        
        ('cat3_embd', len(cat3_vocab), 1, input_7),
        
        ('ship_embd', 3, 1, input_8),

        ('cond_embd', 6, 1, input_9),
        
    ]

    inputs = []
    flatten_layers = []

    for col in input__layers:
        name = col[0]
        input_dim = col[1]
        output_dim = 10
        input_len = col[2]
        input_layer = col[3]
        
        inputs.append(input_layer)
        embd = keras.layers.Embedding(input_dim = input_dim,
                                    output_dim = output_dim,
                                    input_length = input_len, name=name) (input_layer)
        
        
        if input_len > 1:
            flatten = keras.layers.GlobalAveragePooling1D()(embd)
        else:
            flatten = keras.layers.Flatten()(embd)
            
        flatten_layers.append(flatten)
        
        
    fm_layers = []
    for emb1, emb2 in itertools.combinations(flatten_layers, 2):
        dot_layer = keras.layers.Multiply()([emb1, emb2])
        fm_layers.append(dot_layer)

    out = keras.layers.Concatenate() (fm_layers)
    # out_2 = keras.layers.Concatenate() ([input_10, input_11, input_12, input_13, input_14,])

    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Dense(32, kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(out)
    out = PReLU()(out)

    out = keras.layers.Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(out)
    out = PReLU()(out)


    out = keras.layers.Dense(16, kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(out)
    out = PReLU()(out)

    out = keras.layers.Dense(1)(out)

    inputs = [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, ]

    model = Model(inputs, out)

    return model

def load_binary(tknz_path):

    with open(tknz_path + 'setofBrands.pickle', 'rb') as handle:
        setofBrands = pickle.load(handle)

    with open(tknz_path + 'setofCat1.pickle', 'rb') as handle:
        setofCat1 = pickle.load(handle)

    with open(tknz_path + 'setofCat2.pickle', 'rb') as handle:
        setofCat2 = pickle.load(handle)

    with open(tknz_path + 'setofCat3.pickle', 'rb') as handle:
        setofCat3 = pickle.load(handle)

    with open(tknz_path + 'tokenizer_brand_vocab.pickle', 'rb') as handle:
        brand_vocab = pickle.load(handle)

    with open(tknz_path + 'tokenizer_name_vocab.pickle', 'rb') as handle:
        name_vocab = pickle.load(handle)

    with open(tknz_path + 'tokenizer_desc_vocab.pickle', 'rb') as handle:
        desc_vocab = pickle.load(handle)

    with open(tknz_path + 'tokenizer_cat_vocab.pickle', 'rb') as handle:
        cat_vocab = pickle.load(handle)

    with open(tknz_path + 'tokenizer_cat1_vocab.pickle', 'rb') as handle:
        cat1_vocab = pickle.load(handle)

    with open(tknz_path + 'tokenizer_cat2_vocab.pickle', 'rb') as handle:
        cat2_vocab = pickle.load(handle)

    with open(tknz_path + 'tokenizer_cat3_vocab.pickle', 'rb') as handle:
        cat3_vocab = pickle.load(handle)


    return name_vocab, brand_vocab, desc_vocab, cat_vocab, cat1_vocab, cat2_vocab, cat3_vocab, setofBrands, setofCat1, setofCat2, setofCat3

def impute_brand(name, brand, desc):
    brand_name = brand.lower()
    if brand.lower() == "unknown":
        for i in setofBrands:
            if i in name.lower():
                brand_name = i
                break
            elif i in desc.lower():
                brand_name = i
                break
    return brand_name

def impute_cat1(name, cat1, desc):
    cat1 = cat1.lower()
    if cat1.lower() == "unknown":
        for i in setofCat1:
            if i in name.lower():
                cat1 = i
                break
            elif i in desc.lower():
                cat1 = i
                break
    return cat1

def impute_cat2(name, cat2, desc):
    cat2 = cat2.lower()
    if cat2.lower() == "unknown":
        for i in setofCat2:
            if i in name.lower():
                cat2 = i
                break
            elif i in desc.lower():
                cat2 = i
                break
    return cat2

def impute_cat3(name, cat3, desc):
    cat3 = cat3.lower()
    if cat3.lower() == "unknown":
        for i in setofCat3:
            if i in name.lower():
                cat3 = i
                break
            elif i in desc.lower():
                cat3 = i
                break
    return cat3

def handle_missing(details):
    ## list of products
    list_of_products = []

    ## converting products into list of json objects
    for line in details:
        print(line)
        id = line.split(",")[0]
        name = line.split(",")[1]
        brand = line.split(",")[2]
        cat = line.split(",")[3]
        cond = line.split(",")[4]
        ship = line.split(",")[5]
        desc = line.split(",")[6]


        # MISSING VALUES HANDLING

        if cat == "unknown":
            cat = "unknown/unknown/unknown"

        cat1 = cat.lower().split("/")[0]
        cat2 = cat.lower().split("/")[1]
        cat3 = cat.lower().split("/")[2]

        ## impute missing brand names
        brand = impute_brand(name, brand, desc)

        ## impute missing cat1
        cat1 = impute_cat1(name, cat1, desc)

        ## impute missing cat2
        cat2 = impute_cat2(name, cat2, desc)
        
        ## impute missing cat3
        cat3 = impute_cat3(name, cat3, desc)

        ## concatenating
        cat = cat1 + "/" + cat2 + "/" + cat3

        ## fill NaN value of desc with "no desription given."
        if desc == "unknown":
            desc = "no description given"

        product_Object = {

            "id":id,
            "name":name,
            "brand":brand,
            "cat":cat,
            "cond":cond,
            "ship":ship,
            "desc":desc,
            "cat1":cat1,
            "cat2":cat2,
            "cat3":cat3,
            "price":0
        }

        list_of_products.append(product_Object)

    return list_of_products

def decontracted(phrase):
    
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    return phrase

def remove_line(sent):
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    return sent

def accented_to_english(text):
    return unidecode.unidecode(text)

def feature_cleaning(list_of_products):

    for product in list_of_products:
        
        name = product['name'].lower()
        name = decontracted(name)
        name = remove_line(name)
        name = accented_to_english(name)
        name = re.sub(r'[^A-Za-z0-9 ]', r' ', name)
        name = name.strip()
        name = re.sub(' +', ' ', name)
        product['name'] = name


        brand = product['brand'].lower()
        brand = re.sub(r'[^A-Za-z0-9 ]', r' ', brand)
        brand = brand.strip()
        brand = re.sub(' +', ' ', brand)
        brand = re.sub(" ", "_", brand)
        product['brand'] = brand

        desc = product['desc'].lower()
        desc = decontracted(desc)
        desc = remove_line(desc)
        desc = accented_to_english(desc)
        desc = re.sub(r'[^A-Za-z0-9 ]', r' ', desc)
        desc = desc.strip()
        desc = re.sub(' +', ' ', desc)
        product['desc'] = desc


        cat1 = product['cat1'].lower()
        cat1 = accented_to_english(cat1)
        cat1 = re.sub(r'[^A-Za-z0-9 ]', r' ', cat1)
        cat1 = cat1.strip()
        cat1 = re.sub(' +', ' ', cat1)
        cat1 = re.sub(" ", "_", cat1)
        product['cat1'] = cat1

        cat2 = product['cat2'].lower()
        cat2 = accented_to_english(cat2)
        cat2 = re.sub(r'[^A-Za-z0-9 ]', r' ', cat2)
        cat2 = cat2.strip()
        cat2 = re.sub(' +', ' ', cat2)
        cat2 = re.sub(" ", "_", cat2)
        product['cat2'] = cat2

        cat3 = product['cat3'].lower()
        cat3 = accented_to_english(cat3)
        cat3 = re.sub(r'[^A-Za-z0-9 ]', r' ', cat3)
        cat3 = cat3.strip()
        cat3 = re.sub(' +', ' ', cat3)
        cat3 = re.sub(" ", "_", cat3)
        product['cat3'] = cat3

        product['cat'] = product['cat1'] + "_" + product['cat2'] + "_" + product['cat3']

        product['cond'] = int(product['cond'])
        product['ship'] = int(product['ship'])
    
    return list_of_products

class Text_2_Seq:
    def __init__(self, vocab, seq, max_len):
        self.vocab = vocab
        self.seq = seq
        self.max_len = max_len

    def fit_transform(self):
        tokens = self.seq.split(" ")
        padded_seq = []

        for token in tokens:
            if token in self.vocab:
                padded_seq.append(self.vocab[token])
            else:
                padded_seq.append(self.vocab['<UNK>'])

        if len(padded_seq) < self.max_len:

            while len(padded_seq) != self.max_len:
                padded_seq.append(0)

        else:
            padded_seq = padded_seq[:self.max_len]

        return np.array(padded_seq).reshape(1, -1)

def vectorize_and_predict(list_of_products):

    for product in list_of_products:

        name = product['name']
        tknz = Text_2_Seq(name_vocab, name, 13)
        name_padded = tknz.fit_transform()

        brand = product['brand']
        tknz = Text_2_Seq(brand_vocab, brand, 1)
        brand_padded = tknz.fit_transform()

        desc = product['desc']
        tknz = Text_2_Seq(desc_vocab, desc, 160)
        desc_padded = tknz.fit_transform()


        cat = product['cat']
        tknz = Text_2_Seq(cat_vocab, cat, 3)
        cat_padded = tknz.fit_transform()

        cat1 = product['cat1']
        tknz = Text_2_Seq(cat1_vocab, cat1, 1)
        cat1_padded = tknz.fit_transform()

        cat2 = product['cat2']
        tknz = Text_2_Seq(cat2_vocab, cat2, 1)
        cat2_padded = tknz.fit_transform()

        cat3 = product['cat3']
        tknz = Text_2_Seq(cat3_vocab, cat3, 1)
        cat3_padded = tknz.fit_transform()

        cond_padded = np.array(product['cond']).reshape(1, -1)

        ship_padded = np.array(product['ship']).reshape(1, -1)


        X_test = [name_padded, desc_padded,
                brand_padded,
                cat_padded, cat1_padded, cat2_padded, cat3_padded,
                ship_padded, cond_padded
        ]

        print(model.predict(X_test)[0][0])
        price_pred = np.exp(model.predict(X_test)[0][0]) - 1

        product['price'] = price_pred

        

    return list_of_products

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/upload')
def upload():
    return flask.render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():

    global name_vocab
    global brand_vocab
    global desc_vocab
    global cat_vocab
    global cat1_vocab
    global cat2_vocab
    global cat3_vocab
    global setofBrands
    global setofCat1
    global seofCat2
    global setofCat3
    global model
    global response_object

    name_vocab, brand_vocab, desc_vocab, cat_vocab, cat1_vocab, cat2_vocab, cat3_vocab, setofBrands, setofCat1, seofCat2, setofCat3 = load_binary("vectorizers/")

    model = keras.models.load_model("vectorizers/fm.h5")


    #model.load_weight("fm.h5")

    ## reading file
    file = request.files['filename']

    if str(file.filename).split(".")[1] != "txt":
        response_object = {

            "status":False,
            "statusCode":200,
            "result":"pleae upload a text file."

        }
        return jsonify(response_object)

    ## saving fileStorage object to a dir
    file.save("upload/"+file.filename)

    # reading File
    filepath = "upload/"+file.filename
    with open(filepath) as  f:
        details = f.readlines()

    list_of_products = handle_missing(details)
    list_of_products = feature_cleaning(list_of_products)

    predictions = vectorize_and_predict(list_of_products)

    response_object = {
        "status" : True,
        "statusCode" : 200,
        "result": predictions
    }

    return jsonify(response_object)

@app.route('/predict_single', methods=['POST'])
def predict_single():

    global name_vocab
    global brand_vocab
    global desc_vocab
    global cat_vocab
    global cat1_vocab
    global cat2_vocab
    global cat3_vocab
    global setofBrands
    global setofCat1
    global seofCat2
    global setofCat3
    global model
    global response_object

    name_vocab, brand_vocab, desc_vocab, cat_vocab, cat1_vocab, cat2_vocab, cat3_vocab, setofBrands, setofCat1, seofCat2, setofCat3 = load_binary("vectorizers/")

    model = keras.models.load_model("vectorizers/fm.h5")


    #model.load_weight("fm.h5")

    ## reading file
    id = request.form.get("prodid", False)
    name = request.form.get('name', False)
    brand = request.form.get('brand', False)
    cat_name = request.form.get('category', False)
    desc = request.form.get('desc', False)
    ship = request.form.get('ship', False)
    cond = request.form.get('cond', False)


    print(name)
    print(brand)
    print(cat_name)
    print(desc)
    print(ship)
    print(cond)

    details = str(id) + "," + name + "," + brand + "," + cat_name + "," + str(cond) + "," + str(ship) + "," + str(desc) + "\n"


    list_of_products = handle_missing([details])
    list_of_products = feature_cleaning(list_of_products)

    predictions = vectorize_and_predict(list_of_products)

    print(predictions)
    
    return flask.render_template("result.html", price=np.round(predictions[0]['price'], 2))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
    