import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications import VGG16, DenseNet201
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your pre-trained model
model = tf.keras.models.load_model('/home/fansan/Desktop/College_sem/projects/Image_captioning_major/captioning_models/dense_lstm/model.keras')

data = pd.read_csv("/home/fansan/Desktop/College_sem/projects/Image_captioning_major/captioning_models/captions.txt")

def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]",""))
    data['caption'] = data['caption'].apply(lambda x: x.replace("\s+"," "))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word)>1]))
    data['caption'] = "startseq "+data['caption']+" endseq"
    return data

data = text_preprocessing(data)
captions = data['caption'].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Define the image transformation
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img, axis=0)


    feature_extractor = DenseNet201(include_top=False, pooling='avg')
    feature = feature_extractor.predict(img, verbose=0)

    return feature

def idx_to_word(integer,tokenizer):
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None    

def generate_caption(image_path):
    feature = preprocess_image(image_path)
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break
    in_text = in_text.replace("startseq", ">").replace("endseq", "").strip()
    return in_text