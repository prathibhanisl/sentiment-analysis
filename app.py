import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor

def run():
    # Load the pre-trained model
    model = joblib.load('model.joblib')

    # Instantiate the preprocessor
    preprocessor_instance = preprocessor()

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")
    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")
    predicted_sentiment = ""
    if st.button("Predict"):
        # Convert userinput to a pandas Series for transformation
        userinput_series = pd.Series([userinput])
        
        # Preprocess the user input text
        processed_text = preprocessor_instance.transform(userinput_series).iloc[0]
        
        # Predict the sentiment
        predicted_sentiment = model.predict([processed_text])[0]
        
        if predicted_sentiment == 1:
            output = 'positive 👍'
        else:
            output = 'negative 👎'
        
        sentiment = f'Predicted sentiment of "{userinput}" is {output}.'
        st.success(sentiment)

if __name__ == "__main__":
    run()
