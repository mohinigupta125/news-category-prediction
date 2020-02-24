def input_processing(text):
    tokenized_data = [nltk.word_tokenize(i) for i in text]
    phraser_model.fit(tokenized_data)
    lda_input_data = phraser_model.transform(tokenized_data)
    return lda_input_data
    
    
