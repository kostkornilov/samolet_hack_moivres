from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import pipeline
from utils import get_initial_index
import pandas as pd

def inference_function():
    model = AutoModelForTokenClassification.from_pretrained('results/ner_model')
    tokenizer = AutoTokenizer.from_pretrained('results/tokenizer')

    preds_final = []
    preds_val = []

    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    data_test = pd.read_csv('gt_test.csv')

    pred2 = nlp([text for text in data_test.processed_text])
    for prediction, initial_string in zip(pred2, data_test.processed_text.str.split()):
        initial_indicies = preds_final.append(get_initial_index(prediction, initial_string))


    for prediction, tokens in zip(preds_final, data_test.processed_text.str.split()):
        result = ['O'] * len(tokens)
        for key, entity in prediction.items():
            result[key] = entity
        preds_val.append(result)

    data_test.loc[:, 'label'] = preds_val

    data_test.to_csv('inference.csv', index = False)

inference_function()
