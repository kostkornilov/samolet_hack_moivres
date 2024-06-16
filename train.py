import datasets
import numpy as np
from sklearn.metrics import f1_score
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import pipeline
from sklearn.model_selection import train_test_split
from utils import tokenize_and_align_labels, compute_metrics, get_initial_index
import pandas as pd
import ast
import evaluate
import json

def train_function():

    dataset_path = 'train_data.csv'
    model_name = 'sergeyzh/rubert-mini-sts'
    class_weights = {'O': 0.003, 'B-discount': 1, 'B-value': 2, 'I-value': 2}
    tags = []
    preds_final = []
    preds_val = []
    score = []

    df = pd.read_csv(dataset_path)
    df = df[df.target_labels_positions != '{}']

    tokens = df.processed_text.str.split(' ').reset_index().processed_text
    dicts = df.target_labels_positions.apply(ast.literal_eval).reset_index().target_labels_positions

    text_lengths = tokens.apply(len)

    for length, labels in zip(text_lengths, dicts):
        index_label = [(key, pos) for key, positions in labels.items() for pos in positions]
        result = ['O'] * length
        for i in index_label:
            result[i[1]] = i[0]
        tags.append(result)

    final_set = pd.DataFrame({'tokens': tokens.values, 'tags': tags})
    label_list = ['O', 'I-value', 'B-value', 'B-discount']

    train, val = train_test_split(final_set, test_size=0.1, random_state=42)

    dataset_hf = datasets.DatasetDict(
        {'train': datasets.Dataset.from_pandas(train),
        'val': datasets.Dataset.from_pandas(val)}
    ).remove_columns(["__index_level_0__"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_dataset = dataset_hf.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label_list": label_list})

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

    args = TrainingArguments('test-ner',
                            eval_strategy='epoch',
                            learning_rate=0.00002,
                            per_device_train_batch_size=16,
                            per_device_eval_batch_size=16,
                            num_train_epochs=20,
                            weight_decay=0.01,
                            logging_steps=15,
                            report_to='none')

    data_collator = DataCollatorForTokenClassification(tokenizer) # forms a batch

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['val'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda pred: compute_metrics(pred, label_list=label_list)
    )

    trainer.train()

    model.save_pretrained('results/ner_model')
    tokenizer.save_pretrained('results/tokenizer')


    id2label = {
        str(i): label for i, label in enumerate(label_list)
    }

    label2id = {
        label: str(i) for i, label in enumerate(label_list)
    }

    config = json.load(open('results/ner_model/config.json'))

    config['id2label'] = id2label
    config['label2id'] = label2id

    json.dump(config, open('results/ner_model/config.json', 'w'))

    model_finetuned = AutoModelForTokenClassification.from_pretrained('results/ner_model')
    tokenizer = AutoTokenizer.from_pretrained('results/tokenizer')

    nlp = pipeline('ner', model=model_finetuned, tokenizer=tokenizer)

    pred2 = nlp([' '.join(tokens) for tokens in dataset_hf['val']['tokens']])


    for prediction, initial_string in zip(pred2, dataset_hf['val']['tokens']):
        initial_indicies = preds_final.append(get_initial_index(prediction, initial_string))

    for prediction, tokens in zip(preds_final, dataset_hf['val']['tokens']):
        result = ['O'] * len(tokens)
        for key, entity in prediction.items():
            result[key] = entity
        preds_val.append(result)


    for i in range(len(preds_val)):
        sample_weight = [class_weights[label] for label in dataset_hf['val']['tags'][i]]
        score.append(f1_score(dataset_hf['val']['tags'][i], preds_val[i][:len(dataset_hf['val']['tags'][i])], average='weighted', sample_weight=sample_weight))
    print(np.mean(score))

train_function()