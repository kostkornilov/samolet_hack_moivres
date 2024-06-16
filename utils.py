from sklearn.metrics import f1_score
import numpy as np
from transformers import AutoTokenizer


def tokenize_and_align_labels(example, label_all_tokens = True, tokenizer = None, label_list = None):
    tokenized_input = tokenizer(example['tokens'], truncation=True, is_split_into_words=True)
    labels = []

    for i, label in enumerate(example['tags']):
        word_ids = tokenized_input.word_ids(batch_index=i) # returns a list indicating the word corresponding to each token
        previous_word_idx = None

        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        label_ids = [label_list.index(idx) if isinstance(idx, str) else idx for idx in label_ids]
        labels.append(label_ids)
    tokenized_input['labels'] = labels
    return tokenized_input


def compute_metrics(eval_preds, label_list):
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)

    predictions = [
        [label_list[prediction] for (prediction, label) in zip(pred, true_label) if label != -100]
          for pred, true_label in zip(pred_logits, labels)
    ]

    true_labels = [
        [label_list[label] for (prediction, label) in zip(pred, true_label) if label != -100]
          for pred, true_label in zip(pred_logits, labels)
    ]

    class_weights = {'O': 0.003, 'B-discount': 1, 'B-value': 2, 'I-value': 2}
    sample_weight = [[class_weights[label] for label in seq] for seq in true_labels]
    sample_weight = [item for sublist in sample_weight for item in sublist]
    
    predictions_flat = [item for sublist in predictions for item in sublist]
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    
    results = f1_score(true_labels_flat, predictions_flat, average='weighted', sample_weight=sample_weight)
    
    return {
        'f1_weighted': results
    }


def find_word_index_by_char_range(s, start_idx, end_idx):
    words = s.split()
    
    current_char_pos = 0
    
    word_indices = []
    
    for i, word in enumerate(words):
        word_start_pos = current_char_pos
        word_end_pos = current_char_pos + len(word) - 1
        
        if word_start_pos <= end_idx and word_end_pos >= start_idx:
            word_indices.append(i)
            
        current_char_pos += len(word) + 1
    
    return word_indices

def get_initial_index(prediction, initial_string):
    sentence = ' '.join(initial_string)
    entities = {}
    for i in prediction:
        for j in find_word_index_by_char_range(s=sentence, start_idx=i['start'], end_idx=i['end']):
            entities[j] = i['entity']
    return entities

