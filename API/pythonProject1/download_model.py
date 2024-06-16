from transformers import BertTokenizerFast, AutoModelForTokenClassification

model_name = "bert-base-uncased"

tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

tokenizer.save_pretrained("./model")
model.save_pretrained("./model")
