# coding:utf-8
import pandas as pd
import torch
import argparse
from sklearn.metrics import classification_report
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments, Trainer


class LazyNLU_Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, datas, aspects, labels, block_size=512):
        self.datas = datas
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = block_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        :returns:
            torch.Tensor of tokenized text if no errors.
            None if any errors encountered.
        """
        data = self.datas[idx]
        aspect = self.aspects[idx]
        label = self.labels[idx]
        encodings = self.tokenizer(aspect, data, return_tensors="pt", padding='max_length', truncation=True,
                                   max_length=self.max_len)

        item = {key: torch.tensor(val[0]) for key, val in encodings.items()}
        item['labels'] = torch.tensor(label)
        return item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--train_file", default="data/train.csv", type=str)
    parser.add_argument("--test_file", default="data/test.csv", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--total_epoch", default=30, type=int)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Device:", device)
    print("=====Loading Data=====")
    train_df = pd.read_csv(args.train_file, header=0)
    test_df = pd.read_csv(args.test_file, header=0)
    train_text = list(train_df['text'])
    test_text = list(test_df['text'])
    train_aspect = list(train_df['aspect'])
    test_aspect = list(test_df['aspect'])
    train_label = list(train_df['label'])
    test_label = list(test_df['label'])
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    train_dataset = LazyNLU_Dataset(tokenizer, train_text, train_aspect, train_label)
    test_dataset = LazyNLU_Dataset(tokenizer, test_text, test_aspect, test_label)

    print("=====Model Initing=====")
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path).to(device)
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=args.total_epoch,  # total number of training epochs
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=1,
        # weight_decay=args.weight_decay,  # strength of weight decay
        save_strategy="epoch",
        seed=2021,
        logging_dir="logs/",
        logging_strategy="steps",
        logging_steps=100
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        tokenizer=tokenizer,
    )

    print("=====Training=====")
    trainer.train()
    print(trainer.evaluate(test_dataset))
