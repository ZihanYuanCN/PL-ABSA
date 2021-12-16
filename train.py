# coding:utf-8
import pandas as pd
from openprompt.data_utils import InputExample
import torch
import argparse
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptForClassification
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptDataLoader
from sklearn.metrics import classification_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--train_file", default="data/train.csv", type=str)
    parser.add_argument("--test_file", default="data/test.csv", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
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

    print("=====Prompt Preparing=====")
    classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
        "负面",
        "正面"
    ]
    train_dataset = []
    for i in range(len(train_text)):
        train_dataset.append(InputExample(guid=i,
                                          text_a=train_text[i],
                                          text_b=train_aspect[i],
                                          label=int(train_label[i])))
    test_dataset = []
    for i in range(len(test_text)):
        test_dataset.append(InputExample(guid=i,
                                         text_a=test_text[i],
                                         text_b=test_aspect[i],
                                         label=int(test_label[i])))

    plm, tokenizer, model_config, WrapperClass = load_plm("bert", args.model_name_or_path)
    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"}，{"placeholder":"text_b"}是好？{"mask"}',
        tokenizer=tokenizer,
    )
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "负面": ["否", "错"],
            "正面": ["是", "对"],
        },
        tokenizer=tokenizer,
    )
    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    ).to(device)
    train_data_loader = PromptDataLoader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.batch_size
    )

    test_data_loader = PromptDataLoader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.batch_size
    )
    print("=====Model Initing=====")
    from transformers import AdamW, get_linear_schedule_with_warmup
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    import tqdm
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    print("=====Training=====")
    for epoch in range(args.total_epoch):
        tot_loss = 0
        promptModel.train()
        optimizer.zero_grad()
        for step, inputs in tqdm.tqdm(enumerate(train_data_loader)):
            inputs = inputs.to(device)
            logits = promptModel(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)

        # making zero-shot inference using pretrained MLM with prompt
        promptModel.eval()
        with torch.no_grad():
            allpreds = []
            alllabels = []
            for step, inputs in tqdm.tqdm(enumerate(test_data_loader)):
                inputs = inputs.to(device)
                logits = promptModel(inputs)
                labels = inputs['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
        print(acc)
        print(classification_report(alllabels, allpreds, digits=4))
    # predictions would be 1, 0 for classes 'positive', 'negative'
