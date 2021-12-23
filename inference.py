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
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="uer/chinese_roberta_L-12_H-768", type=str)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    # parser.add_argument("--train_file", default="data/train_data.csv", type=str)
    parser.add_argument("--test_file", default="data/test_data.csv", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--total_epoch", default=2, type=int)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Device:", device)
    print("=====Loading Data=====")
    test_df = pd.read_csv(args.test_file, header=0)
    test_text = list(test_df['text'])
    test_aspect = list(test_df['aspect'])

    print("=====Prompt Preparing=====")
    classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
        "负面",
        "正面"
    ]
    test_dataset = []
    for i in range(len(test_text)):
        test_dataset.append(InputExample(guid=i,
                                         text_a=test_text[i],
                                         text_b=test_aspect[i]
                                         ))

    plm, tokenizer, model_config, WrapperClass = load_plm("bert", args.model_name_or_path)
    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"}，{"placeholder":"text_b"}是好评吗？{"mask"}',
        tokenizer=tokenizer,
    )
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "负面": ["否", "错", "不"],
            "正面": ["是", "对"],
        },
        tokenizer=tokenizer,
    )
    promptModel = torch.load('pl-absa.pt').to(device)
    test_data_loader = PromptDataLoader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.batch_size,
        shuffle=False
    )
    promptModel.eval()
    with torch.no_grad():
        allpreds = []
        alllabels = []
        for step, inputs in tqdm.tqdm(enumerate(test_data_loader)):
            inputs = inputs.to(device)
            logits = promptModel(inputs)
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    test_pred = [True if _ == 1 else 0 for _ in allpreds]
    out_data = list(zip(test_text, test_aspect, test_pred))
    out_df = pd.DataFrame(out_data, columns=['text', 'aspect', 'prediction'])
    out_df.to_csv("predictions.csv")
    # while True:
    #     tmp_text = input("请输入text：")
    #     tmp_aspect = input("请输入aspect:")
    #     tmp_dataset = [InputExample(guid=0, text_a=tmp_aspect, text_b=tmp_text)]
    #     tmp_data_loader = PromptDataLoader(
    #         dataset=tmp_dataset,
    #         tokenizer=tokenizer,
    #         template=promptTemplate,
    #         tokenizer_wrapper_class=WrapperClass,
    #         batch_size=1,
    #         shuffle=False
    #     )
    #     promptModel.eval()
    #     with torch.no_grad():
    #         allpreds = []
    #         alllabels = []
    #         for step, inputs in tqdm.tqdm(enumerate(tmp_data_loader)):
    #             inputs = inputs.to(device)
    #             logits = promptModel(inputs)
    #             pred = torch.argmax(logits, dim=-1).cpu().tolist()
    #             if pred[0]:
    #                 print("正向")
    #             else:
    #                 print("负向")

