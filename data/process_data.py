"""
将数据处理为更简单的分类
"""
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="train.tsv", type=str)
    parser.add_argument("--test_file", default="test.tsv", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--total_epoch", default=5, type=int)

    args = parser.parse_args()
    train_df = pd.read_csv(args.train_file, header=0, sep='\t')
    test_df = pd.read_csv(args.test_file, header=0, sep='\t')
    print(train_df)
    aspect_set = sorted(list(set([_.split("#")[0] for _ in list(train_df['text_a'])])))
    print(aspect_set)
    aspect_CN_dict = {"battery": "电池",
                      "cpu": "处理器",
                      "display": "显示屏",
                      "hard_disc": "硬盘",
                      "hardware": "硬件",
                      "keyboard": "键盘",
                      "memory": "内存",
                      "multimedia_devices": "多媒体设备",
                      "os": "操作系统",
                      "phone": "手机",
                      "ports": "接口",
                      "power_supply": "充电器",
                      "software": "软件",
                      "support": "支持",
                      "warranty": "保修",
                      }
    print(len(aspect_set))
    train_text = list(train_df['text_b'])
    train_aspect = list(train_df['text_a'])
    train_label = list(train_df['label'])
    test_text = list(test_df['text_b'])
    test_aspect = list(test_df['text_a'])
    test_label = list(test_df['qid'])
    train_aspect = [aspect_CN_dict[_.split('#')[0]] for _ in train_aspect]
    test_aspect = [aspect_CN_dict[_.split('#')[0]] for _ in test_aspect]
    vis_train = dict()
    vis_test = dict()
    out_train_text = list()
    out_train_aspect = list()
    out_train_label = list()
    for i in range(len(train_aspect)):
        if str(train_text[i] + train_aspect[i]) in vis_train.keys():
            continue
        vis_train[str(train_text[i] + train_aspect[i])] = True
        out_train_text.append(train_text[i])
        out_train_aspect.append(train_aspect[i])
        out_train_label.append(train_label[i])

    out_test_text = list()
    out_test_aspect = list()
    out_test_label = list()
    for i in range(len(test_aspect)):
        if str(test_text[i] + test_aspect[i]) in vis_test.keys():
            continue
        vis_test[str(test_text[i] + test_aspect[i])] = True
        out_test_text.append(test_text[i])
        out_test_aspect.append(test_aspect[i])
        out_test_label.append(test_label[i])
    out_train_df = pd.DataFrame(list(zip(out_train_text, out_train_aspect, out_train_label)), columns=['text', 'aspect', 'label'])
    out_test_df = pd.DataFrame(list(zip(out_test_text, out_test_aspect, out_test_label)), columns=['text', 'aspect', 'label'])
    out_train_df.to_csv("train.csv")
    out_test_df.to_csv("test.csv")
