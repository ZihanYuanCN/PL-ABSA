from senta import Senta

my_senta = Senta()

# 获取目前支持的情感预训练模型, 我们开放了以ERNIE 1.0 large(中文)、ERNIE 2.0 large(英文)和RoBERTa large(英文)作为初始化的SKEP模型
print(my_senta.get_support_model()) # ["ernie_1.0_skep_large_ch", "ernie_2.0_skep_large_en", "roberta_skep_large_en"]

# 获取目前支持的预测任务
print(my_senta.get_support_task()) # ["sentiment_classify", "aspect_sentiment_classify", "extraction"]

# 选择是否使用gpu
use_cuda = False # 设置True or False


# 预测中文评价对象级的情感分类任务
my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="aspect_sentiment_classify", use_cuda=use_cuda)
# while True:
#     # text = str(input("请输入text:"))
#     texts = [text]
#     # aspect = str(input("请输入aspect:"))
#     aspects = [aspect]
#     result = my_senta.predict(texts, aspects)
#     print(result)

texts = ["荣耀Play5T有128GB存储空间，突破空间束缚，高清四摄，后置1300万像素主镜头，激昂青春，魅力四射；体验升级全新人机交互体验，双击截取全屏，画圈局部截屏，画S长截屏，开启花式截屏新玩法。手机颜色很漂亮，是那种稳重的蓝色，高贵而不失典雅，拍照清晰，反应速度快，电池大5000毫安，内存大128G，还有收音机功能呢，最重要的是价钱很合适，想买手机的快点下单吧。"]
aspects = ["手机"]
result = my_senta.predict(texts, aspects)
print(result)


texts = ["荣耀Play5T有128GB存储空间，突破空间束缚，高清四摄，后置1300万像素主镜头，激昂青春，魅力四射；体验升级全新人机交互体验，双击截取全屏，画圈局部截屏，画S长截屏，开启花式截屏新玩法。手机颜色很漂亮，是那种稳重的蓝色，高贵而不失典雅，拍照清晰，反应速度快，电池大5000毫安，内存大128G，还有收音机功能呢，最重要的是价钱很合适，想买手机的快点下单吧。"]
aspects = ["电池"]
result = my_senta.predict(texts, aspects)
print(result)


# # 预测中文观点抽取任务
# my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="extraction", use_cuda=use_cuda)
# texts = ["唐 家 三 少 ， 本 名 张 威 。"]
# result = my_senta.predict(texts, aspects)
# print(result)
