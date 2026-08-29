import os
from datasets import load_dataset

storage_root=f"/data/{os.environ['USER']}/vlm_learning"
dataset_cache=f"{storage_root}/datasets/flickr30k"

print("loading datasets")

dataset=load_dataset(
    "nlphuji/flickr30k",
    split="test",
    cache_dir=dataset_cache,
    trust_remote_code=True,
)
print("数据集对象：", dataset)
print("数据量：", len(dataset))
print("字段名称：", dataset.column_names)

sample=dataset[0]

for key,value in sample.items():#sample.items()表示将字典中的键值对返回
    if isinstance(value,list):#如果value是list，则打印
        print("列表长度：" ,len(value))
        print("列表前两项：",value[:1])
    else:
        print("内容：", value)

image = sample["image"]

print("\n图片信息：")
print("图片类型：", type(image))
print("图片尺寸：", image.size)
print("图片模式：", image.mode)