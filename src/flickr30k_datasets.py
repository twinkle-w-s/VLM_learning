import os
from datasets import load_dataset
from torch.utils.data import Dataset

class Flickr30KDataset(Dataset):
    def __init__(self,split_init,preprocess,tokenizer):
        storage_root = f"/data/{os.environ['USER']}/vlm_learning"
        dataset_cache = f"{storage_root}/datasets/flickr30k"

        full_dataset=load_dataset(
            "nlphuji/flickr30k",
            split="test",#这里的test好像仅表示全部数据，这个数据集构造具有特殊性
            cache_dir=dataset_cache,
            trust_remote_code=True,
        )

        self.dataset=full_dataset.filter(
            lambda sample:sample["split"]==split_init
        )#筛选数据集中符合split的sample

        self.preprocess=preprocess
        self.tokenizer=tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):#逐条抓取数据的方法定义
        sample = self.dataset[index]

        image=sample["image"]
        captions=sample["caption"]
        caption=captions[0]#数据集中对同一个图片有多个不同的描述，这里只用第一个

        pixel_values=self.preprocess(image)#prepross是外部传入的数据预处理方法
        input_ids=self.tokenizer([caption].squeeze(0))

        return{
            "pixel_values":pixel_values,
            "input_ids":input_ids,
            "caption":caption,
        }