
from datasets import load_dataset
from torch.utils.data import Dataset

class Flickr30KBLIPDataset(Dataset):
    def __init__(
        self,
        split,
        processor,
        dataset_cache,
    ):
        full_dataset = load_dataset(
            "nlphuji/flickr30k",
            split="test",
            cache_dir=str(dataset_cache),
            trust_remote_code=True,
        )

        self.dataset = full_dataset.filter(
            lambda sample: sample["split"] == split
        )

        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample["image"].convert("RGB")
        caption = sample["caption"][0]

        encoded=self.processor(
            image=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=30,
            truncation=True
        )#返回字典，里面有pixel_values和input_ids

        item={
            name:value.squeeze(0) for name,value in encoded.items()   
        }#把返回的形状为(1,...)的张量压缩成(...)的张量
        item["caption"]=caption#把原始的caption也放进去，方便后续打印
        return item