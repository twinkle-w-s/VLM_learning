import torch

def recall_at_k(similarity,k):
    k=min(k,similarity.shape[1])

    topk_indices=similarity.topk(
        k=k,
        dim=1,
    ).indices

    targets=torch.arange(
        similarity.shape[0],
        device=similarity.device
    ).unsqueeze(1)#torch.arange生成一个长度和similarity一样的一维向量，.unsqueeze(1)随后增加第一维，变成[1,4]的形式

    correct=(topk_indices==targets).any(dim=1)#any(dim=1)表示维度1上只要有一个为true，就是true

    return correct.float().mean().item()#返回浮点数，即这组batch里满足topk正确的平均数

def evaluate_retrieval(model,dataloader,device,ks=(1,5,10)):
    model.eval()

    all_image_features=[]
    all_text_features=[]

    with torch.inference_mode():
        for batch in dataloader:
            pixel_values =batch["pixel_values"].to(
                device,
                non_blocking=True,
            )
            input_ids =batch["input_ids"].to(
                device,
                non_blocking=True,
            )

            image_features = model.encode_image(pixel_values)
            text_features = model.encode_text(input_ids)

            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )

            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True,
            )

            all_image_features.append(
                image_features.float().cpu()
            )

            all_text_features.append(
                text_features.float().cpu()
            )#特征从gpu回到cpu，防止占用显存


        image_features=torch.cat(
            all_image_features,
            dim=0,
        ).to(device)

        text_features=torch.cat(
            all_text_features,
            dim=0,
        ).to(device)

        similarity_i2t=image_features@text_features.T
        similarity_t2i=similarity_i2t.T

        metrics={}

        for k in ks:
            metrics[f"image2text_recall@{k}"]=recall_at_k(similarity_i2t,k)
            metrics[f"text2image_recall@{k}"]=recall_at_k(similarity_t2i,k)

        metrics["mean_recall@1"]=(
            metrics[f"image2text_recall@1"]+metrics[f"text2image_recall@1"]
        )/2

        return metrics