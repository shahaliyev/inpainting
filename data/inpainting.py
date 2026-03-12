from torch.utils.data import Dataset


class InpaintingDataset(Dataset):
    def __init__(self, dataset: Dataset, mask_gen):
        self.dataset = dataset
        self.mask_gen = mask_gen

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]

        mask = self.mask_gen(image.shape)
        masked_image = image * (1.0 - mask)

        return {
            "image": image,
            "mask": mask,
            "masked_image": masked_image,
            "path": item.get("path"),
        }