import os
import json
from PIL import Image
import time
from datasets import Dataset, Features, Image as DImage, Value, Sequence
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch
import torchvision.transforms as transforms

from PIL import Image
import torchvision.transforms as T

class FluxDataset(TorchDataset):
    def __init__(self, dataset_root, image_res = 512, mask_res = 64):
        self.dataset = self.load_custom_dataset(dataset_root)
        self.image_res = image_res
        self.mask_res = mask_res
        self.image_transform = transforms.Compose([
            transforms.Resize(image_res),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(mask_res),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 检查 item['image'] 是否已经是 PIL.Image 对象
        if isinstance(item['image'], Image.Image):
            image = self.image_transform(item['image'].convert('RGB'))
        else:
            image = self.image_transform(Image.open(item['image']).convert('RGB'))
        
        masks = []
        for mask_file in item['mask_files']:
            mask = self.mask_transform(Image.open(mask_file).convert('L'))
            mask = self.ensure_one_point(mask)
            masks.append(mask)
        masks = torch.stack(masks)
        
        return {
            "image": image,
            "file_name": item['file_name'],
            "height": item['height'],
            "width": item['width'],
            "prompt": item['prompt'],
            "category": item['category'],
            "phrase": item['phrase'],
            "masks": masks,
            "count": item['count']
        }

    @staticmethod
    def ensure_one_point(mask):
        if mask.max() < 0.5:
            max_val = mask.max()
            mask[mask == max_val] = 1
            mask[mask < 1] = 0
        else:
            mask = (mask >= 0.5).float()
        return mask

    @staticmethod
    def load_custom_dataset(dataset_root):
        metadata = []
        with open(os.path.join(dataset_root, "metadata.jsonl"), "r") as f:
            for line in f:
                metadata.append(json.loads(line.strip()))

        mask_files_dict = {}
        masks_dir = os.path.join(dataset_root, "masks")
        for mask_file in os.listdir(masks_dir):
            if mask_file.endswith('.jpg'):
                base_name = mask_file.rsplit('_', 1)[0]
                if base_name not in mask_files_dict:
                    mask_files_dict[base_name] = []
                mask_files_dict[base_name].append(mask_file)

        data = []
        for item in metadata:
            image_path = os.path.join(dataset_root, "images", item["file_name"])
            
            base_filename = os.path.splitext(item["file_name"])[0]
            mask_files = mask_files_dict.get(f'{base_filename}_mask', [])
            mask_files.sort()

            data_item = {
                "image": image_path,
                "file_name": item["file_name"],
                "height": item["height"],
                "width": item["width"],
                "prompt": item["prompt"],
                "category": item["category"],
                "phrase": item["phrase"],
                "mask_files": [os.path.join(dataset_root, "masks", mf) for mf in mask_files],
                "count": len(mask_files)
            }
            data.append(data_item)

        features = Features({
            "image": DImage(),
            "file_name": Value("string"),
            "height": Value("int64"),
            "width": Value("int64"),
            "prompt": Value("string"),
            "category": Value("string"),
            "phrase": Value("string"),
            "mask_files": Sequence(Value("string")),
            "count": Value("int64")
        })

        return Dataset.from_list(data, features=features)

def test_loading_speed(dataset_root, batch_size=1, num_workers=0, num_batches=100, pin_memory=False, persistent_workers=False):
    dataloader = DataLoader(FluxDataset(dataset_root), batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
    end_time = time.time()
    
    total_time = end_time - start_time
    samples_per_second = (num_batches * batch_size) / total_time
    
    print(f"加载 {num_batches} 个batch（共 {num_batches * batch_size} 个样本）用时: {total_time:.2f} 秒")
    print(f"每秒加载样本数: {samples_per_second:.2f}")

if __name__ == "__main__":
    dataset_root = "/mnt/pentagon/gzeng/InstanceToken-UNet/data/FluxCount"
    
    print("测试单线程加载速度（10个batch）:")
    test_loading_speed(dataset_root, num_batches=10, num_workers=1, pin_memory=True, persistent_workers=True)

    print("\n数据集信息:")
    flux_dataset = FluxDataset(dataset_root)
    print(f"数据集大小: {len(flux_dataset)}")
    first_item = flux_dataset[0]
    
    image = first_item['image']
    masks = first_item['masks']
    
    # 保存image和masks到本地
    image_path = f"/mnt/pentagon/gzeng/InstanceToken-UNet/data/{first_item['file_name']}"
    masks_dir = f"/mnt/pentagon/gzeng/InstanceToken-UNet/data/masks_{os.path.splitext(first_item['file_name'])[0]}"

    # 创建保存掩码的目录
    os.makedirs(masks_dir, exist_ok=True)

    # 将张量转换为PIL图像并保存
    to_pil = T.ToPILImage()

    # 保存图像
    pil_image = to_pil(image)
    pil_image.save(image_path)
    print(f"图像已保存到: {image_path}")

    # 保存掩码
    for i in range(masks.shape[0]):
        mask = masks[i, 0]  # 取出第i个掩码，去掉channel维度
        pil_mask = to_pil(mask)
        mask_path = os.path.join(masks_dir, f"mask_{i}.png")
        pil_mask.save(mask_path)
        print(f"掩码 {i} 已保存到: {mask_path}")

    print(f"所有掩码已保存到目录: {masks_dir}")
    
    import pdb; pdb.set_trace()
    
    print(first_item['image'])
    print(first_item['masks'])
    
    print(f"第一个样本的信息: {first_item['file_name']}, 类别: {first_item['category']}, 数量: {first_item['count']}")