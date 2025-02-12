import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from subset_sampler import subset_sampler
from processor import Processor

def collator(data, transform):
    '''
    데이터 로더 집합 함수
    Args:
        data: 데이터로더에서 불러온 배치
        transform: 모델에 맞는 입력 구조 변환
    Returns:s
        {"pixel_values": pixel_values, "labels": labels}
        pixel_values: (배치크기, 채널 수, 이미지 높이, 이미지 너비)
        labels: 클래스 인덱스 값값
    '''
    images, labels = zip(*data)
    pixel_values = torch.stack([transform(image) for image in images])
    labels = torch.tensor([label for label in labels])
    return {"pixel_values": pixel_values, "labels": labels}# ViT 모델의 입력 형태



if __name__=="__main__":
    processor = Processor()
    
    train_dataset = datasets.FashionMNIST(root="./datasets", download=True, train=True)
    test_dataset = datasets.FashionMNIST(root="./datasets", download=True, train=False)
    
    subset_train_dataset = subset_sampler(
        dataset=train_dataset, classes=train_dataset.classes, max_len=1000
    )
    subset_test_dataset = subset_sampler(
        dataset=test_dataset, classes=test_dataset.classes, max_len=100
    )
    
    train_dataloader = DataLoader(
        subset_train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: collator(x, processor.transform()),# 데이터 로더에 집합함수 적용 -> 미니배치 샘플 목록 병합
        drop_last=True
    )
    valid_dataloader = DataLoader(
        subset_test_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: collator(x, processor.transform()),# 데이터 로더에 집합함수 적용 -> 미니배치 샘플 목록 병합
        drop_last=True
    )

    batch=next(iter(train_dataloader))
    for key, value in batch.items():
        print(f"{key} : {value.shape}")