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
