from itertools import chain
from collections import defaultdict
from torch.utils.data import Subset
from torchvision import datasets

def subset_sampler(dataset, classes, max_len):
    '''
    데이터 서브 샘플링 수행
    :dataset: 데이터 셋
    :classes: 클래스 목록
    :max_len: 클래스 별 최대 샘플링 개수
    '''
    target_idx = defaultdict(list)# 기본값이 있는 딕셔너리 클래스
    for idx, label in enumerate(dataset.train_labels):# datset.train_labels or test labels: 정답 라벨 텐서
        target_idx[int(label)].append(idx)# defaultdict는 dict에 "int(label)"이라는 key가 없어도 if문 필요없이이 자동으로 만들어줌.
    
    indices = list(
        chain.from_iterable(
            [target_idx[idx][:max_len] for idx in range(len(classes))]# 인덱스를 1차원 리스트로 풀어준다.
        )
    )
    return Subset(dataset, indices)# 전체 데이터 셋, 추려낼 데이터 인덱스 리스트

if __name__=="__main__":
    train_dataset = datasets.FashionMNIST(root="../datasets", download=True, train=True)
    test_dataset = datasets.FashionMNIST(root="../datasets", download=True, train=False)

    classes = train_dataset.classes# classes 속성: 데이터 셋에 포함된 클래스
    class_to_idx = train_dataset.class_to_idx# class_to_idx속성: 클래스 ID와 클래스가 매핑된 값

    print(classes)
    print(class_to_idx)

    subset_train_dataset = subset_sampler(
        dataset=train_dataset, classes=train_dataset.classes, max_len=1000
    )
    subset_test_dataset = subset_sampler(
        dataset=test_dataset, classes=test_dataset.classes, max_len=100
    )
    print(f"training data size : {len(subset_train_dataset)}")
    print(f"testing data size : {len(subset_test_dataset)}")
    print(train_dataset[0])