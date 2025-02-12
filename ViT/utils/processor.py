import torch
from torchvision import transforms
from transformers import AutoImageProcessor

class Processor:
    def __init__(self):
        self.processor_name = "google/vit-base-patch16-224-in21k"
        self.image_processor = None
        self.process()
        
    def process(self):
        '''
        ViT가 사전학습될 때와 동일한 방식으로 전처리하기 위한 AutoClass

        ex)
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

            print(processor.size) # {'height': 224, 'width': 224}
            print(processor.do_resize) # True
            print(processor.do_normalize) # True
            print(processor.image_mean) # [0.485, 0.456, 0.406]
            print(processor.image_std) # [0.229, 0.224, 0.225]

        '''
        self.image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path=self.processor_name
        )

    def transform(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),# 텐서화
                transforms.Resize(
                    size=(
                        self.image_processor.size["height"],
                        self.image_processor.size["width"]
                    )
                ),
                transforms.Lambda(# 3 dim
                    lambda x: torch.cat([x, x, x], 0)
                ),
                transforms.Normalize(
                    mean=self.image_processor.image_mean,
                    std=self.image_processor.image_std
                )
            ]
        )
        return transform


if __name__=="__main__":
    processor = Processor()
    print(f"size:{processor.image_processor.size}")
    print(f"size:{processor.image_processor.image_mean}")
    print(f"size:{processor.image_processor.image_std}")