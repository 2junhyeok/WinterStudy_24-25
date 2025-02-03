import torch

class GradCAM:
    def __init__(self, model, main, sub):
        '''
        ex)
            main="layer4",
            sub="conv2"
        '''
        self.model = model.eval()# model 평가모드
        self.register_hook(main, sub)
        
    def register_hook(self, main, sub):
        '''
        Hook: 특정 이벤트 발생 시, 다른 코드 실행
        
        '''
        for name, module in self.model.named_children():# 모듈의 이름과 모듈 반환 메서드
            if name == main:
                for sub_name, sub_module in module[-1].named_children():
                    if sub_name == sub:
                        sub_module.register_forward_hook(self.forward_hook)
                        sub_module.register_full_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):# 시그니처 일치를 위한 인자들
        '''
        forward 시 특정 layer의 output을 저장
        '''
        self.feature_map = output# f_k(i,j) 
        
    def backward_hook(self, module, grad_input, grad_output):
        '''
        backward시, 특정 layer의 출력에 대한 손실의 gradient 저장
        '''
        self.gradient = grad_output[0]

    def __call__(self, x):
        '''
        class의 객체를 함수처럼 호출
        '''
        output = self.model(x)
        
        index = output.argmax(axis=1)# [N, 1000]의 output에서 index 추출
        one_hot = torch.zeros_like(output)# output size의 0 vector
        for i in range(output.size(0)):
            one_hot[i][index[i]]=1# index의 위치에만 1 부여
            
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)#?
        
        a_k = torch.mean(self.gradient, dim=(2, 3), keepdim=True)# [N, 512, 7, 7] -> [N, 512, 1, 1]
        grad_cam = torch.sum(a_k * self.feature_map, dim=1)
        grad_cam = torch.relu(grad_cam)
        return grad_cam# [N, 7, 7]