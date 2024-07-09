import torch


tensor1 = torch.randn( 3,4)
tensor2 = torch.randn(3,4)
tensor3 = torch.randn(3,4)
print(tensor1,"\n", tensor2,"\n",  tensor3)
# 在第0维拼接，即增加行数
concat0 = torch.cat((tensor1, tensor2, tensor3), dim=0)
print("Concatenated along dimension 0:")
print(concat0)
print("Shape of concat0:", concat0.shape)  # 应该是 (9, 4)