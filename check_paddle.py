import paddle
import torch

def check_paddle():
    print(torch.cuda.is_available())
if __name__ == "__main__":
    check_paddle()
