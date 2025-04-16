from torch import nn
from abc import abstractmethod
from typing import Type, TypeVar, Any

T = TypeVar('T')

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: T) -> list[T]:
        raise NotImplementedError

    def decode(self, input: T) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> T:
        raise NotImplementedError

    def generate(self, x: T, **kwargs) -> T:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: T) -> T:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> T:
        pass