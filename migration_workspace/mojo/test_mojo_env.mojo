from mojo.tensor import Tensor, TensorShape, DType

fn main():
    let shape = TensorShape(2, 2)
    let tensor = Tensor(DType.float32, shape)
    print("Mojo environment test successful!")
    print("Tensor shape:", tensor.shape())
