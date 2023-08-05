"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar, )


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        ipt = node.inputs[0]
        grad = out_grad * self.scalar * power_scalar(ipt, self.scalar - 1)
        return [grad]


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs ** 2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes:
            return array_api.swapaxes(a, *self.axes)
        else:
            return array_api.swapaxes(a, -1, -2)

    def gradient(self, out_grad, node):
        ret = transpose(out_grad, axes=self.axes)
        return [ret]


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        shape_in = node.inputs[0].shape
        return reshape(out_grad, shape_in)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        # shape_in = node.inputs[0].shape
        # shape_in_p = len(shape_in)-1
        # self.reduce_dim = []
        # # broadcast后的shape从右往左遍历
        # for idx in range(len(out_grad.shape)-1, -1, -1):
        #     if shape_in_p < 0: 
        #         self.reduce_dim.append(idx)
        #         continue
        #     # 否则取broadcast后的dim，和input的dim            
        #     broadcast_dim_size = self.shape[idx]
        #     input_dim_size = shape_in[shape_in_p]
            
        #     # 比较是否相等，如果不等，说明发生了broadcast，需要将当前dim添加到reduce_dim
        #     if broadcast_dim_size != input_dim_size: 
        #         self.reduce_dim.append(idx)
        #     shape_in_p -= 1
        # return array_api.reshape(array_api.sum(out_grad, axis=tuple(self.reduce_dim)), shape_in)
        ipt = node.inputs[0]
        grad = out_grad
        for _ in range(len(out_grad.shape) - len(ipt.shape)):
            grad = summation(grad, axes=0)
        for i, dim in enumerate(ipt.shape):
            if dim == 1:
                grad = grad.cached_data.sum(axis=i, keepdims=True)
            grad = Tensor(grad)

        grad = reshape(grad, ipt.shape)
        return [grad]


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        # shape_in = node.inputs[0].shape
        # shape_out = [1] * len(shape_in)
        # if self.axes:
        #     reduce_dim = set([self.axes]) if isinstance(self.axes, int) else set(self.axes)
        # else:
        #     reduce_dim = set(range(len(shape_in)))
        # out_dim = 0
        # for idx in range(len(shape_in)):
        #     if idx not in reduce_dim:
        #         shape_out[idx] = out_grad.shape[out_dim]
        #         out_dim += 1
        # return broadcast_to(reshape(out_grad, tuple(shape_out)), shape_in)
        ipt = node.inputs[0]
        if self.axes:
            grad = array_api.expand_dims(out_grad.cached_data, self.axes)
            if isinstance(self.axes, int):
                repeat = ipt.shape[self.axes]
                grad = array_api.repeat(grad, repeat, self.axes)
            else:
                repeat = []
                for i in self.axes:
                    repeat.append(ipt.shape[i])
                for r, a in zip(repeat, self.axes):
                    grad = array_api.repeat(grad, r, a)
        else:
            grad = array_api.ones_like(ipt.cached_data) * out_grad.cached_data

        grad = Tensor(grad)

        return [grad]


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        grad_a = matmul(out_grad, transpose(rhs))
        grad_b = matmul(transpose(lhs), out_grad)
        if grad_a.shape != lhs.shape: 
            length = len(grad_a.shape) - len(lhs.shape)
            grad_a = summation(grad_a, axes=tuple(range(length)))
        if grad_b.shape != rhs.shape:
            length = len(grad_b.shape) - len(rhs.shape)
            grad_b = summation(grad_b, axes=tuple(range(length)))
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return out_grad * -1


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return exp(node.inputs[0]) * out_grad


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        ipt = node.inputs[0]
        grad = Tensor(
            (ipt.cached_data > 0).astype(array_api.float32)) * out_grad
        return [grad]


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        maxz = array_api.max(Z, axis=self.axes, keepdims=1)
        ret = array_api.log(
            array_api.exp(Z - maxz).sum(axis=self.axes, keepdims=1)) + maxz
        return array_api.squeeze(ret)

    def gradient(self, out_grad, node):
        ipt = node.inputs[0].cached_data
        maxz = array_api.max(ipt, axis=self.axes, keepdims=1)

        ez = array_api.exp(ipt-maxz)
        sez = array_api.sum(ez, self.axes)
        lsez = array_api.log(sez) + maxz.squeeze()

        d_lsez = out_grad.cached_data
        d_sez = d_lsez / sez

        shape_in = ez.shape
        shape_out = [1] * len(shape_in)
        if self.axes:
            reduce_dim = set(self.axes)
        else:
            reduce_dim = set(range(len(shape_in)))
        out_dim = 0
        for idx in range(len(shape_in)):
            if idx not in reduce_dim:
                shape_out[idx] = d_sez.shape[out_dim]
                out_dim += 1
        d_ez = array_api.broadcast_to(array_api.reshape(d_sez, tuple(shape_out)), shape_in)

        d_z = d_ez * (ez)

        return [Tensor(d_z)]


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
