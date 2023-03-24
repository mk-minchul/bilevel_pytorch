from torch.autograd import grad
import torch
from torch.optim.sgd import SGD
import torch.nn as nn

def print_status(grad, start, end, prefix):
    if isinstance(grad, torch.Tensor):
        grad = grad.item()
    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()

    return print(f'{prefix} Gradient: {grad}. (Parameter: {start} -> {end}) {"not updated" if start == end else ""}')


def manual_update(model, lr, grads=None):
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = 'WARNING:manual_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            print(msg)
        for p, g in zip(params, grads):
            if g is not None:
                p.update = - lr * g
    return update_module(model)

def update_module(module, updates=None, memo=None):
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p in memo:
            module._parameters[param_key] = memo[p]
        else:
            if p is not None and hasattr(p, 'update') and p.update is not None:
                updated = p + p.update
                p.update = None
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        else:
            if buff is not None and hasattr(buff, 'update') and buff.update is not None:
                updated = buff + buff.update
                buff.update = None
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    return module

class MultiplyModule(nn.Module):
    def __init__(self, model):
        super(MultiplyModule, self).__init__()
        self.param = nn.Parameter(model)

    def forward(self, x):
        return self.param * x


if __name__ == '__main__':

    # manual gradient
    input = torch.tensor(2.0, requires_grad=True)
    model_param = torch.tensor(3.0, requires_grad=True)
    model = MultiplyModule(model_param)
    input_start = input.item()
    model_start = model.param.item()
    loss = model(input)
    gradients = grad(loss,   # loss
                     list(model.parameters())+[input], # parameters
                     retain_graph=True,
                     create_graph=True,
                     allow_unused=False)
    model_grad, input_grad = gradients[:1], gradients[1:]
    model = manual_update(model=model, lr=1.0, grads=model_grad)
    print_status(model_grad[0], model_start, model.param, 'Model')
    print_status(input_grad[0], input_start, input, 'Input')
    input_start = input.item()
    model_start = model.param.item()

    new_input = torch.tensor(4.0, requires_grad=False)
    loss = model(new_input)
    loss.backward()
    print_status(model_grad[0], model_start, model.param, 'Model')
    print_status(input.grad, input_start, input, 'Input')

    # manual gradient (Unfold 2 steps)
    input = torch.tensor(2.0, requires_grad=True)
    model_param = torch.tensor(3.0, requires_grad=True)
    model = MultiplyModule(model_param)
    input_start = input.item()
    model_start = model.param.item()
    loss = model(input)
    gradients = grad(loss,   # loss
                     list(model.parameters())+[input], # parameters
                     retain_graph=True,
                     create_graph=True,
                     allow_unused=False)
    model_grad, input_grad = gradients[:1], gradients[1:]
    model = manual_update(model=model, lr=1.0, grads=model_grad)
    print(f'model grad: {model_grad[0].item()}, updated param: {model_start} --> {model.param.item()}')
    print(f'input grad: {input_grad[0].item()}, updated param: {input_start} --> {input.item()}  {"not updated" if input_start == input.item() else ""}')
    input_start = input.item()
    model_start = model.param.item()

    loss = model(input)
    gradients = grad(loss,   # loss
                     list(model.parameters())+[input], # parameters
                     retain_graph=True,
                     create_graph=True,
                     allow_unused=False)
    model_grad, input_grad = gradients[:1], gradients[1:]
    model = manual_update(model=model, lr=1.0, grads=model_grad)
    print(f'model grad: {model_grad[0].item()}, updated param: {model_start} --> {model.param.item()}')
    print(f'input grad: {input_grad[0].item()}, updated param: {input_start} --> {input.item()}  {"not updated" if input_start == input.item() else ""}')
    input_start = input.item()
    model_start = model.param.item()

    new_input = torch.tensor(4.0, requires_grad=False)
    loss = model(new_input)
    loss.backward()
    input.grad


