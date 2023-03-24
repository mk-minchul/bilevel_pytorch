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


def bilevel_optim(model, train_input, train_label, train_loss_fn, val_input, val_label, val_loss_fn, k=1, inner_lr=1.0):

    model_params = list(model.parameters())
    if k == 0:
        val_loss = val_loss_fn(model(val_input), val_label)
        model_grad = grad(val_loss, model_params, retain_graph=True, create_graph=True, allow_unused=False)
        return model_grad

    # inner loop (update model using train_input)
    for i in range(k):
        train_loss = train_loss_fn(model(train_input), train_label)
        model_grad = grad(train_loss, model_params, retain_graph=True, create_graph=True, allow_unused=False)
        model = manual_update(model=model, lr=inner_lr, grads=model_grad)
        print("Current GPU memory usage:", torch.cuda.memory_allocated() / (1024 ** 3), "GB")

    # outer loop (use loss of val_input to get the train_input)
    val_loss = val_loss_fn(model(val_input), val_label)
    train_input_grad = grad(val_loss, [train_input], retain_graph=True, create_graph=True, allow_unused=False)
    train_input_grad = train_input_grad[0] if len(train_input_grad) == 1 else train_input_grad
    print("Current GPU memory usage:", torch.cuda.memory_allocated() / (1024 ** 3), "GB")

    return train_input_grad


def dummy_loss_fn(pred, loss):
    return pred


if __name__ == '__main__':

    # init model and input
    input = torch.tensor(2.0, requires_grad=True)
    new_input = torch.tensor(4.0, requires_grad=False)
    model_param = torch.tensor(3.0, requires_grad=True)
    model = MultiplyModule(model_param)

    input_grad = bilevel_optim(model,
                               train_input=input, train_label=None, train_loss_fn=dummy_loss_fn,
                               val_input=new_input, val_label=None, val_loss_fn=dummy_loss_fn, k=1)
    print(input_grad)

    from torchvision.models import resnet50
    import time
    model = resnet50(pretrained=True).cuda()
    poison = torch.randn(1,3,224,224).clone().detach().requires_grad_(True).cuda()
    train_input = torch.randn(10,3,224,224).cuda() + poison
    train_label = torch.ones(10).long().cuda()
    val_input = torch.randn(10,3,224,224).clone().detach().requires_grad_(True).cuda()
    val_label = torch.zeros(10).long().cuda()

    start = time.time()
    input_grad = bilevel_optim(model,
                               train_input=train_input, train_label=train_label, train_loss_fn=nn.CrossEntropyLoss(),
                               val_input=val_input, val_label=val_label, val_loss_fn=nn.CrossEntropyLoss(), k=1)
    end = time.time()

    # Print the current and maximum GPU memory usage
    print("Maximum GPU memory usage:", torch.cuda.max_memory_allocated() / (1024 ** 3), "GB")
    print('Time Taken: {}'.format(end-start))