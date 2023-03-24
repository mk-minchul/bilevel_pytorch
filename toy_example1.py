from torch.autograd import grad
import torch
from torch.optim.sgd import SGD

def print_status(grad, start, end, prefix):
    if isinstance(grad, torch.Tensor):
        grad = grad.item()
    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()

    return print(f'{prefix} Gradient: {grad}. (Parameter: {start} -> {end}) {"not updated" if start == end else ""}')

if __name__ == '__main__':


    input_start = 2.0
    model_start = 3.0

    # 1) usual procedure
    input = torch.tensor(2.0, requires_grad=False)
    model = torch.tensor(3.0, requires_grad=True)
    loss = input * model
    sgd = SGD(params=[model], lr=1.0)
    loss.backward()
    sgd.step()
    print_status(model.grad, model_start, model, 'Model')

    # 2) with input and model together optimized
    input = torch.tensor(2.0, requires_grad=True)
    model = torch.tensor(3.0, requires_grad=True)
    loss = input * model
    sgd = SGD(params=[input, model], lr=1.0)
    loss.backward()
    sgd.step()
    print_status(model.grad, model_start, model, 'Model')
    print_status(input.grad, input_start, input, 'Input')

    # 3) update only model but populate the gradient for input
    input = torch.tensor(2.0, requires_grad=True)
    model = torch.tensor(3.0, requires_grad=True)
    loss = input * model
    sgd = SGD(params=[model], lr=1.0)
    loss.backward()
    sgd.step()
    print_status(model.grad, model_start, model, 'Model')
    print_status(input.grad, input_start, input, 'Input')

    # 4) update only model but populate the gradient for input
    # 4.1) inner loop update 1
    input = torch.tensor(2.0, requires_grad=True)
    model = torch.tensor(3.0, requires_grad=True)
    input_start = input.item()
    model_start = model.item()
    loss = input * model
    sgd = SGD(params=[model], lr=1.0)
    loss.backward()
    sgd.step()
    print('Update1')
    print_status(model.grad, model_start, model, 'Model')
    print_status(input.grad, input_start, input, 'Input')

    input.grad = torch.zeros_like(input.grad) # empty grad
    model.grad = torch.zeros_like(model.grad) # empty grad
    input_start = input.item()
    model_start = model.item()
    #  4.2) inner loop update 2
    new_input = torch.tensor(4.0, requires_grad=False)
    new_loss = new_input * model
    sgd = SGD(params=[model], lr=1.0)
    new_loss.backward()
    sgd.step()
    print('Update2')
    print_status(model.grad, model_start, model, 'Model')
    print_status(input.grad, input_start, input, 'Input')
    print('The problem is input grad equal to zero. No gradient flows back to input.')
    print("It doesn't work because SGD update is in-place op.")
