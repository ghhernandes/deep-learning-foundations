from torch.autograd import Function



class ReluFunction(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.clamp_min(0.)
        ctx.save_for_backward(i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        return (i>0).float() * grad_output



class MSE(Function):
    @staticmethod
    def forward(ctx, x, targets):
        result = (x.squeeze() - targets).pow(2).mean()
        ctx.save_for_backward(x, targets)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x, targets = ctx.saved_tensors
        return (2. * (x.squeeze() - targets).unsqueeze(-1)) / targets.shape[0]



class LinearFunction(Function):
    @staticmethod
    def forward(ctx, i, w, b):
        result = i @ w.t() + b
        ctx.save_for_backward(i, w, b)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        i, w, b = ctx.saved_tensors
        return w.t() @ grad_output
