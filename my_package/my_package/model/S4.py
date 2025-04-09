import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import opt_einsum as oe
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
import math

all = ["S4Model", "S4Simple"]

class S4Simple(nn.Module):
    def __init__(self, input_dim, n_classes, state_dim = 64, model_dim = 1024):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, model_dim),
                                 nn.ReLU())
        self.s4_block = nn.Sequential(nn.LayerNorm(model_dim),
                                      S4D(d_model=model_dim, d_state=state_dim, transposed=False))
        self.fc2 = nn.Linear(model_dim, n_classes)

    def forward(self, x):
        x = self.fc1(x.unsqueeze(0))
        x = self.s4_block(x)
        x = torch.max(x, axis=1).values# type: ignore
        logits = self.fc2(x)
        return logits[0]

class S4Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_dim = config.model["model_dim"]
        self.state_dim = config.model["state_dim"]
        self.input_dim = config.data["input_dim"]
        self.n_classes = config.data["n_classes"]

        self.fc1 = nn.Sequential(nn.Linear(self.input_dim, self.model_dim),
                                 nn.ReLU())
        self.s4_block = nn.Sequential(nn.LayerNorm(self.model_dim),
                                      S4D(d_model=self.model_dim, d_state=self.state_dim, transposed=False))
        self.fc2 = nn.Linear(self.model_dim, 512)
        self.relu = nn.LeakyReLU()
        self.fc3 = nn.Linear(512, self.n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.s4_block(x)
        x = torch.max(x, axis=1).values# type: ignore
        x = self.fc2(x)
        x = self.relu(x)
        logits = self.fc3(x)
        Y_hat = torch.argmax(logits, dim=0)
        results_dict = {'logits': logits, 'Y_hat': Y_hat}
        return results_dict

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X
#X_dropped = dropout_nd(X),X与X_dropped维度相同
class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters围绕SSKernelDiag的包装器,用于生成对角SSM参数
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # type: ignore # (H)
        C = _r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # type: ignore # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)
#输出维度为【d_model，L】

class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L)  # (H L)
        u_f = torch.fft.rfft(u.to(torch.float32), n=2*L)  # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y
#transposed=True时，input_tensor = torch.randn(batch_size, d_model, sequence_length) ,
#output_tensor = s4d_model(input_tensor) ,output_tensor.shape结果为(batch_size, d_model, sequence_length)

if __name__ == "__main__":
    from thop import profile
    model = S4Simple(512, 6)
    # x = torch.randn((100, 512))
    # with torch.no_grad():
    #     y = model(x)
    # print(y.shape)

    dummy_input = torch.randn(1000, 512)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False) # type: ignore
    print(flops/1e9, params/1e9)