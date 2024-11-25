import torch
import os
from typing import Callable, Iterable
from src.interpolation.utils import LagrangePolynomialInterpolation
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

class PolynomialInterpolation(torch.nn.Module):
    """Base class for all polynomial interpolation models
    Inherited classes should implement:
    - property: sample_num
    - property: sample_points, equally initialized
    - property: sample_values, linearly initialized, y = x
    - method: forward_values
    - method: update
    - method: (optional) prepare_to_fit
    - or reset any according to need
    - to.(device) for non-nn.Parameter: intervals, bl, br
    """
    N: int # number of intervals
    degree: int # degree of polynomial
    intervals: torch.Tensor # used for masking, update, detach from sample_points
    sample_num: int # number of sample points
    sample_points: torch.nn.Parameter # used for interpolation, trainable
    sample_values: torch.nn.Parameter # used for interpolation, trainable
    bl: torch.Tensor # left boundary for interpolation, update from sample_points
    br: torch.Tensor # right boundary for interpolation, update from sample_points
    sl: torch.nn.Parameter # slope outside left boundary for interpolation, trainable
    sr: torch.nn.Parameter # slope outside right boundary for interpolation, trainable
    min_val: float = None # minimum value of output
    max_val: float = None # maximum value of output
    post_process: Callable # post process function
    interpolate: Callable # interpolation function
    matrixlt: torch.Tensor # used for masking transform, N + 1 x N
    matrixgt: torch.Tensor # used for masking transform, N + 1 x N

    objective_func: Callable # objective function for fitting
    def __init__(self, N=10, degree=1, bl=-10., br=10., sl=0., sr=0., 
                 device="cpu", dtype=torch.float32, min_val=None, max_val=None, logging_dir="./"):
        super(PolynomialInterpolation, self).__init__()
        # unexpected behavior
        if degree < 1:
            raise ValueError("polynomial degree should be no less than 1")
        if N < 1:
            raise ValueError("N should be no less than 1")
        if bl >= br:
            raise ValueError("bl should be less than br")
        
        self.device = device
        self.dtype = dtype
        
        # initialization
        self.N = N
        self.degree = degree
        self.bl = torch.tensor(bl, dtype=dtype, device=device)
        self.br = torch.tensor(br, dtype=dtype, device=device)
        # init as equal interval
        self.intervals = torch.linspace(bl, br, N + 1, dtype=dtype, device=device)
        self.sl = torch.nn.Parameter(torch.tensor(sl, dtype=dtype))
        self.sr = torch.nn.Parameter(torch.tensor(sr, dtype=dtype))
        self.min_val = min_val
        self.max_val = max_val
        # clamp output to desired range. None for no clamp.
        self.post_process = lambda output: output.clamp(self.min_val, self.max_val) \
            if self.min_val is not None and self.max_val is not None else output
        self.interpolate = LagrangePolynomialInterpolation()

        # used for masking
        self.matrixlt = torch.cat((torch.eye(self.N), torch.zeros(1, self.N)), 0).to(device=device, dtype=dtype) # N + 1 x N
        self.matrixgt = torch.cat((torch.zeros(1, self.N), torch.eye(self.N)), 0).to(device=device, dtype=dtype) # N + 1 x N
        
        # check for fast compute
        # TODO: fixed and freeze if retrain?
        self.has_exported = False

        # logging
        self._logging_dir = logging_dir
        # used for logging data distribution
        # TODO: need to clean up or cache when resume training
        self.distribution = torch.tensor([], device=device, dtype=dtype)

        # TODO: need a common parameter for all models? torch.nn.Paramter for trainable parameters. tensor for non-trainable parameters
        # TODO: based on above, we can figure out a way to export funciton

    @property
    def logging_dir(self):
        os.makedirs(self._logging_dir, exist_ok=True)
        return self._logging_dir

    def forward(self, input):
        output = self.forward_values(input)
        return self.post_process(output)

    def forward_values(self, input):
        """use polynomial to calculate output
        call self.export_func() first to export coefficients

        self.coefficients: N x degree + 1
        input: shape

        so far 3x faster
        """
        # TODO: do we need to handle non-batch input?
        shape = input.shape

        # calculate polynomial interpolation
        # Vandermonde matrix
        X = torch.ones((*shape, self.degree + 1), dtype=self.dtype, device=self.device)
        for i in range(1, self.degree + 1):
            X[..., i] = input ** i
        yp = torch.matmul(X, self.coefficients.transpose(-1, -2)) # shape x N

        # calculate boundary values
        ybl = self.sl * (input - self.bl) + self.sample_values[0]
        ybr = self.sr * (input - self.br) + self.sample_values[-1]

        # interval boundary mask
        maskbl, maskbr, maskother = self.handle_boundary(input)
        return maskbl * ybl + maskbr * ybr + torch.sum(maskother * yp, -1)

    def set_freeze(self, params = None, freeze=True):
        if params is None:
            named_params = self.named_parameters()
        else:
            if isinstance(params, str) or not isinstance(params, Iterable):
                params = [params]
            named_params = [(param, self.get_parameter(param)) for param in params]
        print("set {}freeze:".format("" if freeze else "un"))
        for name, param in named_params:
            param.requires_grad = not freeze
            print(name, end=", ")
        print()

    def get_name(self):
        # TODO: what for?
        return """{}_N_{}_degree_{}""".format(self._get_name(), self.N, self.degree)

    def extra_repr(self) -> str:
        return """N={}, degree={}, bl={}, br={}, sl={}, sr={}, min_val={}, max_val={}""".format(
            self.N, self.degree, self.bl.item(), self.br.item(), self.sl.item(), self.sr.item(), self.min_val, self.max_val
        )

    def handle_boundary(self, input):
        '''
        interpolate within the respective interval
        use mask to select the interval
        '''

        maskbl = input <= self.intervals[0] # mask for left boundary
        maskbr = input >= self.intervals[-1] # mask for right boundary

        # mask for inside boundaries
        input = torch.unsqueeze(input, -1)
        masklt = self.intervals <= input # B x N + 1
        maskgt = self.intervals > input # B x N + 1
        # matrixlt N + 1 x N
        # matrixgt N + 1 x N
        masklt = torch.matmul(masklt.float(), self.matrixlt) # B x N
        maskgt = torch.matmul(maskgt.float(), self.matrixgt) # B x N

        maskother = masklt * maskgt
        return maskbl, maskbr, maskother

    def prepare_to_fit(self):
        """
        init necessary parameters respective to objective function
        """

    def fit(self, objective_func, generator, optimizer_class, criterion_class, epoch=10000, batch=128, lr=None, xmin=-10, xmax=10, step=0.001):
        self.objective_func = objective_func
        self.prepare_to_fit()

        lr = ((self.br - self.bl) / self.N * 0.001) if lr is None else lr
        print("Learning rate: ", lr)

        # TODO: need a learning rate scheduler?
        # TODO: other training configurations

        self.train()
        optimizer = optimizer_class(self.parameters(), lr=lr)
        criterion = criterion_class()

        # can continue training
        file = open(os.path.join(self.logging_dir, "log.txt"), "a")
        writer = SummaryWriter(os.path.join(self.logging_dir, "tensorboard"))
        fig, ax = plt.subplots()
        images = []
    
        for i in tqdm(range(epoch)):
            data = generator(batch).to(self.device)
            output = self(data)
            label = objective_func(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            msg = {"epoch": i, "loss": loss.item()}
            file.write(str(msg) + "\n")
            writer.add_scalar("train_loss", loss, i)
            self.distribution = torch.cat((self.distribution, data), 0)
            if i % 10 == 0 or i == epoch - 1:
                text = plt.text(0.5, 1.03, i, horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
                ref, eval = self.test(objective_func, xmin, xmax, step)
                images.append([ref, eval, text])

            self.update()


        file.close()
        writer.close()
        ArtistAnimation(fig, images, repeat=True, blit=False, interval=100, repeat_delay=2000).save(os.path.join(self.logging_dir, "progress.gif"), writer="pillow")
        fig.clear()
        ax.clear()

        plt.title("input distribution")
        plt.hist(self.distribution.cpu().detach().numpy(), bins=100)
        plt.savefig(os.path.join(self.logging_dir, "input_distribution.png"))
        plt.clf()

    def test(self, objective_func, xmin=-10, xmax=10, step=0.001):
        self.eval()
        with torch.no_grad():
            x = torch.arange(xmin, xmax, step).to(self.device)
            y = self(x)
            ref_y = objective_func(x)
        
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            ref_y = ref_y.cpu().detach().numpy()

        ref, = plt.plot(x, ref_y, label="ref", color="blue")
        eval, = plt.plot(x, y, label="train", color="orange")
        return ref, eval
        # return title + ref + eval + legend
        # plt.savefig(os.path.join(self.logging_dir, "activation.png"))
        # plt.clf()

        # for name, param in self.named_parameters():
        #     print(name, param)
        # print("Intervals: ", self.intervals)

    # TODO: why commented?
    def to(self, *args, **kwargs):
        super(PolynomialInterpolation, self).to(*args, **kwargs)
        self.intervals = self.intervals.to(*args, **kwargs)
        self.matrixlt = self.matrixlt.to(*args, **kwargs)
        self.matrixgt = self.matrixgt.to(*args, **kwargs)
        self.device = self.intervals.device

    # TODO: auto update is the key to the traditional training
    # how about we just calculate in real time? time-consuming if not updated?
    def update(self):
        # call update when fitting
        raise NotImplementedError("""at least update: intervals, bl, br""")

    def _init(self):
        raise NotImplementedError("""at least init: intervals, bl, br""")

    def load_state_dict(self, *args, **kwargs):
        super(PolynomialInterpolation, self).load_state_dict(*args, **kwargs)
        self._init() # Important: update non-parameters, continue training may have error initial state

    def export_params(self, filename):
        raise NotImplementedError

    def export_func(self):
        """calculate parameters of polynomials interpolation for fast compute
        method:
            polynomial: P_n(x) = a0 + a1 * x + a2 * x^2 + ... + an * x^n
            interpolation is a series of equations: P_n(xi) = yi, i = 0, 1, ..., n
            solve the coeffients: V_n+1 * A = Y -> A = V_n+1^-1 * Y

            V_n+1: Vandermonde matrix, n + 1 x n + 1 -> N x n + 1 x n + 1
            1 x0 x0^2 ... x0^n
            1 x1 x1^2 ... x1^n
            ...
            1 xn xn^2 ... xn^n

            A: coefficients, n + 1 x 1 -> N x n + 1 x 1
            a0 a1 ... an

            Y: values, n + 1 x 1 -> N x n + 1 x 1
            y0 y1 ... yn

        warning:
            singular matrix may cause error
        """
        try:
            Y = self.sample_values.cpu().detach().clone()
            X = self.sample_points.cpu().detach().clone()
            V = torch.ones((self.N, self.degree + 1, self.degree + 1), dtype=self.dtype)
            A = torch.zeros((self.N, self.degree + 1), dtype=self.dtype)
            for i in range(self.N):
                xp = X[i * self.degree: (i + 1) * self.degree + 1]
                yp = Y[i * self.degree: (i + 1) * self.degree + 1]
                for j in range(1, self.degree + 1):
                    V[i, :, j] = xp ** j
                A[i, :] = torch.matmul(torch.inverse(V[i]), yp)
            self.coefficients = A.to(self.device)
            self.has_exported = True
            print("model polynomial function exported")
        except torch._C._LinAlgError:
            print("singular matrix, could not export coefficients. please check sample points")
