import torch
from src.interpolation.polynomial import PolynomialInterpolation
from src.registry import interpolator

@interpolator.register("adaptive")
class AdaptiveIntervalInterpolation(PolynomialInterpolation):
    # Support:
    def __init__(self, N=10, degree=1, bl=-10., br=10., device="cpu", dtype=torch.float32, *args, **kwargs):
        super(AdaptiveIntervalInterpolation, self).__init__( N=N, degree=degree, bl=bl, br=br, device=device, dtype=dtype, *args, **kwargs)
        
        self.sample_num = 1 + N * degree
        # init as equidistant
        self.sample_points = torch.nn.Parameter(torch.linspace(bl, br, self.sample_num, dtype=self.dtype))
        # self.sample_points gradient update is inplace operation. need clone()
        self.sample_points_buffer = self.sample_points.detach().clone()
        self.bl = self.sample_points[0] # same memory address
        self.br = self.sample_points[-1] # same memory address
        # TODO: why this is trainable???
        self.sample_values = torch.nn.Parameter(torch.linspace(bl, br, self.sample_num, dtype=self.dtype).sigmoid()) # sigmoid for faster convergence

    def forward(self, input):
        if self.has_exported:
            output = super(AdaptiveIntervalInterpolation, self).forward(input)
            return self.post_process(output)

        # use mask rather than condition
        shape = input.shape
        input = torch.flatten(input)

        # calculate boundary values
        ybl = self.sl * (input - self.bl) + self.sample_values[0]
        ybr = self.sr * (input - self.br) + self.sample_values[-1]

        # boundary mask
        maskbl, maskbr, maskother = self.handle_boundary(input)

        input = torch.unsqueeze(input, -1)
        yp = torch.tensor([], device=self.device)
        for i in range(self.N):
            yp = torch.cat((yp, self.interpolate(input, self.sample_points[i * self.degree: (i + 1) * self.degree + 1], self.sample_values[i * self.degree: (i + 1) * self.degree + 1])), -1)
        output = maskbl * ybl + maskbr * ybr + torch.sum(maskother * yp, -1)
        return self.post_process(output.reshape(shape))

    # def to(self, *args, **kwargs):
    #     print("tuned")
    #     super(TunedInterval, self).to(*args, **kwargs)
    #     self.sample_points_buffer = self.sample_points_buffer.to(*args, **kwargs)
    #     self.bl = self.bl.to(*args, **kwargs)
    #     self.br = self.br.to(*args, **kwargs)

    def update(self):
        # update sample points
        with torch.no_grad():
            for i in range(self.sample_num):
                if i == 0:
                    self.sample_points[i].clamp_(None, self.sample_points_buffer[i + 1])
                elif i == self.sample_num - 1:
                    self.sample_points[i].clamp_(self.sample_points_buffer[i - 1], None)
                else:
                    self.sample_points[i].clamp_(self.sample_points_buffer[i - 1], self.sample_points_buffer[i + 1])
        # update buffer
        self.sample_points_buffer = self.sample_points.detach().clone()
        # update intervals
        for i in range(self.N + 1):
            self.intervals[i] = self.sample_points[i * self.degree].detach()

    def _init(self):
        # init buffer
        self.sample_points_buffer = self.sample_points.detach().clone()
        # init intervals
        for i in range(self.N + 1):
            self.intervals[i] = self.sample_points[i * self.degree].detach()
