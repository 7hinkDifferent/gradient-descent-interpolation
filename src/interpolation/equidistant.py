import torch
from src.interpolation.polynomial import PolynomialInterpolation
from src.registry import interpolator

@interpolator.register("equidistant")
class EquidistantIntervalInterpolation(PolynomialInterpolation):
    # Support:
    def __init__(self, N=10, degree=1, bl=-10., br=10., device="cpu", dtype=torch.float32, *args, **kwargs):
        super(EquidistantIntervalInterpolation, self).__init__(N=N, degree=degree, bl=bl, br=br, device=device, dtype=dtype, *args, **kwargs)
        
        self.bl = torch.nn.Parameter(torch.tensor(bl, dtype=self.dtype))
        self.br = torch.nn.Parameter(torch.tensor(br, dtype=self.dtype))
        self.sample_num = 1 + N * degree
        self.sample_points = torch.linspace(bl, br, self.sample_num, dtype=self.dtype, device=self.device)
        self.sample_values = torch.linspace(bl, br, self.sample_num, dtype=self.dtype, device=self.device).sigmoid() # sigmoid for faster convergence

    def prepare_to_fit(self):
        self.sample_values = torch.tensor([self.objective_func(x) for x in self.sample_points], device=self.device)

    def forward_values(self, input):
        if self.has_exported:
            output = super(EquidistantIntervalInterpolation, self).forward_values(input)
            return output
        
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
        d = (self.br - self.bl) / self.N / self.degree # point interval length
        for i in range(self.N):
            yp = torch.cat((yp, self.interpolate(input, torch.tensor([self.bl + d * (j + i * self.degree) for j in range(self.degree + 1)], device=self.device), self.sample_values[i * self.degree: (i + 1) * self.degree + 1])), -1)
            # yp = torch.cat((yp, self.interpolate(input, self.sample_points[i * self.degree: (i + 1) * self.degree + 1])), -1) # wrong grad error!
        output = maskbl * ybl + maskbr * ybr + torch.sum(maskother * yp, -1)
        return output.reshape(shape)

    def to(self, *args, **kwargs):
        super(EquidistantIntervalInterpolation, self).to(*args, **kwargs)
        self.sample_points = self.sample_points.to(*args, **kwargs)

    def update(self):
        device = self.bl.device
        # update sample points
        self.sample_points = torch.linspace(self.bl.detach(), self.br.detach(), self.sample_num, dtype=self.dtype, device=device)
        # update intervals
        for i in range(self.N + 1):
            self.intervals[i] = self.sample_points[i * self.degree].detach()

        self.prepare_to_fit()



@interpolator.register("equidistant_tuned_values")
class EquidistantTunedValuesIntervalInterpolation(PolynomialInterpolation):
    # Support:
    def __init__(self, N=10, degree=1, bl=-10., br=10., device="cpu", dtype=torch.float32, *args, **kwargs):
        super(EquidistantTunedValuesIntervalInterpolation, self).__init__(N=N, degree=degree, bl=bl, br=br, device=device, dtype=dtype, *args, **kwargs)
        
        self.bl = torch.nn.Parameter(torch.tensor(bl, dtype=self.dtype))
        self.br = torch.nn.Parameter(torch.tensor(br, dtype=self.dtype))
        self.sample_num = 1 + N * degree
        self.sample_points = torch.linspace(bl, br, self.sample_num, dtype=self.dtype, device=device)
        self.sample_values = torch.nn.Parameter(torch.linspace(bl, br, self.sample_num, dtype=self.dtype).sigmoid()) # sigmoid for faster convergence

    def forward_values(self, input):
        if self.has_exported:
            output = super(EquidistantTunedValuesIntervalInterpolation, self).forward_values(input)
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
        d = (self.br - self.bl) / self.N / self.degree # point interval length
        for i in range(self.N):
            yp = torch.cat((yp, self.interpolate(input, torch.tensor([self.bl + d * (j + i * self.degree) for j in range(self.degree + 1)], device=self.device), self.sample_values[i * self.degree: (i + 1) * self.degree + 1])), -1)
            # yp = torch.cat((yp, self.interpolate(input, self.sample_points[i * self.degree: (i + 1) * self.degree + 1])), -1) # wrong grad error!
        output = maskbl * ybl + maskbr * ybr + torch.sum(maskother * yp, -1)
        return self.post_process(output.reshape(shape))

    def to(self, *args, **kwargs):
        super(EquidistantTunedValuesIntervalInterpolation, self).to(*args, **kwargs)
        self.sample_points = self.sample_points.to(*args, **kwargs)

    def update(self):
        # update sample points
        self.sample_points = torch.linspace(self.bl.detach(), self.br.detach(), self.sample_num, dtype=self.dtype)
        # update intervals
        for i in range(self.N + 1):
            self.intervals[i] = self.sample_points[i * self.degree].detach()
