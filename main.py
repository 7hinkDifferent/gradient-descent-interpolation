import argparse, os
from src.seed import seed_everything
import torch
import src
import src.registry as registry

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu") # TODO: currently not supported cuda training
    # model
    parser.add_argument("--model", choices=["equidistant", "equidistant_tuned_values",
                                             "adaptive", "adaptive_tuned_values"], default="equidistant")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--bl", type=float, default=-10)
    parser.add_argument("--br", type=float, default=10)
    parser.add_argument("--sl", type=float, default=0)
    parser.add_argument("--sr", type=float, default=0)
    parser.add_argument("--min_val", type=float, default=None)
    parser.add_argument("--max_val", type=float, default=None)
    # fit-train
    parser.add_argument("--objective_func", choices=["sigmoid", "tanh", "relu"], default="sigmoid")
    parser.add_argument("--dist", choices=["N", "U"], default="N", help="distribution type for data generator")
    parser.add_argument("--dist_param", nargs="+", type=int, default=(0,5), help="(mean, std) for N, (a, b) for U")
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--optim", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss_fn", choices=["mse", "l1"], default="mse")
    parser.add_argument("--xmin", type=float, default=-10)
    parser.add_argument("--xmax", type=float, default=10)
    parser.add_argument("--step", type=float, default=0.001)
    # logging
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--logging_root", type=str, default="./logs")

    args = parser.parse_args()
    args.dist_param = tuple(args.dist_param)
    return args

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    logging_dir = os.path.join(args.logging_root, args.exp_name)
    interpolation = registry.interpolator.build(name=args.model, degree=args.degree, N=args.N, 
                                                      bl=args.bl, br=args.br, sl=args.sl, sr=args.sr, 
                                                      min_val=args.min_val, max_val=args.max_val, 
                                                      logging_dir=logging_dir, device=args.device)
    interpolation.to(args.device)
    # interpolation.set_freeze(["bl", "br"])
    # interpolation.set_freeze(["sl", "sr"])
    print(interpolation)
    for n, p in interpolation.named_parameters():
        print(n, p, p.grad)
    objective_func = registry.objective_function.build(args.objective_func)
    generator = {
        "N": lambda batch: torch.randn((batch)) * args.dist_param[1] + args.dist_param[0],
        "U": lambda batch: torch.rand((batch)) * (args.dist_param[1] - args.dist_param[0]) + args.dist_param[0],
    }
    generator = generator[args.dist]
    optimizer_class = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    optimizer_class = optimizer_class[args.optim]
    criterion_class = {
        "mse": torch.nn.MSELoss,
        "l1": torch.nn.L1Loss,
    }
    criterion_class = criterion_class[args.loss_fn]
    # fit
    interpolation.fit(objective_func, generator, optimizer_class, criterion_class, 
                      epoch=args.epoch, batch=args.batch, lr=args.lr, xmin=args.xmin, xmax=args.xmax, step=args.step)
    params = interpolation.state_dict()
    torch.save(params, os.path.join(logging_dir, "model.pth"))
    print("model saved: ", os.path.join(logging_dir, "model.pth"))
    for n, p in interpolation.named_parameters():
        print(n, p, p.grad)