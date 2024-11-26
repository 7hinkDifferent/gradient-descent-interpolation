'''
how to we backpropagate the gradient?
can we learn a specific activation for specific task and model (layer)?
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import scipy
import argparse

import src
import src.registry as registry
from src.seed import seed_everything

epsilon = 1e-8

def test_fitting(model, objective_func, logging_dir, xmin=-10, xmax=10, step=0.001, device="cpu"):
    """test metric
    - abs error accumulate
    - abs error average
    - max abs error
    - abs error distribution
    - relative error accumulate
    - relative error average
    - max relative error
    - relative error distribution
    - time used

    """
    model.eval()
    logging_dict = {
        "metric":{ # scipy.integrate.trapz
            "abs_error_acc": 0,
            "abs_error_avg": 0,
            "max_abs_error": 0,
            "relative_error_acc": 0,
            "relative_error_avg": 0,
            "max_relative_error": 0,
        },
        "intervals": model.intervals,
        "params": {name: param for name, param in model.named_parameters()},
    }

    logging_dir = os.path.join(logging_dir, "fitting")
    os.makedirs(logging_dir, exist_ok=True)

    # maximum step is 0.001
    step = min(step, (xmax - xmin) / 1000)

    # sample test
    with torch.no_grad():
        x = torch.arange(xmin, xmax, step).to(device)
        sample_num = x.shape[0]
        # time measurement
        start_time = time.time()
        y = model(x)
        model_time = time.time() - start_time
        ref_y = objective_func(x)
        ref_time = time.time() - model_time - start_time
    
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        ref_y = ref_y.detach().cpu().numpy()

        # plot intervals and inner points
        intervals = model.intervals
        pivot = model(intervals)
        intervals = intervals.cpu().detach().numpy()
        pivot = pivot.cpu().detach().numpy()

        sample_points = model.sample_points.cpu().detach().numpy()
        sample_values = model.sample_values.cpu().detach().numpy()

    # error
    abs_error = abs(y - ref_y)
    abs_error_max_idx = np.argmax(abs_error)
    relative_error = abs((y - ref_y) / (ref_y + epsilon))
    relative_error_max_idx = np.argmax(relative_error)

    # metric - integrate
    logging_dict["metric"]["abs_error_acc"] = scipy.integrate.simps(abs(y - ref_y), dx=step)
    logging_dict["metric"]["abs_error_avg"] = logging_dict["metric"]["abs_error_acc"] / (xmax - xmin)
    logging_dict["metric"]["max_abs_error"] = x[abs_error_max_idx], y[abs_error_max_idx], ref_y[abs_error_max_idx], abs_error[abs_error_max_idx]
    logging_dict["metric"]["relative_error_acc"] = scipy.integrate.simps(abs((y - ref_y) / (ref_y + epsilon)), dx=step)
    logging_dict["metric"]["relative_error_avg"] = logging_dict["metric"]["relative_error_acc"] / (xmax - xmin)
    logging_dict["metric"]["max_relative_error"] = x[relative_error_max_idx], y[relative_error_max_idx], ref_y[relative_error_max_idx], relative_error[relative_error_max_idx]
    logging_dict["metric"]["model_time"] = model_time
    logging_dict["metric"]["ref_time"] = ref_time

    print(logging_dict)

    # logging
    with open(os.path.join(logging_dir, "metric.txt"), "w") as f:
        for key, value in logging_dict.items():
            f.writelines(f"{key}: {value}\n")

    # plotting
    plt.title("fitting curve")
    plt.plot(x, ref_y, label="ref")
    plt.plot(x, y, label="train")
    plt.plot(intervals, pivot, "o", label="intervals", color="red")
    plt.plot(sample_points, sample_values, "x", label="samples", color="green")
    plt.legend()
    plt.savefig(os.path.join(logging_dir, "fitting.png"))
    plt.clf()

    plt.title("absoulte error distribution")
    plt.plot(x, abs_error)
    plt.savefig(os.path.join(logging_dir, "abs_error.png"))
    plt.clf()

    plt.title("relative error distribution")
    plt.plot(x, relative_error)
    plt.savefig(os.path.join(logging_dir, "relative_error.png"))
    plt.clf()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    # model
    parser.add_argument("--model", choices=["equidistant", "equidistant_tuned_values",
                                             "adaptive", "adaptive_tuned_values"], default="equidistant")
    parser.add_argument("--path", type=str, default="./logs/exp/model.pth")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--bl", type=float, default=-10)
    parser.add_argument("--br", type=float, default=10)
    parser.add_argument("--sl", type=float, default=0)
    parser.add_argument("--sr", type=float, default=0)
    parser.add_argument("--min_val", type=float, default=None)
    parser.add_argument("--max_val", type=float, default=None)
    # fit-test
    parser.add_argument("--fit-test", action="store_true", default=False)
    parser.add_argument("--objective_func", choices=["sigmoid", "tanh", "relu"], default="sigmoid")
    parser.add_argument("--xmin", type=float, default=-10)
    parser.add_argument("--xmax", type=float, default=10)
    parser.add_argument("--step", type=float, default=0.001)
    # logging
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--logging_root", type=str, default="./test")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    logging_dir = os.path.join(args.logging_root, args.exp_name)
    interpolation = registry.interpolator.from_pretrained(name=args.model, path=args.path, degree=args.degree, N=args.N, 
                                                      bl=args.bl, br=args.br, sl=args.sl, sr=args.sr, 
                                                      min_val=args.min_val, max_val=args.max_val, 
                                                      logging_dir=logging_dir, device=args.device, freeze=True)
    interpolation.to(args.device)
    objective_func = registry.objective_function.build(args.objective_func)
    test_fitting(interpolation, objective_func=objective_func, logging_dir=logging_dir, 
                 xmin=args.xmin, xmax=args.xmax, step=args.step, device=args.device)