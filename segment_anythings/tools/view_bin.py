"""View a raw f32 dump (.bin) produced by sam_encoder_runner.

Usage:
    python view_bin.py <file.bin> [NCHW-dims e.g. 1,256,64,64] [--count=100]

Prints element count, NaN/Inf counts, min/max/mean/std, per-channel mean
(when a 4D [1,C,H,W] shape is given), and the first `--count` values
(default 16).
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="View a raw f32 dump (.bin).")
    parser.add_argument("file", help="path to .bin (little-endian f32)")
    parser.add_argument("shape", nargs="?", default=None,
                        help="NCHW dims, e.g. 1,256,64,64")
    parser.add_argument("--count", type=int, default=16,
                        help="print first N values (default 16)")
    args = parser.parse_args()

    shape = None
    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(","))

    a = np.fromfile(args.file, dtype=np.float32)
    print(f"file  : {args.file}")
    print(f"elems : {a.size}")
    print(f"bytes : {a.size * 4}")
    if shape is not None:
        if np.prod(shape) != a.size:
            print(f"WARNING: shape {shape} product != {a.size}, ignoring shape")
        else:
            a = a.reshape(shape)
    print(f"shape : {a.shape}")
    print(f"nan   : {int(np.isnan(a).sum())}")
    print(f"inf   : {int(np.isinf(a).sum())}")
    if a.size:
        print(f"min/max/mean/std : {a.min():.6g} / {a.max():.6g} / "
              f"{a.mean():.6g} / {a.std():.6g}")

    if a.ndim == 4 and a.shape[0] == 1:  # [1,C,H,W]
        ch_mean = a[0].mean(axis=(1, 2))
        ch_std = a[0].std(axis=(1, 2))
        show = min(16, a.shape[1])
        print(f"per-channel mean (first {show} of {a.shape[1]}):")
        print("  " + " ".join(f"{ch_mean[i]:.4g}" for i in range(show)))
        print(f"channel mean min/max : {ch_mean.min():.6g} / {ch_mean.max():.6g}")

    flat = a.ravel()
    n = min(args.count, flat.size)
    print(f"first {n} values : {flat[:n].tolist()}")


if __name__ == "__main__":
    main()
