# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
import torchaudio, torch
from tqdm import tqdm
import argparse

from flowdec.data.feature_extractors import ComplexSTFT, CompressAmplitudesAndScale, InvertibleSequential
from flowdec.util.other import t2n


n_samples = 2500
secs = 2
seed = 302


def rreplace(s, old, new, occurrence=1):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def random_crop_or_pad_pair(x, y, tgtlen):
    # Ensure y is not longer than x on the right-hand side
    y = y[..., :x.shape[-1]]
    
    # Handle the case where the length of x matches the target length
    if x.shape[-1] == tgtlen:
        return x, y
    
    # If x is shorter than the target length, pad both x and y
    elif x.shape[-1] < tgtlen:
        padding = (0, tgtlen - x.shape[-1])
        x_padded = torch.nn.functional.pad(x, padding)
        y_padded = torch.nn.functional.pad(y, padding)
        return x_padded, y_padded
    
    # If x is longer than the target length, crop both x and y
    else:
        start = np.random.randint(0, x.shape[-1] - tgtlen)
        x_cropped = x[..., start:start + tgtlen]
        y_cropped = y[..., start:start + tgtlen]
        return x_cropped, y_cropped


def load48000(path):
    au, fs = torchaudio.load(path)
    if au.shape[0] != 1:
        au = torch.mean(au, dim=0, keepdim=True)
    if fs != 48000:
        au = torchaudio.functional.resample(au, fs, 48000, lowpass_filter_width=256)
        fs = 48000
    return au


def filter_frames(spec_x, spec_y):
    return spec_x, spec_y


def normalize_noisy(x, y):
    normfac = y.abs().max() + 1e-5
    return x/normfac, y/normfac


def get_feats(batch_x, batch_y, *, n_fft, hop_length, alpha, sr, device):
    extractor = InvertibleSequential([
        ComplexSTFT(window_fn='hann', n_fft=n_fft, hop_length=hop_length, sampling_rate=sr),
        CompressAmplitudesAndScale(compression_exponent=alpha, scale_factor=1.0)
    ]).to(device)
    xy_normalized_audios = [
        normalize_noisy(au_x, au_y)
        for (au_x, au_y) in tqdm(zip(batch_x, batch_y), total=len(batch_x))
    ]
    batch_xy_feats = [
        filter_frames(extractor(au_x.to(device)), extractor(au_y.to(device)))
        for (au_x, au_y) in xy_normalized_audios
    ]
    batch_x_feats = [xy[0][0] for xy in batch_xy_feats]
    batch_y_feats = [xy[1][0] for xy in batch_xy_feats]
    return batch_x_feats, batch_y_feats, extractor


def abs_quantile(x, q):
    return np.quantile(np.abs(x).reshape(-1), q)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", type=str, required=True)
    parser.add_argument("--delim", type=str, default=' ---> ')

    parser.add_argument("--alpha", type=float, required=True, help='Amplitude compression exponent')
    parser.add_argument("--nfft", type=int, required=True)
    parser.add_argument("--hop", type=int, required=True)

    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--n-samples", type=int, default=2500)
    parser.add_argument("--sample-duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=302)
    parser.add_argument(
        "--qx", type=float, default=0.997,
        help="Quantile of the global distribution of clean audio x that we will rescale to 1.0 (via *beta). 0.997 by default."
    )
    parser.add_argument(
        "--qrmse", type=float, default=0.997,
        help="Quantile of the RMSEs of clean x vs coded y that can be used as a reasonable default sigma_y. 0.997 by default."
    )
    parser.add_argument("--per-band", action='store_true', help="Pass to calculate frequency-dependent sigma_y. The --qrmse value is then used for each frequency band separately to determine a per-band sigma_y.")
    parser.add_argument("--outfile-suffix", type=str, required=False, help="Suffix to append to the results filename")
    parser.add_argument("--overwrite", action='store_true', help="If passed, will recalculate and overwrite results file even if it exists")
    parser.add_argument("--device", type=int, default=0, help="Index of CUDA device to use for feature extraction")
    args = parser.parse_args()
    print(args, file=sys.stderr)

    outfile_suffix = ''
    outfile_suffix += f'_n{args.n_samples}' if args.n_samples != 2500 else ''
    outfile_suffix += '_perband' if args.per_band else ''
    outfile_suffix += f'_{args.outfile_suffix}' if args.outfile_suffix is not None else ''
    outfile_path = os.path.join(
        os.path.dirname(args.pairs_file),
        f"flowdec_autoparams_nfft{args.nfft}_hop{args.hop}_alpha{args.alpha}_seed{args.seed}{outfile_suffix}.txt"
    )

    if os.path.isfile(outfile_path) and not args.overwrite:
        print("Output file exists, printing its contents:", file=sys.stderr)
        with open(outfile_path, 'r') as f:
            for line in f:
                print(line.rstrip("\n"))
        sys.exit(0)

    print("Running...", file=sys.stderr)
    np.random.seed(args.seed)
    with open(args.pairs_file, 'r') as f:
        pairs = [l.strip() for l in f.readlines()]
        batch_pairs = [l.split(args.delim) for l in np.random.choice(pairs, args.n_samples, replace=False)]
        batch_x_files = [p[0] for p in batch_pairs]
        batch_y_files = [p[1] for p in batch_pairs]
    assert len(batch_x_files) == len(batch_x_files)
    assert len(batch_x_files) == args.n_samples
    batch_xy = [
        random_crop_or_pad_pair(load48000(fx), load48000(fy), 48000*secs)
        for fx, fy in tqdm(zip(batch_x_files, batch_y_files), total=len(batch_x_files))
    ]
    batch_x = [bxy[0] for bxy in batch_xy]
    batch_y = [bxy[1] for bxy in batch_xy]

    model_params = dict(
        alpha=args.alpha, n_fft=args.nfft, hop_length=args.hop, sr=args.sr, device=f'cuda:{args.device}')
    batch_x_feats, batch_y_feats, _ = get_feats(batch_x, batch_y, **model_params)
    all_bins_x = torch.cat([f.reshape(-1) for f in batch_x_feats])
    all_bins_y = torch.cat([f.reshape(-1) for f in batch_y_feats])

    abs_quantile_x = abs_quantile(t2n(all_bins_x), args.qx)
    abx_qnormal = all_bins_x / abs_quantile_x
    aby_qnormal = all_bins_y / abs_quantile_x
    spec_diffs = [(afy_ - afx_) for (afy_, afx_) in zip(batch_y_feats, batch_x_feats)]

    if args.per_band:
        rmses_per_band = np.array([
            torch.linalg.norm(diff.squeeze(), ord=2, dim=-1).cpu().numpy() / diff.shape[-2]**0.5
            for diff in spec_diffs
        ])
        rmse_quantile_per_band = np.quantile(rmses_per_band, args.qrmse, axis=0)
        per_band_outfile_path = rreplace(outfile_path, '.txt', 'sigy_perband.npy')
        print(f"Writing resulting per_band sigma_y to {per_band_outfile_path}", file=sys.stderr)
        np.save(per_band_outfile_path, rmse_quantile_per_band / 3)
    else:
        rmses = np.array([
            torch.linalg.norm(diff.reshape(-1), ord=2).cpu().item() / diff.numel()**0.5
            for diff in spec_diffs
        ])
        rmse_quantile = np.quantile(rmses, args.qrmse)

    print(f"Writing results to {outfile_path}", file=sys.stderr)
    with open(outfile_path, "w") as f:
        for stream in (sys.stdout, f):
            print(f"Input pairs file: {os.path.abspath(args.pairs_file)}", file=stream)
            print(f"Args: {args}", file=stream)
            print(f"=== Results ===", file=stream)
            print(f"   \tq{args.qx}( |x|  ) = {abs_quantile_x:.3f}, max( |x|  ) = {all_bins_x.abs().max():.3f}", file=stream)

            if args.per_band:
                print(f"-->\tbeta={1/abs_quantile_x:.2f}, sigma_y=<written to {per_band_outfile_path}>", file=stream)
            else:
                print(f"   \tq{args.qrmse}( RMSE ) = {rmse_quantile:.3f}, max( RMSE ) = {np.max(rmses):.3f}", file=stream)
                print(f"-->\tbeta={1/abs_quantile_x:.2f}, sigma_y={rmse_quantile / 3:.2f}", file=stream)
