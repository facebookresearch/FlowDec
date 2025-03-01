# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import contextlib
from argparse import ArgumentParser
import os
import torch
import torchaudio
from torchaudio import load, save
from tqdm import tqdm
import hydra

from flowdec.model import EnhancementModel


def main():
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(config_path="config/", version_base="1.3")

    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="The checkpoint to load the model from.")
    parser.add_argument("--files", type=str, required=True, help="Input directory or filelist containing *.wav files.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory to store enhanced files. (Will be created if needed)")
    parser.add_argument("--N", type=int, required=True, help="Number of time discretization points. Equal to NFE when using Euler. Use 6 (with --solver midpoint) to match the main model from the paper.")
    parser.add_argument("--single-file", action='store_true', help="Pass to treat `--files` as a path to a single file to enhance")
    parser.add_argument("--exclude-files-matching", type=str, required=False, help="Exclude files from the input list/folder whose basenames contain this string.")

    # Score model only
    parser.add_argument("--predictor", type=str, default="reverse_diffusion", help="SCORE ONLY: Predictor type. 'reverse_diffusion' by default",
                        choices=["euler_maruyama", "reverse_diffusion"])
    parser.add_argument("--corrector", type=str, default="ald", help="SCORE ONLY: Corrector type. 'ald' by default",
                        choices=["ald", "none"])
    parser.add_argument("--snr", type=float, default=0.5, help="SCORE ONLY: SNR for corrector (if applicable). 0.5 by default.")

    # Flow model only
    parser.add_argument("--solver", type=str, default="midpoint", help="FLOW ONLY: solver type. 'midpoint' by default")

    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--ema", type=bool, default=True)
    parser.add_argument("--skip-existing", type=bool, default=True)

    parser.add_argument("--i-min", type=int, default=None)
    parser.add_argument("--i-max", type=int, default=None)
    parser.add_argument("--rtf", action='store_true', help="Pass to time and calculate RTF for each file")
    args = parser.parse_args()

    checkpoint_file = args.ckpt
    files = args.files
    target_dir = args.outdir

    enhance_kwargs = dict(
        N=args.N,
        # Flow stuff
        solver=args.solver,
        # Score stuff
        predictor=args.predictor, corrector=args.corrector, snr=args.snr,
    )
    print(f"Enhance kwargs: {enhance_kwargs}")
    os.makedirs(target_dir, exist_ok=True)

    print("Loading model from checkpoint...")
    model = EnhancementModel.load_from_checkpoint(checkpoint_file, map_location=args.device, ema=args.ema)
    model.eval()
    print("Done loading model.")

    triples_list_txt_file = None
    if os.path.isfile(files):
        filepaths, from_pairs = read_list(files) if not args.single_file else [files]
        if from_pairs:
            triples_list_filename = "triples_list" + (f"_{args.i_min}-{args.i_max}" if args.i_max else "") + ".txt"
            triples_list_txt_file = os.path.join(target_dir, triples_list_filename)
            print(f"Detected pairs list! Will write out triples list to: {triples_list_txt_file}")
            clean_filepaths = [fp[0] for fp in filepaths]
            noisy_filepaths = [fp[1] for fp in filepaths]
    else:
        noisy_filepaths = sorted(glob.glob(f'{files}/*.wav'))

    if args.exclude_files_matching is not None:
        noisy_filepaths = (f for f in noisy_filepaths if not args.exclude_files_matching in f)

    triples_file_ctxmgr = open(triples_list_txt_file, "w") if triples_list_txt_file else contextlib.nullcontext()
    if args.rtf:
        rtf_file = triples_list_txt_file.replace("triples_list", "rtfs").replace(".txt", ".csv")
        print("Will write to RTF file", rtf_file)
        rtf_file_ctxmgr = open(rtf_file, "w")
    else:
        rtf_file_ctxmgr = contextlib.nullcontext()
    with torch.no_grad(), triples_file_ctxmgr as trf, rtf_file_ctxmgr as rtf_f:
        if rtf_f is not None:
            print("path,runtime,filetime,rtf", file=rtf_f)

        tqdm_iter = tqdm(
            noisy_filepaths, unit="file",
            total=(
                args.i_max - args.i_min if args.i_min is not None and args.i_max is not None else len(noisy_filepaths)
            )
        )
        for i, noisy_filepath in enumerate(tqdm_iter):
            tqdm_iter.set_description(os.path.basename(noisy_filepath))
            if args.i_min is not None and i < args.i_min:
                continue
            if args.i_max is not None and i > args.i_max:
                continue

            basename = os.path.basename(noisy_filepath)
            output_filepath = os.path.join(target_dir, basename)
            #import pdb; pdb.set_trace()
            if not os.path.exists(output_filepath) or not args.skip_existing:
                # Load wav, enhance, write
                y, sr = load(noisy_filepath)
                if y.shape[-1] / sr <= 30.0:
                    if sr != model.sampling_rate:
                        print("RESAMPLING from", sr, "to", model.sampling_rate)
                        y = torchaudio.functional.resample(y, sr, model.sampling_rate, lowpass_filter_width=64)
                        sr = model.sampling_rate
                    if args.rtf:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()

                    x_hat = model.enhance(y, **enhance_kwargs)

                    if args.rtf:
                        end.record()
                        torch.cuda.synchronize()
                        runtime = start.elapsed_time(end) / 1000.0  # ms -> s
                        filetime = (y.shape[-1] / sr)
                        rtf = runtime / filetime
                        print(runtime, filetime, "-> rtf =", rtf)
                        if rtf_f is not None:
                            rtf_line=f"{output_filepath},{runtime:.5f},{filetime:.5f},{rtf:.5f}"
                            print(rtf_line, file=rtf_f)
                    save(output_filepath, x_hat.cpu(), sr)
                else:
                    print("Skipping file due to length:", noisy_filepath)

            if trf is not None:
                triples_line = f"{clean_filepaths[i]} ---> {noisy_filepaths[i]} ---> {output_filepath}"
                print(triples_line, file=trf)


def read_list(listfile):
    filenames = []
    from_pairs = False
    with open(listfile, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                if ' ---> ' in line:
                    # assume pairs list, use only second filename
                    from_pairs = True
                    filenames.append(line.split(' ---> '))
                elif ',' in line:
                    # assume pairs list, use only second filename
                    from_pairs = True
                    filenames.append(line.split(','))
                else:
                    assert not from_pairs, "Inconsistent file list format with and without pairs detected!"
                    filenames.append(line)
    return filenames, from_pairs


if __name__ == '__main__':
    main()
