import sys
import json
import numpy as np
from tqdm import tqdm
from os.path import join, isfile, abspath
from os import listdir, makedirs
from argparse import ArgumentParser
from librosa import resample, power_to_db
from librosa import load as librosa_load
from librosa.feature import melspectrogram
from librosa.util import normalize as librosa_normalize


def compute_spectrogram(contexts, fs):
    hop_size = 160  # Samples
    window_length = 160 * 3  # Samplesself.lr
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 7600.0, 100
    pad_width = (
        ((0, 0), (0, window_length - hop_size))
        if contexts.ndim > 1
        else ((0, window_length - hop_size))
    )
    mel_spectrograms = melspectrogram(
        y=np.pad(contexts, pad_width),
        sr=fs,
        n_fft=window_length,
        hop_length=hop_size,
        win_length=window_length,
        center=False,
        n_mels=num_mel_bins,
        fmin=lower_edge_hertz,
        fmax=upper_edge_hertz,
    )
    # mel_spectrograms = power_to_db(mel_spectrograms, ref=np.max)
    return mel_spectrograms


def get_filenames(source_dir):
    return [
        f
        for f in listdir(source_dir)
        if isfile(join(source_dir, f)) and f.endswith(".wav")
    ]


def prepare_spectrogram(
    context, full_sample_rate, context_sample_rate, context_length_source_fs, channels
):
    spectrogram = []
    if 2 in channels:
        context = resample(
            context[-(context_length_source_fs // 4) :],
            orig_sr=full_sample_rate,
            target_sr=context_sample_rate,
        )
        context = librosa_normalize(context)  # normalize to unity peak level
        spectrogram_2s = compute_spectrogram(
            resample(
                context[-(context_length_source_fs // 4) :],
                orig_sr=full_sample_rate,
                target_sr=context_sample_rate,
            ),
            context_sample_rate,
        )
        spectrogram.append(spectrogram_2s)
    if 4 in channels:
        spectrogram_4s = compute_spectrogram(
            resample(
                context[-(context_length_source_fs // 2) :],
                orig_sr=full_sample_rate,
                target_sr=context_sample_rate // 2,
            ),
            context_sample_rate // 2,
        )
        spectrogram.append(spectrogram_4s)
    if 8 in channels:
        spectrogram_8s = compute_spectrogram(
            resample(
                context, rig_sr=full_sample_rate, target_sr=context_sample_rate // 4
            ),
            context_sample_rate // 4,
        )
        spectrogram.append(spectrogram_8s)

    return np.stack(spectrogram, axis=0)


def main(
    mode: str,
    source_dir: str,
    target_dir: str,
    full_sample_rate: int,
    packet_size: int,
    extra_dim: int,
    context_length: int,
    packet_skip_factor: int,
    prev_packets: int,
):
    filenames = get_filenames(source_dir)

    pbar = tqdm(filenames, ascii=True)
    for filename in pbar:
        pbar.set_description(f"Processing {filename}")
        current_file_dirname = join(target_dir, f"{filename.split('.')[0]}")
        audio, _ = librosa_load(
            join(source_dir, filename), sr=full_sample_rate, mono=True, dtype=np.float32
        )

        context_length_source_fs = context_length * full_sample_rate

        if mode == "verma":
            left_bound = context_length_source_fs
        elif mode == "parcnet":
            sys.path.insert(0, abspath("."))
            from model import ARModel

            ar = ARModel(p=128, diagonal_load=0.001)
            left_bound = packet_size * 10

        packet_idx = 0
        for start_idx in range(
            left_bound, len(audio), packet_size * packet_skip_factor
        ):
            end_idx = start_idx + packet_size
            if end_idx >= len(audio) or end_idx + full_sample_rate >= len(audio):
                break

            sample = {}

            sample["lost_packet"] = audio[start_idx : end_idx + extra_dim]
            sample["last_packet"] = audio[
                start_idx - packet_size * prev_packets : start_idx
            ]
            sample["start_sample_idx"] = start_idx

            if mode == "parcnet":
                sample["linear_prediction"] = ar.predict(
                    audio[start_idx - packet_size * 10 : start_idx],
                    (packet_size + extra_dim),
                )
                sample["last_packet"] = np.concatenate(
                    (sample["last_packet"], np.zeros(packet_size + extra_dim))
                )
                sample["last_packet"] = sample["last_packet"][np.newaxis, :]

            makedirs(current_file_dirname, exist_ok=True)
            np.savez(join(current_file_dirname, f"{packet_idx}.npz"), **sample)
            packet_idx += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--source-dir", type=str, default="../dataset/BachCello")
    parser.add_argument(
        "-t", "--target-dir", type=str, default="../dataset/BachCello_prepared"
    )
    parser.add_argument("-sr", "--sample-rate", type=int, default=16000)
    parser.add_argument("-ps", "--packet-size", type=int, default=128)
    parser.add_argument("-ed", "--extra-dim", type=int, default=80)
    parser.add_argument("-cl", "--context-length", type=int, default=8)
    parser.add_argument("-psf", "--packet-skip-factor", type=int, default=100)
    parser.add_argument("-pp", "--prev-packets", type=int, default=7)
    parser.add_argument("-m", "--mode", type=str, default="parcnet")
    args = parser.parse_args()
    context_channels = set([int(c) for c in args.context_channels])

    if args.mode not in ["parcnet"]:
        raise ValueError(
            f"Mode {args.mode} not supported. Choose between 'parcnet' and 'verma'."
        )

    makedirs(args.target_dir, exist_ok=True)
    with open(join(args.target_dir, "args.json"), "w") as dump:
        json.dump(args.__dict__, dump)

    main(
        args.mode,
        args.source_dir,
        args.target_dir,
        args.sample_rate,
        args.packet_size,
        args.extra_dim,
        args.context_length,
        args.packet_skip_factor,
        args.prev_packets,
    )