# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import time


def parse_class_labels(class_labels, num_classes):
    if class_labels is None:
        return list(range(num_classes))
    labels = [int(label.strip()) for label in class_labels.split(",") if label.strip()]
    for label in labels:
        if label < 0 or label >= num_classes:
            raise ValueError(f"class label {label} is outside [0, {num_classes})")
    return labels


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu" 
    #print("device = ", device, flush=True)
    #print(torch.cuda.device_count(), flush=True)

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae_path = args.vae_path or f"stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    class_labels = parse_class_labels(args.class_labels, args.num_classes)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    start_time = time.time()
    print(f"Sampling {len(class_labels)} classes with batch size {args.batch_size}.")

    for batch_start in range(0, len(class_labels), args.batch_size):
        batch_labels = class_labels[batch_start:batch_start + args.batch_size]
        n = len(batch_labels)

        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(batch_labels, device=device)

        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        model_kwargs['interval']        = args.interval
        model_kwargs['max_order']       = args.max_order
        model_kwargs['test_FLOPs']      = args.test_FLOPs
        model_kwargs['mode']            = args.mode
        
        model_kwargs['cluster_num']       = args.cluster_num
        model_kwargs['cluster_method']    = args.cluster_method
        model_kwargs['propagation_ratio'] = args.propagation_ratio
        model_kwargs['k']                 = args.k
        model_kwargs['tet_alpha']         = args.tet_alpha
        model_kwargs['tet_max_iters']     = args.tet_max_iters
        model_kwargs['resca_solver']      = args.resca_solver
        model_kwargs['resca_proxy_method'] = args.resca_proxy_method

        if args.ddim_sample:
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        else:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )

        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        for label, sample in zip(batch_labels, samples):
            save_image(sample, f"{label}.png", normalize=True, value_range=(-1, 1))

        print(f"Saved classes {batch_labels[0]}-{batch_labels[-1]}.")

    print(f"Total Sampling took {time.time() - start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--class-labels", type=str, default=None,
                        help="Comma-separated class ids to sample. Defaults to all classes.")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--ddim-sample", action="store_true", default=False)
    parser.add_argument("--interval", type=int, default=4) 
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--test-FLOPs", action="store_true", default=False)
    parser.add_argument("--mode", type=str, choices=['Taylor', 'ClusCa', 'ResCa'], default='ResCa')

    # ClusCa/ResCa parameters
    parser.add_argument("--cluster-num", type=int, default=16)
    parser.add_argument("--cluster-method", type=str, choices=['kmeans', 'random', 'kmeans++'], default='kmeans')
    parser.add_argument("--propagation-ratio", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=1, help="num of selected fresh tokens per cluster")
    parser.add_argument("--tet-alpha", type=float, default=0.6)
    parser.add_argument("--tet-max-iters", type=int, default=8)
    parser.add_argument("--resca-solver", type=str, choices=['ie', 'it', 'bdf2'], default='ie')
    parser.add_argument("--resca-proxy-method", type=str, choices=['center', 'random', 'center-random'], default='center')

    args = parser.parse_args()
    main(args)
