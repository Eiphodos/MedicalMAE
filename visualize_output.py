import argparse
import torch
import numpy as np
import os

from pathlib import Path
from PIL import Image
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from modeling_pretraining import MAE
from vit_pytorch import ViT
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training script', add_help=False)

    parser.add_argument('--ckpt_path', type=str,
                        help='Path to the checkpoint to use')
    parser.add_argument('--img_path', type=str,
                        help='Path to the image to feed through the network')
    parser.add_argument('--output_dir', type=str,
                        help='path where to save, empty for no saving')

    parser.add_argument('--mask_ratio', default=0.75, type=int,
                        help='Ratio of the total number of patches to be masked')
    parser.add_argument('--image_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size for backbone patch embedding')
    parser.add_argument('--num_channels', default=1, type=int,
                        help='Number of channels for each input')
    parser.add_argument('--encoder_dim', default=1024, type=int,
                        help='Token dimension for the transformer encoder')
    parser.add_argument('--encoder_depth', default=12, type=int,
                        help='Depth of the transformer encoder')
    parser.add_argument('--encoder_heads', default=12, type=int,
                        help='Number of heads for the transformer encoder')
    parser.add_argument('--decoder_dim', default=512, type=int,
                        help='Token dimension for the transformer decoder')
    parser.add_argument('--decoder_depth', default=8, type=int,
                        help='Depth of the transformer decoder')
    parser.add_argument('--decoder_heads', default=6, type=int,
                        help='Number of heads for the transformer decoder')

    return parser.parse_args()


def get_model(args):
    enc = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=1000, # Does not matter, unused
        dim=args.encoder_dim,
        depth=args.encoder_depth,
        heads=args.encoder_heads,
        channels=args.num_channels,
        mlp_dim=2048 # Does not matter, unused
    )

    mae = MAE(
        encoder=enc,
        masking_ratio=args.mask_ratio,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads
    )

    return mae


def get_img(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        nimg = np.array(img).astype(np.uint8)
        img = Image.fromarray(nimg)
        return img.convert('L')


def get_transforms(args):
    mean = (0.5)
    std = (0.25)
    t = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    return t


def denormalize_to_np(image_tensor):
    image_tensor = ((image_tensor*0.25) + 0.5)*255
    image = image_tensor.numpy().astype(np.uint8)
    return image


def rearrage_tensor(patch_size, image_size, image_tensor):
    n_patches = image_size // patch_size
    r = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=n_patches, w=n_patches)
    return r(image_tensor).squeeze()


def main(args):
    # Set static seeds
    torch.manual_seed(13)
    np.random.seed(13)
    torch.use_deterministic_algorithms(True)

    model = get_model(args)
    sd = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(sd['model'])

    org_img = get_img(args.img_path)
    out_name = os.path.split(args.img_path)[-1][:-4] + os.path.split(args.ckpt_path)[-1][:-4]

    transforms = get_transforms(args)

    t_img = transforms(org_img).unsqueeze(dim=0)

    patches, unmasked_indices, masked_indices, batch_range, masked_patches, pred_pixel_values = model.forward_with_output(t_img)

    print(patches.shape)
    print(unmasked_indices.shape)
    print(masked_indices.shape)
    print(masked_patches.shape)
    print(pred_pixel_values.shape)

    rpatches = rearrage_tensor(args.patch_size, args.image_size, patches)
    print(rpatches.shape)

    pred_img = patches.clone().detach()
    for i, data in zip(masked_indices.squeeze(), pred_pixel_values.squeeze()):
        pred_img[0][i] = data
    pred_img = rearrage_tensor(args.patch_size, args.image_size, pred_img).detach()

    masked_img = patches.clone().detach()
    for i in masked_indices.squeeze():
        masked_img[0][i] = torch.tensor([255]*256)
    masked_img = rearrage_tensor(args.patch_size, args.image_size, masked_img).detach()

    org_np_img = denormalize_to_np(rpatches)
    org_im_img = Image.fromarray(org_np_img)
    org_out_name = out_name + '_org.png'
    org_im_img.save(os.path.join(args.output_dir, org_out_name))

    pred_np_img = denormalize_to_np(pred_img)
    pred_im_img = Image.fromarray(pred_np_img)
    pred_out_name = out_name + '_pred.png'
    pred_im_img.save(os.path.join(args.output_dir, pred_out_name))

    masked_np_img = denormalize_to_np(masked_img)
    masked_im_img = Image.fromarray(masked_np_img)
    masked_out_name = out_name + '_mask.png'
    masked_im_img.save(os.path.join(args.output_dir, masked_out_name))

if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)