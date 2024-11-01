import os
from ldm.models.mytoencoder import AutoencoderKL
from diffusion.model import DTCA_XL_2
import torch
import einops
import yaml
from diffusion import IDDPM
from utils_ import Get_tager_sample_h5npy, video_tensor_to_gif

with open('ldm/autoencoder_kl_32x32x4.yaml', 'r') as file:
        autoencodercfg = yaml.safe_load(file)

auto_kl = AutoencoderKL(autoencodercfg).to("cuda")
auto_kl.load_state_dict(torch.load("pre_pthautoencoder.pth", map_location="cuda"))
model = DTCA_XL_2().to("cuda")
model.load_state_dict(torch.load("end_clearly_latentshift4.pth"),strict=True)


device_id = "cuda"
train_1 = Get_tager_sample_h5npy("output.txt")
train_loader = torch.utils.data.DataLoader(train_1,
                                                batch_size = 2,
                                               pin_memory=True,
                                               num_workers=4,
                                               shuffle=True,
                                               )
_,condition,groundtruth,names = next(iter(train_loader))

diffusion = IDDPM(str(1000), learn_sigma=True, pred_sigma=True,
                            snr=False)
condition= condition.to(device_id)
condition_h = condition.clone()
# flow_x = flow_x.to(device_id)
# flow_y = flow_y.to(device_id)
my_c =einops.rearrange(condition,  "b c f w h -> (b f) c w h")

with torch.no_grad():
        auto_kl.eval()
        my_c = auto_kl(my_c)
        # flow_x1 = auto_kl_ddp(flow_x)
        # flow_y1 = auto_kl_ddp(flow_y)
        z = torch.randn(32, 4, 32, 32, device=device_id,requires_grad=False)
samples = diffusion.p_sample_loop(
            model.forward_with_sample, z.shape, z, clip_denoised=False, progress=True,
            device="cuda",
            model_kwargs=dict(condition = my_c,condition_high_pix=condition_h,flow_x = None,flow_y=None)
            )
samples = samples / 0.3242
groundtruth = einops.rearrange(groundtruth,"b c f w h -> c f (b w ) h").to("cuda")
with torch.no_grad():
    sampled_images = auto_kl(samples, False)  # 32 32 4
    sampled_images = torch.clip(sampled_images, 0, 1)
    # sampled_images = einops.rearrange(sampled_images,"(b f) c w h ->b c f w h",f=16)
    sampled_images = einops.rearrange(sampled_images, "(b f) c w h ->c f (b w) h",f=16)
    video_tensor_to_gif(torch.cat([sampled_images,groundtruth],dim=-1),os.path.join("result/", f"{1}.gif"))








