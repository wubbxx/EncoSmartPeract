import torch
from PIL import Image
import torchvision.transforms as T
from odise.modeling.meta_arch.ldm import LdmFeatureExtractor
import torchvision.transforms as T

def main():
    image_path = "/home/wubinxu/peract/try.png"  # 替换为实际的图片路径
    image = Image.open(image_path).convert("RGB")

    # 将图片转换为 tensor (gt_rgb)
    transform = T.Compose([
        T.Resize((128, 128)),  # 根据需要调整大小
        T.ToTensor()           # 转换为 tensor
    ])

    gt_rgb = transform(image).unsqueeze(0)  # 添加 batch 维度，得到 (1, 3, 128, 128) 的 tensor
    gt_rgb = gt_rgb.permute(0, 2, 3, 1)  # 将 (batch_size, channels, height, width) 转换回 (batch_size, height, width, channels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gt_rgb = gt_rgb.to(device)  # 将数据移动到 GPU

    diffusion_extractor = LdmFeatureExtractor(
                    encoder_block_indices=(5, 7),
                    unet_block_indices=(2, 5, 8, 11),
                    decoder_block_indices=(2, 5),
                    steps=(0,),
                    captioner=None,
                )
    diffusion_preprocess = T.Resize(512, antialias=True)
    batched_input = {'img': diffusion_preprocess(gt_rgb.permute(0, 3, 1, 2)), 'caption': "picking up an object"}
    feature_list, lang_embed = diffusion_extractor(batched_input) # list of visual features, and 77x768 language embedding
    used_feature_idx = -1  
    gt_embed = feature_list[used_feature_idx]

    print("Extracted feature shape:", gt_embed.shape)
    print(gt_embed)

# 确保脚本被直接运行时调用 main()
if __name__ == "__main__":
    main()
