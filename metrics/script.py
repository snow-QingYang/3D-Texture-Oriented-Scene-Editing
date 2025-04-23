import torch
from PIL import Image
from torchvision import transforms
from clip_metric import ClipSimilarity  # 假设你的类保存在 clip_metric.py 文件中
import argparse

# 预处理 transform（范围[0,1]，不做Resize，让ClipSimilarity自己做）
preprocess = transforms.Compose([
    transforms.ToTensor()
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    return preprocess(image).unsqueeze(0)  # shape: (1, 3, H, W)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClipSimilarity(name=args.clip_model).to(device)

    image_0 = load_image(args.image0).to(device)
    image_1 = load_image(args.image1).to(device)

    # 使用两个 dummy 的 text prompt 来匹配
    text_0 = [args.text0]
    text_1 = [args.text1]

    sim_0, sim_1, sim_direction, sim_image = model(image_0, image_1, text_0, text_1)

    print(f"Similarity of Image 0 and Text 0: {sim_0.item():.4f}")
    print(f"Similarity of Image 1 and Text 1: {sim_1.item():.4f}")
    print(f"Directional Similarity (Δimage vs Δtext): {sim_direction.item():.4f}")
    print(f"Similarity between Image 0 and Image 1: {sim_image.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image0", type=str, required=True, help="Path to first image")
    parser.add_argument("--image1", type=str, required=True, help="Path to second image")
    parser.add_argument("--text0", type=str, default="a photo", help="Text for image 0")
    parser.add_argument("--text1", type=str, default="a photo", help="Text for image 1")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14", help="CLIP model to use")
    args = parser.parse_args()
    main(args)
