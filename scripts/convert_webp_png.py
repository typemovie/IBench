from PIL import Image

def convert_webp_to_png(input_path, output_path):
    with Image.open(input_path) as img:
        img.save(output_path, 'PNG')

# 示例用法
input_path = r'D:\common_tools\IBench\data\images\chineseid\VCG211374260806.webp'  # 输入的 WebP 文件路径
output_path = '../data/images/chineseid/VCG211374260806.png'  # 输出的 PNG 文件路径

convert_webp_to_png(input_path, output_path)