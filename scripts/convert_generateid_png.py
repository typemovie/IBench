import os
from PIL import Image


def process_images(input_directory, output_directory):
    """
    遍历输入目录中的所有文件，尝试读取每个文件并将其保存为PNG格式。

    :param input_directory: 输入目录路径
    :param output_directory: 输出目录路径
    """
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_directory):
        # 构造文件的完整路径
        file_path = os.path.join(input_directory, filename)
        # 截取文件名的前50个字符（不包括扩展名）
        truncated_filename = os.path.splitext(filename)[0][:50]

        # 尝试打开文件并保存为PNG格式
        try:
            with Image.open(file_path) as img:
                # 构造输出文件的完整路径，保留原始文件名但扩展名改为png
                output_path = os.path.join(output_directory, truncated_filename + '.png')
                # 保存图片为PNG格式
                img.save(output_path, 'PNG')
                print(f'Successfully processed {filename}')
        except Exception as e:
            print(f'Failed to process {filename}: {e}')


if __name__ == '__main__':
    # 定义输入和输出目录
    config = {
        "input_dir": r'D:\common_tools\IBench\data\images\generateid',
        "output_dir": r'D:\common_tools\IBench\data\images\generateid_name'
    }

    output_dir = config['output_dir']
    input_dir = config['input_dir']
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 调用函数处理图片
    process_images(input_dir, output_dir)
