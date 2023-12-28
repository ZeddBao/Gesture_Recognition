import os
import Augmentor
from tqdm import tqdm

# 获得当前工作目录
current_dir = os.getcwd()
AUGMENTED_BUFFER_DIR = os.path.join(current_dir, 'dataset_augmented_buffer/')

def augment_images(data_path):
    # 使用数据增强库，递归地将 dataset_origin/ 目录下每个目录中的图片（注意排除视频）随机旋转、水平翻转、垂直翻转进行采样
    # 最后在每个有图片的目录下新建一个 dataset_augmented/ 目录，将增强后的图片保存在其中
    file_list = os.listdir(data_path)
    augmented_dir = os.path.join(AUGMENTED_BUFFER_DIR, os.path.basename(data_path))
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)
    for file in tqdm(file_list):
        file_path = os.path.join(data_path, file)
        # 如果file_path不是目录，则跳过
        if os.path.isdir(file_path):
            augment_images(file_path)
        elif file.endswith(('.JPG', 'jpg', '.png', '.PNG')):

            # 创建Augmentor管道
            p = Augmentor.Pipeline(
                source_directory=data_path, output_directory=augmented_dir)

            # 定义增强操作
            p.rotate(probability=0.7, max_left_rotation=25,
                     max_right_rotation=25)
            p.flip_left_right(probability=0.5)
            p.flip_top_bottom(probability=0.5)

            # 执行增强并生成增强的图像
            p.sample(8)
        elif file.endswith(('.mp4', 'MP4')):
            # 将视频复制到 dataset_augmented_buffer/ 目录下
            os.system(f'cp {file_path} {augmented_dir}')

if __name__ == '__main__':
    augment_images('dataset_origin/')