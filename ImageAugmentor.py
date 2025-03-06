import cv2
import os
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm
import glob

class ImageAugmentor:
    def __init__(self, root_input_dir, output_dir="augmented_images"):
        self.root_input_dir = root_input_dir
        self.output_dir = output_dir
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.class_stats = {}
        self.setup_directory_structure()

    def rotate_90(self, image):
        """Xoay 90 độ ngẫu nhiên"""
        transform = A.Compose([
            A.RandomRotate90(p=1.0)
        ])
        return transform(image=image)['image']

    def transpose(self, image):
        """Chuyển vị ảnh"""
        transform = A.Compose([
            A.Transpose(p=1.0)
        ])
        return transform(image=image)['image']

    def add_gaussian_noise(self, image):
        """Thêm nhiễu Gaussian"""
        transform = A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
        ])
        return transform(image=image)['image']

    def apply_blur_effects(self, image):
        """Áp dụng các hiệu ứng blur khác nhau"""
        transform = A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0)
            ], p=1.0)
        ])
        return transform(image=image)['image']

    def geometric_distortion(self, image):
        """Biến dạng hình học"""
        transform = A.Compose([
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.2, p=1.0),
                A.GridDistortion(distort_limit=0.2, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=1.0)
        ])
        return transform(image=image)['image']

    def enhance_contrast_color(self, image):
        """Điều chỉnh độ tương phản và màu sắc nâng cao"""
        transform = A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        ])
        return transform(image=image)['image']

    def rotate_10(self, image):
        """Xoay tối đa 10 độ"""
        transform = A.Compose([
            A.Rotate(limit=10, p=1.0)
        ])
        return transform(image=image)['image']

    def adjust_lighting(self, image):
        """Điều chỉnh lighting với giới hạn 0.2"""
        transform = A.Compose([
            A.RandomBrightness(limit=0.2, p=1.0)
        ])
        return transform(image=image)['image']

    def apply_warp(self, image):
        """Áp dụng warp với giới hạn 0.2"""
        transform = A.Compose([
            A.GridDistortion(distort_limit=0.2, p=1.0)
        ])
        return transform(image=image)['image']

    def apply_zoom(self, image):
        """Áp dụng zoom với hệ số 1.1"""
        transform = A.Compose([
            A.Zoom(scale_range=(1.0, 1.1), p=1.0)
        ])
        return transform(image=image)['image']

    def horizontal_flip(self, image):
        """Lật ngang với xác suất 50%"""
        transform = A.Compose([
            A.HorizontalFlip(p=0.5)
        ])
        return transform(image=image)['image']

    def adjust_brightness_contrast_saturation(self, image):
        """Điều chỉnh độ sáng, tương phản và bão hòa"""
        transform = A.Compose([
            A.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.85, 1.15),
                saturation=(0.95, 1.05),
                p=1.0
            )
        ])
        return transform(image=image)['image']

    def combined_augmentation(self, image):
        """Kết hợp nhiều phương pháp augmentation"""
        transform = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=10, p=0.5)
            ], p=1.0),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=7, p=0.5)
            ], p=0.5),
            
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.2, p=0.5),
                A.GridDistortion(distort_limit=0.2, p=0.5)
            ], p=0.5),
            
            A.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.85, 1.15),
                saturation=(0.95, 1.05),
                p=0.8
            ),
            
            A.HorizontalFlip(p=0.5),
            
            A.OneOf([
                A.Zoom(scale_range=(1.0, 1.1), p=0.5),
                A.GridDistortion(distort_limit=0.2, p=0.5)
            ], p=0.5)
        ])
        return transform(image=image)['image']

    def process_single_image(self, image_path, class_name, num_samples=3):
        """Xử lý một ảnh với tất cả các phương pháp"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh từ {image_path}")

            base_filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(base_filename)[0]
            
            results = {'original': image, 'paths': {}}

            # Lưu ảnh gốc
            original_path = self.save_image(image, class_name, 'original', base_filename)
            results['paths']['original'] = original_path

            # Dictionary các phương pháp augmentation
            augmentation_methods = {
                'rotate_90': self.rotate_90,
                'transpose': self.transpose,
                'gaussian_noise': self.add_gaussian_noise,
                'blur_effects': self.apply_blur_effects,
                'geometric': self.geometric_distortion,
                'contrast_color': self.enhance_contrast_color,
                'rotate_10': self.rotate_10,
                'lighting': self.adjust_lighting,
                'warp': self.apply_warp,
                'zoom': self.apply_zoom,
                'horizontal_flip': self.horizontal_flip,
                'brightness_contrast_sat': self.adjust_brightness_contrast_saturation,
                'combined': self.combined_augmentation
            }

            # Áp dụng từng phương pháp
            for method_name, method_func in augmentation_methods.items():
                try:
                    augmented = method_func(image)
                    filename = f"{name_without_ext}_{method_name}.jpg"
                    path = self.save_image(augmented, class_name, 'augmented', filename)
                    results[method_name] = augmented
                    results['paths'][method_name] = path
                except Exception as e:
                    print(f"Lỗi khi áp dụng {method_name}: {str(e)}")

            return results

        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
            return None

    # Thêm các phương thức khác của class (setup_directory_structure, save_image, etc.)
    def setup_directory_structure(self):
        """Tạo cấu trúc thư mục output tương ứng với input"""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        class_folders = [d for d in os.listdir(self.root_input_dir) 
                        if os.path.isdir(os.path.join(self.root_input_dir, d))]

        for class_folder in class_folders:
            for method in ['original', 'augmented']:
                os.makedirs(os.path.join(self.output_dir, class_folder, method), exist_ok=True)

    def save_image(self, image, class_name, method, filename):
        """Lưu ảnh với tên file và thư mục phù hợp"""
        output_path = os.path.join(self.output_dir, class_name, method, filename)
        cv2.imwrite(output_path, image)
        return output_path

    def process_all_folders(self, num_samples=3):
        """Xử lý tất cả ảnh trong tất cả các folder"""
        class_folders = [d for d in os.listdir(self.root_input_dir) 
                        if os.path.isdir(os.path.join(self.root_input_dir, d))]
        all_results = {}
        total_processed = 0

        print(f"Tìm thấy {len(class_folders)} folder lớp")
        
        for class_folder in class_folders:
            folder_path = os.path.join(self.root_input_dir, class_folder)
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(self.supported_formats)]
            
            print(f"\nXử lý folder {class_folder}: {len(image_files)} ảnh")
            folder_results = {}
            
            for image_file in tqdm(image_files, desc=f"Xử lý {class_folder}"):
                image_path = os.path.join(folder_path, image_file)
                result = self.process_single_image(image_path, class_folder, num_samples)
                if result:
                    folder_results[image_path] = result
                    total_processed += 1
            
            all_results[class_folder] = folder_results
            self.class_stats[class_folder] = len(folder_results)

        print(f"\nĐã xử lý tổng cộng {total_processed} ảnh")
        self.print_statistics()
        return all_results

    def print_statistics(self):
        """In thống kê về số lượng ảnh đã xử lý"""
        print("\nThống kê số lượng ảnh:")
        print("-" * 40)
        for class_name, count in self.class_stats.items():
            print(f"{class_name}: {count} ảnh")
        print("-" * 40)
        print(f"Tổng số: {sum(self.class_stats.values())} ảnh")

    def __init__(self, input_dir, output_dir="augmented_images"):
        """
        Khởi tạo ImageAugmentor
        :param input_dir: Thư mục chứa ảnh đầu vào
        :param output_dir: Thư mục lưu ảnh augmented
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.create_output_dirs()

    def create_output_dirs(self):
        """Tạo các thư mục cần thiết để lưu ảnh"""
        subdirs = ['method1', 'method2', 'method3', 'denoised', 'original']
        for subdir in subdirs:
            path = os.path.join(self.output_dir, subdir)
            os.makedirs(path, exist_ok=True)

    def get_image_files(self):
        """Lấy danh sách tất cả các file ảnh trong thư mục input"""
        image_files = []
        for format in self.supported_formats:
            image_files.extend(glob.glob(os.path.join(self.input_dir, f'*{format}')))
            image_files.extend(glob.glob(os.path.join(self.input_dir, f'*{format.upper()}')))
        return sorted(image_files)

    def denoise_image(self, image):
        """
        Áp dụng các phương pháp loại bỏ nhiễu
        """
        try:
            # Phương pháp 1: Non-local Means Denoising
            denoised_1 = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Phương pháp 2: Bilateral Filter
            denoised_2 = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Phương pháp 3: Median Blur
            denoised_3 = cv2.medianBlur(image, 5)
            
            return {
                'nlm': denoised_1,
                'bilateral': denoised_2,
                'median': denoised_3
            }
        except Exception as e:
            print(f"Lỗi khi denoise ảnh: {str(e)}")
            return None

    def rotate_90(image):
        """Xoay 90 độ ngẫu nhiên"""
        transform = A.Compose([
            A.RandomRotate90(p=1.0)
        ])
        return transform(image=image)['image']

    def transpose(image):
        """Chuyển vị ảnh"""
        transform = A.Compose([
            A.Transpose(p=1.0)
        ])
        return transform(image=image)['image']

    def add_gaussian_noise(image):
        """Thêm nhiễu Gaussian"""
        transform = A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
        ])
        return transform(image=image)['image']

    def apply_blur_effects(image):
        """Áp dụng các hiệu ứng blur khác nhau"""
        transform = A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0)
            ], p=1.0)
        ])
        return transform(image=image)['image']

    def geometric_distortion(image):
        """Biến dạng hình học"""
        transform = A.Compose([
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.2, p=1.0),
                A.GridDistortion(distort_limit=0.2, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=1.0)
        ])
        return transform(image=image)['image']

    def enhance_contrast_color(image):
        """Điều chỉnh độ tương phản và màu sắc nâng cao"""
        transform = A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        ])
        return transform(image=image)['image']

    def rotate_10(self,lfimage):
        """Xoay tối đa 10 độ"""
        transform = A.Compose([
            A.Rotate(limit=10, p=1.0)
        ])
        return transform(image=image)['image']

    def adjust_lighting(self,image):
        """Điều chỉnh lighting với giới hạn 0.2"""
        transform = A.Compose([
            A.RandomBrightness(limit=0.2, p=1.0)
        ])
        return transform(image=image)['image']

    def apply_warp(self,image):
        """Áp dụng warp với giới hạn 0.2"""
        transform = A.Compose([
            A.GridDistortion(distort_limit=0.2, p=1.0)
        ])
        return transform(image=image)['image']

    def apply_zoom(self,image):
        """Áp dụng zoom với hệ số 1.1"""
        transform = A.Compose([
            A.Zoom(scale_range=(1.0, 1.1), p=1.0)
        ])
        return transform(image=image)['image']

    def horizontal_flip(self,image):
        """Lật ngang với xác suất 50%"""
        transform = A.Compose([
            A.HorizontalFlip(p=0.5)
        ])
        return transform(image=image)['image']

    def adjust_brightness_contrast_saturation(self,image):
        """Điều chỉnh độ sáng, tương phản và bão hòa"""
        transform = A.Compose([
            A.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.85, 1.15),
                saturation=(0.95, 1.05),
                p=1.0
            )
        ])
        return transform(image=image)['image']

    # Tích hợp các phương pháp vào các method chính
    def method_1_augmentation(self, image):
        """Phương pháp augmentation thứ nhất"""
        transforms = [
            self.rotate_10(image),
            self.adjust_lighting(image),
            self.apply_warp(image),
            self.apply_zoom(image)
        ]
        return np.random.choice(transforms)

    def method_2_augmentation(self, image):
        """Phương pháp augmentation thứ hai"""
        transforms = [
            self.horizontal_flip(image),
            self.adjust_brightness_contrast_saturation(image)
        ]
        return np.random.choice(transforms)

    def method_3_augmentation(self, image):
        """Phương pháp augmentation thứ ba"""
        transforms = [
            self.rotate_90(image),
            self.transpose(image),
            self.add_gaussian_noise(image),
            self.apply_blur_effects(image),
            self.geometric_distortion(image),
            self.enhance_contrast_color(image)
        ]
        return np.random.choice(transforms)

    # Thêm một phương pháp mới kết hợp nhiều hiệu ứng
    def combined_augmentation(self, image):
        """Kết hợp nhiều phương pháp augmentation"""
        transform = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=10, p=0.5)
            ], p=1.0),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=7, p=0.5)
            ], p=0.5),
            
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.2, p=0.5),
                A.GridDistortion(distort_limit=0.2, p=0.5)
            ], p=0.5),
            
            A.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.85, 1.15),
                saturation=(0.95, 1.05),
                p=0.8
            ),
            
            A.HorizontalFlip(p=0.5),
            
            A.OneOf([
                A.Zoom(scale_range=(1.0, 1.1), p=0.5),
                A.GridDistortion(distort_limit=0.2, p=0.5)
            ], p=0.5)
        ])
        return transform(image=image)['image']


    def save_image(self, image, folder, prefix):
        """Lưu ảnh với timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        path = os.path.join(self.output_dir, folder, filename)
        cv2.imwrite(path, image)
        return path

    def process_single_image(self, image_path, num_samples=3):
        """Xử lý một ảnh với tất cả các phương pháp"""
        try:
            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh từ {image_path}")

            # Lưu ảnh gốc
            original_filename = os.path.basename(image_path)
            original_path = os.path.join(self.output_dir, 'original', original_filename)
            cv2.imwrite(original_path, image)

            results = {
                'original': image,
                'paths': {'original': original_path}
            }

            # Xử lý denoise
            denoised_images = self.denoise_image(image)
            if denoised_images:
                results['denoised'] = denoised_images
                for denoise_type, denoised_img in denoised_images.items():
                    path = self.save_image(denoised_img, 'denoised', 
                                         f'{os.path.splitext(original_filename)[0]}_{denoise_type}')
                    results['paths'][f'denoised_{denoise_type}'] = path

            # Tạo và lưu augmented images
            for method_num in range(1, 4):
                method_results = []
                method_paths = []
                augment_func = getattr(self, f'method_{method_num}_augmentation')
                
                for i in range(num_samples):
                    augmented = augment_func(image)
                    path = self.save_image(augmented, f'method{method_num}',
                                         f'{os.path.splitext(original_filename)[0]}_method{method_num}_sample{i+1}')
                    method_results.append(augmented)
                    method_paths.append(path)
                
                results[f'method{method_num}'] = method_results
                results['paths'][f'method{method_num}'] = method_paths

            return results

        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
            return None

    def process_directory(self, num_samples=3):
        """Xử lý tất cả ảnh trong thư mục input"""
        image_files = self.get_image_files()
        if not image_files:
            print(f"Không tìm thấy ảnh trong thư mục {self.input_dir}")
            return

        print(f"Tìm thấy {len(image_files)} ảnh")
        results = {}

        for image_path in tqdm(image_files, desc="Đang xử lý ảnh"):
            result = self.process_single_image(image_path, num_samples)
            if result:
                results[image_path] = result

        return results

    def visualize_results(self, results, max_images=5):
        """Hiển thị kết quả cho một số ảnh"""
        if not results:
            print("Không có kết quả để hiển thị")
            return

        # Chọn ngẫu nhiên một số ảnh để hiển thị
        image_paths = list(results.keys())[:max_images]
        
        for image_path in image_paths:
            result = results[image_path]
            
            # Tạo figure với nhiều subplot
            rows = 4  # Original + 3 methods
            cols = 4  # Original/Denoised + 3 samples
            fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
            fig.suptitle(f'Results for {os.path.basename(image_path)}')
            
            # Hiển thị ảnh gốc và denoised
            axs[0, 0].imshow(cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title('Original')
            
            if 'denoised' in result:
                for i, (denoise_type, denoised) in enumerate(result['denoised'].items(), 1):
                    if i < cols:
                        axs[0, i].imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
                        axs[0, i].set_title(f'Denoised ({denoise_type})')

            # Hiển thị augmented images
            for method_num in range(1, 4):
                method_results = result[f'method{method_num}']
                for j, img in enumerate(method_results):
                    if j < cols:
                        axs[method_num, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        axs[method_num, j].set_title(f'Method {method_num} Sample {j+1}')

            # Tắt axes
            for ax in axs.flat:
                ax.axis('off')

            plt.tight_layout()
            plt.show()

def main():
    # Thư mục chứa ảnh đầu vào
    input_dir = "train\images"  # Thay đổi đường dẫn này
    
    # Khởi tạo augmentor
    augmentor = ImageAugmentor(input_dir=input_dir)

    try:
        # Xử lý tất cả ảnh trong thư mục
        results = augmentor.process_directory(num_samples=3)
        
        # Hiển thị kết quả
        augmentor.visualize_results(results)
        
        print("\nXử lý hoàn tất. Ảnh đã được lưu trong thư mục augmented_images")

    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    main()
