import cv2
import os
import argparse
import sys
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import crop_faces

def video_preprocess(dataset = "", root_dir = ""):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
    ]

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        fnames = os.listdir(dir)
        for fname in fnames:
            if fname.split('.')[0][-1] == ')':
                os.remove(os.path.join(dir, fname))
                print(f'---------  remove {os.path.join(dir, fname)} for endwith ([0-9])  ---------')
        for fname in sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0])):
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
        return images
    
    def make_dataset_from_video(video):
        images = []
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = cap.read()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fname = f'{i:05d}'
            if ret:
                images.append((fname, frame))
        cap.release()
        return images
  
    if dataset == "PURE":
        date_list = sorted(os.listdir(root_dir))
        for date in date_list:
            # read video
            video_dir = os.path.join(root_dir, date, date)
            video_save_dir = os.path.join(root_dir, date, 'align_crop_pic')
            if not os.path.exists(video_save_dir):
                os.makedirs(video_save_dir)
            else:
                if len(os.listdir(video_save_dir)) == len(os.listdir(video_dir)):
                    print(f'already processed {video_save_dir} -----------------')
                    continue
            files = []
            for fname in sorted(os.listdir(video_dir)):
                if is_image_file(fname):
                    path = os.path.join(video_dir, fname)
                    fname = fname.split('.')[0]
                    files.append((fname, path))
                else:
                    raise ValueError(f'frame {fname} is not png')
            
            image_size = 128
            scale = 0.8
            center_sigma = 1.0
            xy_sigma = 3.0
            use_fa = False

            crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

            if crops is None:
                print(f'too less face detected in file {video_dir} -----------------')
                continue

            for i in range(len(crops)):
                img = crops[i]
                img.save(os.path.join(video_save_dir, files[i][0] + '.png'))
            print(f'generated {len(crops)} png images in file {video_save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COHFACE', help='VIPL or UBFC or PURE or COHFACE')
    parser.add_argument('--dataset_dir', type=str, default='', help='dataset dir')
    args = parser.parse_args()
    video_preprocess(args.dataset, args.dataset_dir)
