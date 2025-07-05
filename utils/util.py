'''
utils.py contains the following snippets:
    1. AvgrageMeter: used to calculate the average value of a variable
    2. cxcorr_align: used to align the rppg signals and ecg signals (by cross-correlation) ref. to cxcorr_np loss
    3. get_landmark, align_face, crop_image, compute_transform, crop_faces, crop_faces_by_quads: used to align the video frames

some codes are from https://github.com/rotemtzaban/STIT.

before using the face_align functions, you need to download the shape_predictor_68_face_landmarks.dat file to ./
'''

import torch, math, PIL, dlib, face_alignment, scipy
import scipy.ndimage
import skimage.io as io
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import torch.nn.functional as F




class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        
def update_avg_meters(meter_dict, update_dict, n):
    for key in update_dict.keys():
        meter_dict[key].update(update_dict[key], n=n)

def cxcorr_align(preds, labels):
    nom = torch.linalg.norm(preds, keepdim=True) * torch.linalg.norm(labels, keepdim=True)
    zi = torch.fft.irfft(torch.fft.rfft(preds)*torch.fft.rfft(labels.flip(-1)))
    cxcorr = zi/nom
    # cxcorr = cxcorr.abs()
    # out = np.abs(out)
    for b in range(cxcorr.shape[0]):
        _cxcorr = cxcorr[b]
        # max_idx = np.where(np.diff(np.sign(np.diff(_cxcorr))) < 0)[0] + 1
        # min_idx = np.where(np.diff(np.sign(np.diff(_cxcorr))) > 0)[0] + 1
        # torch
        max_idx = torch.where(torch.diff(torch.sign(torch.diff(_cxcorr))) < 0)[0] + 1
        min_idx = torch.where(torch.diff(torch.sign(torch.diff(_cxcorr))) > 0)[0] + 1

        shift = min_idx[0] if min_idx[0] < max_idx[0] else 0 - max_idx[0]
        preds[b] = torch.roll(preds[b], shift.item(), dims=0)

    return preds

def pearson_correlation_coefficient(preds, labels):
    sum_x = np.sum(preds)  # x
    sum_y = np.sum(labels)  # y
    sum_xy = np.sum(preds * labels)  # xy
    sum_x2 = np.sum(pow(preds, 2))  # x^2
    sum_y2 = np.sum(pow(labels, 2))  # y^2
    N = preds.shape[0]
    return (N * sum_xy - sum_x * sum_y) / (
        np.sqrt((N * sum_x2 - pow(sum_x, 2)) * (N * sum_y2 - pow(sum_y, 2)))
    )

def cal_psd_hr(output, Fs, return_type):
    
    cur_device = output.device
    
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = 2 * math.pi * torch.arange(0, N, dtype=torch.float) / N
        hanning = torch.from_numpy(np.hanning(N)).type(torch.FloatTensor).view(1, -1)

        k = k.type(torch.FloatTensor).to(cur_device)
        two_pi_n_over_N = two_pi_n_over_N.to(cur_device)
        hanning = hanning.to(cur_device)
        
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        return complex_absolute
    
    output = output.view(1, -1)
    
    N = output.size()[1]
    bpm_range = torch.arange(40, 180, dtype=torch.float)
    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz
    
    # only calculate feasible PSD range [0.7,4]Hz
    complex_absolute = compute_complex_absolute_given_k(output, k, N)
    complex_absolute = (1.0 / complex_absolute.sum()) * complex_absolute
    complex_absolute = complex_absolute.view(-1)
    whole_max_val, whole_max_idx = complex_absolute.max(0) # max返回（values, indices）
    whole_max_idx = whole_max_idx.type(torch.float) # 功率谱密度的峰值对应频率即为心率

    if return_type == 'psd':
        return complex_absolute
    elif return_type == 'hr':
        return whole_max_idx + 40	# Analogous Softmax operator

# functions below are used to align the video frames

def get_landmark(filepath, predictor, detector=None, fa=None):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    if fa is not None:
        image = io.imread(filepath)
        lms, _, bboxes = fa.get_landmarks(image, return_bboxes=True)
        return None if len(lms) == 0 else lms[0]
    if detector is None:
        detector = dlib.get_frontal_face_detector()
    if isinstance(filepath, PIL.Image.Image):
        img = np.array(filepath)
    else:
        img = dlib.load_rgb_image(filepath)
    dets = detector(img)

    for k, d in enumerate(dets):
        shape = predictor(img, d)
        break
    else:
        return None
    t = list(shape.parts())
    a = [[tt.x, tt.y] for tt in t]
    return np.array(a)

def align_face(filepath_or_image, predictor, output_size, detector=None,
               enable_padding=False, scale=1.0):
    """
    :param filepath: str
    :return: PIL Image
    """

    c, x, y = compute_transform(filepath_or_image, predictor, detector=detector,
                                scale=scale)
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    return crop_image(
        filepath_or_image, output_size, quad, enable_padding=enable_padding
    )

def crop_image(filepath, output_size, quad, enable_padding=False):
    x = (quad[3] - quad[1]) / 2
    qsize = np.hypot(*x) * 2
    # read image
    if isinstance(filepath, PIL.Image.Image):
        img = filepath
    else:
        img = PIL.Image.open(filepath)
    transform_size = output_size
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if (crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]):
        img = img.crop(crop)
        quad -= crop[:2]
    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    return img

def compute_transform(filepath, predictor, detector=None, scale=1.0, fa=None):
    lm = get_landmark(filepath, predictor, detector, fa)
    if lm is None:
        # raise Exception(f'Did not detect any faces in image: {filepath}')
        if not isinstance(filepath, PIL.Image.Image):
            print(f'{str(filepath.split("/")[-1][:-4])} ', end = '')
        return None, None, None
    lm_chin = lm[: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

    x *= scale
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return c, x, y

def crop_faces(IMAGE_SIZE, files, scale, center_sigma=0.0, xy_sigma=0.0, use_fa=False):
    if use_fa:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
        predictor = None
        detector = None
    else:
        fa = None
        predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        detector = dlib.get_frontal_face_detector()

    cs, xs, ys = [], [], []
    for _, path in tqdm(files):
        c, x, y = compute_transform(path, predictor, detector=detector,
                                    scale=scale, fa=fa)
        cs.append(c)
        xs.append(x)
        ys.append(y)

    ## 补全没能检测到人脸帧对应的 c, x, y
    cs_cut_none = [i for i in cs if i is not None]
    xs_cut_none = [i for i in xs if i is not None]
    ys_cut_none = [i for i in ys if i is not None]

    if len(cs_cut_none) < len(cs):
        print()

    # if len(cs_cut_none) < len(cs) * 0.5:
    #     print(f'------------ left [{len(cs_cut_none)} / {len(cs)}] frames ', end='')
    #     return None, None, None
    if not cs_cut_none:
        print(f'------------ left [{len(cs_cut_none)} / {len(cs)}] frames ', end='')
        return None, None, None

    cs_mean = np.mean(cs_cut_none, axis=0)
    xs_mean = np.mean(xs_cut_none, axis=0)
    ys_mean = np.mean(ys_cut_none, axis=0)
    for i in range(len(cs)):
        if cs[i] is None:
            cs[i] = cs_mean
        if xs[i] is None:
            xs[i] = xs_mean
        if ys[i] is None:
            ys[i] = ys_mean


    cs = np.stack(cs)
    xs = np.stack(xs)
    ys = np.stack(ys)
    if center_sigma != 0:
        cs = gaussian_filter1d(cs, sigma=center_sigma, axis=0)

    if xy_sigma != 0:
        xs = gaussian_filter1d(xs, sigma=xy_sigma, axis=0)
        ys = gaussian_filter1d(ys, sigma=xy_sigma, axis=0)

    quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
    quads = list(quads)

    crops, orig_images = crop_faces_by_quads(IMAGE_SIZE, files, quads)

    return crops, orig_images, quads

def crop_faces_by_quads(IMAGE_SIZE, files, quads):
    orig_images = []
    crops = []
    for quad, (_, path) in tqdm(zip(quads, files), total=len(quads)):
        crop = crop_image(path, IMAGE_SIZE, quad.copy())
        orig_image = path if isinstance(path, PIL.Image.Image) else Image.open(path)
        orig_images.append(orig_image)
        crops.append(crop)
    return crops, orig_images

def calc_alignment_coefficients(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.extend(
            (
                [p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]],
                [0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]],
            )
        )
    a = np.matrix(matrix, dtype=float)
    b = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(a.T * a) * a.T, b)
    return np.array(res).reshape(8)

def augment_flip(clip):
    if torch.rand(1) > 0.8:
        clip = torch.flip(clip, [4])
        clip = torch.flip(clip, [3])
    return clip


def augment_time_reversal(clip):
    if torch.rand(1) > 0.8:
        clip = torch.roll(clip, clip.shape[2] // 4, 2)
    return clip

def resize_clip(clip, length):
    '''
    Input:
        clip: tensor of shape [B,C,T,H,W]
        length: number of time points in output interpolated sequence
    Returns:
        Tensor of shape [B,C,T,H,W]
    '''
    T = clip.shape[2]
    clip = F.interpolate(clip, (T, length, length), mode='trilinear', align_corners=False)
    return clip

def random_resized_crop(clip, crop_scale_lims=[0.4, 1]):
    ''' Randomly crop a subregion of the video and resize it back to original size.
    Arguments:
        clip (np.array): expects [B,C,T,H,W]
    Returns:
        clip (np.array): same dimensions as input
    '''
    crop_scale = np.random.uniform(crop_scale_lims[0], crop_scale_lims[1])
    crop_length = np.round(crop_scale * clip.shape[3]).astype(int)
    crop_start_lim = np.max(clip.shape[3] - (crop_length), 0)
    x1 = np.random.randint(0, crop_start_lim+1)
    y1 = x1
    x2 = x1 + crop_length
    y2 = y1 + crop_length
    cropped_clip = clip[:,:,:,y1:y2,x1:x2]
    resized_clip = resize_clip(cropped_clip, clip.shape[3])
    return resized_clip

def augment_gaussian_noise(clip):
    B, C, T, H, W = clip.shape
    device = clip.device
    clip_noise = torch.normal(0, 5 / 255, (B, H, W)).to(device)
    clip_noise = clip_noise.unsqueeze(1).unsqueeze(1)    
    clip = clip + clip_noise
    return clip


def gaussian_kernel2d(kernel_size, sigma):
    """Generate a 2D Gaussian kernel."""
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel = kernel / np.sum(kernel)
    return kernel
def augment_motion_blur(clip):
    # 在每张图像的2D维度进行高斯模糊
    # 生成高斯核，5 & 0.1.
    device = clip.device
    kernel_size = 5
    sigma = 0.1
    kernel = gaussian_kernel2d(kernel_size, sigma)
    kernel = torch.from_numpy(kernel).float().to(device)
    # 进行卷积操作
    # 假设 clip 的形状为 [B, C, T, H, W]
    B, C, T, H, W = clip.shape
    clip = clip.view(B * C * T, 1, H, W)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    blurred_clip = F.conv2d(clip, kernel, padding=kernel_size // 2)
    blurred_clip = blurred_clip.view(B, C, T, H, W)
    return blurred_clip
    