"""
gen_conv_test.py

Generate firmware/conv_test_data.h for the StarDist 2D_versatile_fluo first
conv layer tests.

Outputs:
  - conv_test_data.h: compact C metadata, the existing small exhaustive conv1
    dataset, full-image metadata, samples, and expected rolling hash.
  - conv_input_image.hex: full normalized float32 image words for testbench
    preloading at CONV_FULL_IMAGE_BASE.

By default the full-image input is the first TIFF found in firmware/. Useful
overrides:
  CONV_INPUT_TIF=/path/to/image.tif python3 firmware/gen_conv_test.py
  CONV_CROP_X=100 CONV_CROP_Y=200 python3 firmware/gen_conv_test.py
  CONV_USE_RAMP=1 python3 firmware/gen_conv_test.py
  CONV_FULL_SYNTH=17x19 python3 firmware/gen_conv_test.py
"""

import glob
import os
import re
import struct

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SCRIPT_DIR = os.path.dirname(__file__)
OUT_PATH = os.path.join(SCRIPT_DIR, 'conv_test_data.h')
IMAGE_HEX_PATH = os.path.join(SCRIPT_DIR, 'conv_input_image.hex')
OUT_H = 6
OUT_W = 6
OUT_C = 32
CONV_K = 9
FULL_IMAGE_BASE = 0x00200000
FULL_EXPECTED_N_BANKS = 1


def float_to_word(f):
    return struct.unpack('<I', struct.pack('<f', float(f)))[0]


def float_to_hex(f):
    return '0x{:08X}'.format(float_to_word(f))


def word_to_float(word):
    return struct.unpack('<f', struct.pack('<I', int(word) & 0xFFFFFFFF))[0]


def hex_to_float(word):
    return word_to_float(int(word, 16))


def arr_to_c(name, arr, comment=''):
    flat = arr.flatten()
    lines = []
    if comment:
        lines.append(f'/* {comment} */')
    lines.append(f'#define {name.upper()}_LEN {len(flat)}')
    lines.append(f'static const unsigned int {name}[{len(flat)}] = {{')
    for i in range(0, len(flat), 4):
        chunk = flat[i:i+4]
        lines.append('    ' + ', '.join(float_to_hex(v) for v in chunk) + ',')
    lines.append('};')
    return '\n'.join(lines)


def int_arr_to_c(name, arr, comment=''):
    lines = []
    if comment:
        lines.append(f'/* {comment} */')
    lines.append(f'#define {name.upper()}_LEN {len(arr)}')
    lines.append(f'static const unsigned int {name}[{len(arr)}] = {{')
    for i in range(0, len(arr), 4):
        lines.append('    ' + ', '.join(f'0x{int(v) & 0xFFFFFFFF:08X}' for v in arr[i:i+4]) + ',')
    lines.append('};')
    return '\n'.join(lines)


def extract_header_array(header_text, name):
    pattern = rf'static const unsigned int {name}\[[^\]]+\] = \{{(.*?)\}};'
    match = re.search(pattern, header_text, flags=re.S)
    if not match:
        raise RuntimeError(f'Could not find {name} in existing conv_test_data.h')
    words = re.findall(r'0x[0-9A-Fa-f]{8}', match.group(1))
    return np.array([hex_to_float(word) for word in words], dtype=np.float32)


def load_weights_from_existing_header():
    with open(OUT_PATH) as f:
        header_text = f.read()
    weights = extract_header_array(header_text, 'conv_weights').reshape(OUT_C, CONV_K)
    biases = extract_header_array(header_text, 'conv_biases')
    w_keras_shape = np.zeros((3, 3, 1, OUT_C), dtype=np.float32)
    for c in range(OUT_C):
        w_keras_shape[:, :, 0, c] = weights[c].reshape(3, 3)
    return w_keras_shape, biases, 'relu', 'existing conv_test_data.h weights'


def load_weights():
    try:
        from stardist.models import StarDist2D

        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        first_conv = model.keras_model.get_layer('conv2d')
        weights, biases = first_conv.get_weights()
        activation = first_conv.activation.__name__
        print('Loaded StarDist 2D_versatile_fluo conv2d weights')
        print(f'First conv: kernel={weights.shape}, bias={biases.shape}, '
              f'padding={first_conv.padding}, activation={activation}')
        return weights.astype(np.float32), biases.astype(np.float32), activation, 'StarDist 2D_versatile_fluo conv2d'
    except Exception as exc:
        print(f'Could not import/load StarDist model ({type(exc).__name__}: {exc})')
        print('Reusing weights and biases from existing conv_test_data.h')
        return load_weights_from_existing_header()


def find_default_tiff():
    candidates = sorted(glob.glob(os.path.join(SCRIPT_DIR, '*.tif')) +
                        glob.glob(os.path.join(SCRIPT_DIR, '*.tiff')))
    return candidates[0] if candidates else None


def normalize_image_array(arr):
    if np.issubdtype(arr.dtype, np.integer):
        scale = float(np.iinfo(arr.dtype).max)
    else:
        scale = float(np.max(arr)) if np.max(arr) != 0 else 1.0
    return (arr.astype(np.float32) / np.float32(scale)).astype(np.float32), scale


def load_full_image():
    synth = os.environ.get('CONV_FULL_SYNTH')
    if synth:
        w_str, h_str = synth.lower().split('x')
        width = int(w_str)
        height = int(h_str)
        vals = np.arange(height * width, dtype=np.float32).reshape(height, width)
        vals = (vals % 257) / np.float32(257.0)
        return vals.astype(np.float32), f'synthetic {width}x{height} pattern', None, 1.0

    tif_path = os.environ.get('CONV_INPUT_TIF') or find_default_tiff()
    if not tif_path:
        vals = np.linspace(0.0, 1.0, 17 * 19, dtype=np.float32).reshape(19, 17)
        return vals, 'synthetic fallback 17x19 ramp (no TIFF found)', None, 1.0

    from PIL import Image

    image = Image.open(tif_path)
    arr = np.array(image)
    if arr.ndim != 2:
        raise RuntimeError(f'Expected grayscale TIFF, got shape {arr.shape}')
    norm, scale = normalize_image_array(arr)
    desc = (f'{os.path.basename(tif_path)} full image, width={arr.shape[1]}, '
            f'height={arr.shape[0]}, dtype={arr.dtype}, scale={scale:g}')
    print(f'Loaded TIFF input: {tif_path}')
    print(f'Image shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}')
    return norm, desc, tif_path, scale


def make_small_input(full_image):
    if os.environ.get('CONV_USE_RAMP') == '1':
        inp = np.linspace(0.0, 1.0, OUT_H * OUT_W, dtype=np.float32).reshape(OUT_H, OUT_W)
        return inp, '6x6 ramp 0..1'

    if full_image.shape[0] < OUT_H or full_image.shape[1] < OUT_W:
        inp = np.linspace(0.0, 1.0, OUT_H * OUT_W, dtype=np.float32).reshape(OUT_H, OUT_W)
        return inp, '6x6 ramp 0..1 (full image too small for crop)'

    crop_x = os.environ.get('CONV_CROP_X')
    crop_y = os.environ.get('CONV_CROP_Y')
    x = int(crop_x) if crop_x is not None else (full_image.shape[1] - OUT_W) // 2
    y = int(crop_y) if crop_y is not None else (full_image.shape[0] - OUT_H) // 2
    crop = full_image[y:y+OUT_H, x:x+OUT_W]
    return crop.astype(np.float32), f'6x6 center crop from full image x={x}, y={y}'


def compute_raw_output(inp_padded, weights, biases, height, width):
    raw = np.zeros((height, width, OUT_C), dtype=np.float32)
    for k in range(CONV_K):
        ky = k // 3
        kx = k - ky * 3
        in_plane = inp_padded[ky:ky+height, kx:kx+width]
        for c in range(OUT_C):
            raw[:, :, c] = np.float32(
                raw[:, :, c] + np.float32(in_plane * np.float32(weights[ky, kx, 0, c]))
            )
    for c in range(OUT_C):
        raw[:, :, c] = np.float32(raw[:, :, c] + np.float32(biases[c]))
    return raw


def hash_update(h0, h1, value):
    value &= 0xFFFFFFFF
    h0 ^= value
    h0 = (h0 * 16777619) & 0xFFFFFFFF
    h1 = (h1 + value + 0x9E3779B9 + ((h1 << 6) & 0xFFFFFFFF) + (h1 >> 2)) & 0xFFFFFFFF
    return h0, h1


def hash_full_output(raw):
    height, width, channels = raw.shape
    h0 = 2166136261
    h1 = 0x9E3779B9
    for v in (height, width, channels, height * width * channels):
        h0, h1 = hash_update(h0, h1, int(v))
    for pixel in range(height * width):
        y = pixel // width
        x = pixel - y * width
        for c in range(channels):
            word = float_to_word(raw[y, x, c])
            h0, h1 = hash_update(h0, h1, pixel)
            h0, h1 = hash_update(h0, h1, c)
            h0, h1 = hash_update(h0, h1, word)
    return h0, h1


def make_sample_points(height, width, raw):
    pixels = {
        0,
        width - 1,
        (height - 1) * width,
        height * width - 1,
        width // 2,
        (height // 2) * width,
        (height // 2) * width + (width // 2),
        max(0, width - 2),
        max(0, width - 1),
        min(height * width - 1, width),
        min(height * width - 1, width + 1),
        min(height * width - 1, 510),
        min(height * width - 1, 511),
        min(height * width - 1, 512),
        min(height * width - 1, 513),
        max(0, height * width - 2),
        height * width - 1,
    }
    state = 0xC001D00D
    for _ in range(16):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        pixels.add(state % (height * width))

    channels = [0, 1, 7, 16, 31]
    samples = []
    for pixel in sorted(pixels):
        y = pixel // width
        x = pixel - y * width
        for c in channels:
            samples.append((pixel, c, float_to_word(raw[y, x, c])))
    samples.sort()
    return samples


weights, biases, activation, weight_source = load_weights()
full_image, full_desc, full_path, full_scale = load_full_image()
small_input, small_desc = make_small_input(full_image)
small_padded = np.pad(small_input, 1, mode='constant', constant_values=0.0).astype(np.float32)
small_raw = compute_raw_output(small_padded, weights, biases, OUT_H, OUT_W)

full_h, full_w = full_image.shape
full_padded = np.pad(full_image, 1, mode='constant', constant_values=0.0).astype(np.float32)
print(f'Computing expected full-image conv1 output for {full_w}x{full_h}...')
full_raw = compute_raw_output(full_padded, weights, biases, full_h, full_w)
print('Computing full-image expected hash and samples...')
hash0, hash1 = hash_full_output(full_raw)
samples = make_sample_points(full_h, full_w, full_raw)

with open(IMAGE_HEX_PATH, 'w') as f:
    for word in (float_to_word(v) for v in full_image.flatten()):
        f.write(f'{word:08x}\n')

sample_pixels = [p for p, _, _ in samples]
sample_channels = [c for _, c, _ in samples]
sample_expected = [w for _, _, w in samples]
smoke_indices = [0, min(full_h * full_w - 1, full_w // 2), min(full_h * full_w - 1, full_w), full_h * full_w - 1]
smoke_values = [float_to_word(full_image.flat[i]) for i in smoke_indices]

with open(OUT_PATH, 'w') as f:
    f.write('/* Auto-generated by gen_conv_test.py -- do not edit manually */\n')
    f.write('/* StarDist 2D_versatile_fluo, first conv layer (3x3x1->32) */\n')
    f.write(f'/* Weight source: {weight_source} */\n')
    f.write(f'/* Small input: {small_desc}; padded to 8x8 for im2col */\n')
    f.write(f'/* Full input: {full_desc}; hex file: {os.path.basename(IMAGE_HEX_PATH)} */\n\n')
    f.write('#ifndef CONV_TEST_DATA_H\n#define CONV_TEST_DATA_H\n\n')

    f.write(arr_to_c('conv_input_padded', small_padded,
                     f'Padded small input: 8x8 float32 from {small_desc}'))
    f.write('\n\n')

    weights_out = np.stack([weights[:, :, 0, c].flatten() for c in range(OUT_C)])
    f.write(arr_to_c('conv_weights', weights_out,
                     'Weights: [out_channel * 9 + spatial_idx], row-major kh*kw'))
    f.write('\n\n')

    f.write(arr_to_c('conv_biases', biases, 'Biases: one per output channel (32 values)'))
    f.write('\n\n')

    f.write(arr_to_c('conv_expected', small_raw,
                     'Expected small output (dot+bias, before activation): shape (6,6,32)'))
    f.write('\n\n')

    f.write(f'/* Keras activation: {activation} */\n')
    f.write(f'#define CONV_OUT_H {OUT_H}\n')
    f.write(f'#define CONV_OUT_W {OUT_W}\n')
    f.write(f'#define CONV_OUT_C {OUT_C}\n')
    f.write(f'#define CONV_K     {CONV_K}   /* kernel spatial size: 3x3 */\n\n')

    f.write('/* Full-image tiled conv1 metadata */\n')
    f.write(f'#define CONV_FULL_IMAGE_BASE 0x{FULL_IMAGE_BASE:08X}u\n')
    f.write(f'#define CONV_FULL_IMAGE_H {full_h}u\n')
    f.write(f'#define CONV_FULL_IMAGE_W {full_w}u\n')
    f.write(f'#define CONV_FULL_IMAGE_WORDS {full_h * full_w}u\n')
    f.write(f'#define CONV_FULL_EXPECTED_N_BANKS {FULL_EXPECTED_N_BANKS}u\n')
    f.write(f'#define CONV_FULL_EXPECTED_HASH0 0x{hash0:08X}u\n')
    f.write(f'#define CONV_FULL_EXPECTED_HASH1 0x{hash1:08X}u\n')
    f.write(f'#define CONV_FULL_SAMPLE_COUNT {len(samples)}u\n\n')

    f.write(int_arr_to_c('conv_full_smoke_indices', smoke_indices,
                         'Image preload smoke-test word indices'))
    f.write('\n\n')
    f.write(int_arr_to_c('conv_full_smoke_values', smoke_values,
                         'Expected image preload smoke-test words'))
    f.write('\n\n')
    f.write(int_arr_to_c('conv_full_sample_pixels', sample_pixels,
                         'Full-image exact sample output pixel indices'))
    f.write('\n\n')
    f.write(int_arr_to_c('conv_full_sample_channels', sample_channels,
                         'Full-image exact sample output channels'))
    f.write('\n\n')
    f.write(int_arr_to_c('conv_full_sample_expected', sample_expected,
                         'Full-image exact sample expected output words'))
    f.write('\n\n')

    f.write('#endif /* CONV_TEST_DATA_H */\n')

print(f'\nWritten: {OUT_PATH}')
print(f'Written: {IMAGE_HEX_PATH}')
print(f'Full image hash: 0x{hash0:08X} 0x{hash1:08X}; samples={len(samples)}')
print('Sample small input values:')
for row in small_input:
    print('  ' + ' '.join(f'{v:.6f}' for v in row))
