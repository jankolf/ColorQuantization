import numpy as np
from PIL import Image



def quantize_colors_division(img: np.ndarray, num_colors_per_channel):
    if num_colors_per_channel == 256:
        return img

    div = 256 // num_colors_per_channel
    quantized = (img // div * div + div // 2).astype(np.float32)
    
    return quantized


def quantize_colors_kmeans(img, num_colors_per_channel):
    if num_colors_per_channel == 256:
        return img

    channels_first = False
    if img.shape[0] == 3:
        channels_first = True
        img = np.transpose(img, (1, 2, 0))
    
    img = img.astype(np.uint8)

    r_quant = Image.fromarray(img[:,:,0]).quantize(
                        colors=num_colors_per_channel, 
                        method=Image.MEDIANCUT
              ).convert("L")
    g_quant = Image.fromarray(img[:,:,1]).quantize(
                        colors=num_colors_per_channel, 
                        method=Image.MEDIANCUT
              ).convert("L")
    b_quant = Image.fromarray(img[:,:,2]).quantize(
                        colors=num_colors_per_channel, 
                        method=Image.MEDIANCUT
              ).convert("L")

    if channels_first:
        quantized = np.empty(
                        (3, img.shape[0], img.shape[1]),
                        dtype=np.float32
                    )
        quantized[0,:,:] = r_quant
        quantized[1,:,:] = g_quant
        quantized[2,:,:] = b_quant
    else:
        quantized = np.empty(
                        (img.shape[0], img.shape[1], 3),
                        dtype=np.float32
                    )
        quantized[:,:, 0] = r_quant
        quantized[:,:, 1] = g_quant
        quantized[:,:, 2] = b_quant
    
    return quantized
