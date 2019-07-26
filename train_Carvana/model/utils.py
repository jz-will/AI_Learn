import numpy as np
import cv2


# 图片保存
def save_images(input, output1, output2, input_path, image_path, max_samples=4):
    # 在图片宽度上concatenate=>[batch_size,324,648,1](横向)
    image = np.concatenate([output1, output2], axis=2)  # concat 4D array, along width.
    # 纵向concatenate的图片个数=min(max_samples,batch_size)
    if max_samples > int(image.shape[0]):
        max_samples = int(image.shape[0])

    image = image[0:max_samples, :, :, :]
    # [
    image = np.concatenate([image[i, :, :, :] for i in range(max_samples)], axis=0)
    # concat 3D array, along axis=0, i.e. along height. shape: (648, 648, 1).

    # save image.
    # scipy.misc.toimage(), array is 2D(gray, reshape to (H, W)) or 3D(RGB).
    # scipy.misc.toimage(image, cmin=0., cmax=1.).save(image_path) # image_path contain image path and name.
    # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
    cv2.imwrite(image_path, np.uint8(image.clip(0., 1.) * 255.))

    # save input
    if input is not None:
        input_data = input[0:max_samples, :, :, :]
        # [1024,256,3]
        input_data = np.concatenate([input_data[i, :, :, :] for i in range(max_samples)], axis=0)
        cv2.imwrite(input_path, np.uint8(input_data.clip(0., 1.) * 255.))
