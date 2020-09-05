import os
import glob
from PIL import Image, ImageFile
from tqdm import tqdm
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    out_path = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )
    img.save(out_path)


if __name__ =="__main__":
    input_folder = "/media/vatsal/Movies & Games/down_siim-isic-melanoma-classification/jpeg/train"
    output_folder = "/media/vatsal/Movies & Games/down_siim-isic-melanoma-classification/train224"

    images = glob.glob(os.path.join(input_folder, "*.jpg"))

    Parallel(n_jobs=12)(
        delayed(resize_image)(
            i,
            output_folder,
            (224,224),
        ) for i in tqdm(images)
    )

    input_folder = "/media/vatsal/Movies & Games/down_siim-isic-melanoma-classification/jpeg/test"
    output_folder = "/media/vatsal/Movies & Games/down_siim-isic-melanoma-classification/test224"

    images = glob.glob(os.path.join(input_folder, "*.jpg"))

    Parallel(n_jobs=12)(
        delayed(resize_image)(
            i,
            output_folder,
            (224,224),
        ) for i in tqdm(images)
    )
