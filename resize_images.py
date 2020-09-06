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
    image_size = (224,224)
    input_folder = "data/train"
    output_folder = "data/train224"

    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    

    
    Parallel(n_jobs=12)(
        delayed(resize_image)(
            i,
            output_folder,
            image_size,
        ) for i in tqdm(images)
    )

    input_folder = "data/test"
    output_folder = "data/test224"

    images = glob.glob(os.path.join(input_folder, "*.jpg"))

    Parallel(n_jobs=12)(
        delayed(resize_image)(
            i,
            output_folder,
            image_size,
        ) for i in tqdm(images)
    )
