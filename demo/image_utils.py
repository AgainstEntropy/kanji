import os

from PIL import Image


class ImageStitcher:
    def __init__(
            self, 
            tmp_dir: str,
            img_res: int = 64,
            img_per_line: int = 10,
            verbose: bool = False
        ):
        self.update_tmp_dir(tmp_dir)

        self.img_res = img_res if isinstance(img_res, tuple) else (img_res, img_res)
        self.img_per_line = img_per_line

        self.verbose = verbose

        self.reset()

    def reset(self):
        self.cached_img = None
        self.img_num = 0
        self.num_lines = 1

        self.total_width = self.img_res[0] * self.img_per_line
        self.total_height = self.img_res[1] * self.num_lines

    def update_tmp_dir(self, tmp_dir: str):
        tmp_dir = os.path.abspath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        self.tmp_img_path_template = os.path.join(tmp_dir, "img_%03d.png")

    def add(self, img: Image, text: str = None):
        
        img = img.resize(self.img_res)

        if self.cached_img is None:
            new_img = Image.new('RGBA', (self.total_width, self.total_height))
            new_img.paste(img, (0, 0))
        else:
            num_lines = self.img_num // self.img_per_line + 1
            if num_lines > self.num_lines:
                self.num_lines = num_lines
                self.total_height = self.img_res[1] * self.num_lines
                new_img = Image.new('RGBA', (self.total_width, self.total_height))
                new_img.paste(self.cached_img, (0, 0))
            elif num_lines == self.num_lines:
                new_img = self.cached_img

            y_offset = self.img_res[1] * (num_lines - 1)
            x_offset = self.img_res[0] * (self.img_num % self.img_per_line)
            new_img.paste(img, (x_offset, y_offset))

        save_path = self.tmp_img_path_template % self.img_num
        if text is not None:
            save_path = save_path.replace(".png", f"_{text}.png")
        new_img.save(save_path)
        self.cached_img = new_img
        self.img_num += 1

        if self.verbose:
            print(f"Saved image to {save_path}")

        return save_path
    