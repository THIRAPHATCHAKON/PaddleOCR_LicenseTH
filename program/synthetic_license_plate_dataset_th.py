import random
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path("dataset_value_final") # file name output 

IMG_TRAIN_DIR = ROOT / "img" / "train"
IMG_VAL_DIR = ROOT / "img" / "val"

LABEL_TRAIN_FILE = ROOT / "label" / "train" / "train.txt" 
LABEL_VAL_FILE = ROOT / "label" / "val" / "val.txt" 

DICT_FILE = ROOT / "plate_dict.txt" # name dict output

for d in [
    IMG_TRAIN_DIR,
    IMG_VAL_DIR,
    LABEL_TRAIN_FILE.parent,
    LABEL_VAL_FILE.parent
]:
    d.mkdir(parents=True, exist_ok=True)


FONT_PATH = r"D:\license_plate_and_car_detection\Sarun's ThangLuang.ttf" # Path Font


PLATE_FONT_SIZE_MIN = 10
PLATE_FONT_SIZE_MAX = 32

PROVINCE_FONT_SIZE_RATIO = 0.48

PROVINCE_FONT_SIZE_MIN = max(6, round(PLATE_FONT_SIZE_MIN * PROVINCE_FONT_SIZE_RATIO))

PROVINCE_FONT_SIZE_MAX = round(PLATE_FONT_SIZE_MAX * PROVINCE_FONT_SIZE_RATIO)


IMAGE_HEIGHT = 48
IMAGE_WIDTH = 180 

NUM_IMAGES = 20000 # num of imgs

TRAIN_RATIO = 0.9

PROVINCES = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา",
    "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก",
    "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน",
    "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา",
    "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์", "แพร่", "พะเยา", "ภูเก็ต",
    "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี",
    "ลพบุรี", "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี",
    "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อุดรธานี", "อุตรดิตถ์",
    "อุทัยธานี", "อุบลราชธานี", "อำนาจเจริญ", "เบตง"
]

# Thai Letters
THAI_LETTERS = list("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")

DIGITS = list("0123456789")


# Balanced Sampler
class BalancedSampler:

    def __init__(self, items):
        self.items = list(items)
        self.pool = []
        self.counts = Counter()

    def next(self):

        if not self.pool:
            self.pool = self.items[:]
            random.shuffle(self.pool)

        item = self.pool.pop()

        self.counts[item] += 1

        return item

    def report(self, title):

        print(f"\n=== {title} ===")

        if not self.counts:
            print("  ยังไม่มีข้อมูล")
            return

        total = sum(self.counts.values())
        vals = list(self.counts.values())

        print(f"  รวมทั้งหมด: {total} ครั้ง | " f"ตัวที่ไม่ซ้ำ: {len(self.counts)}/{len(self.items)}")
        print(f"  น้อยสุด: {min(vals)} | " f"มากสุด: {max(vals)} | " f"เฉลี่ย: {total / len(self.items):.1f}")

        for item, count in self.counts.most_common():
            print(f"    {item}: {count}")

        missing = set(self.items) - set(self.counts)

        if missing:
            print(f"  ยังไม่เคยออกเลย " f"({len(missing)} ตัว): {sorted(missing)}")

# Samplers
letter_sampler = BalancedSampler(THAI_LETTERS)
digit_sampler = BalancedSampler(DIGITS)
province_sampler = BalancedSampler(PROVINCES)

# Random Functions
def balanced_digits(n):

    return "".join(digit_sampler.next() for _ in range(n))

def random_letters_only():

    n_chars = random.randint(2, 4)

    return "".join(letter_sampler.next() for _ in range(n_chars))


def random_plate():

    style = random.randint(0, 2)

    if style == 0:
        text = (letter_sampler.next() + letter_sampler.next() + balanced_digits(4))
        is_red = False
    elif style == 1:
        text = (digit_sampler.next() + letter_sampler.next() + letter_sampler.next() + balanced_digits(4))
        is_red = False
    else:
        text = (letter_sampler.next() + "-" + balanced_digits(4))
        is_red = True


    return text, is_red

def crop_to_content(img, bg_color, margin=6):

    w, h = img.size
    gray = np.array(img.convert("L"))
    bg_gray = int(0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])
    mask = gray < (bg_gray - 30)


    if mask.any():
        ys, xs = np.where(mask)
        top = ys.min()
        bottom = ys.max()
        left = xs.min()
        right = xs.max()
    else:
        top = 0
        bottom = h
        left = 0
        right = w

    top = max(0, top - margin)
    left = max(0, left - margin)
    bottom = min(h, bottom + margin)
    right = min(w, right + margin)

    return img.crop((left, top, right, bottom))

def biased_small_font_size(min_size, max_size, power=2.0):
    t = random.random() ** power

    return round(min_size + (max_size - min_size) * t)

def draw_text(text, is_red, is_province=False):

    if is_red:
        bg_color = (176, 30, 30)
    else:
        bg_color = (255, 255, 255)

    text_color = (0, 0, 0)

    if is_province:
        font_size = biased_small_font_size(PROVINCE_FONT_SIZE_MIN, PROVINCE_FONT_SIZE_MAX)
    else:
        font_size = biased_small_font_size(PLATE_FONT_SIZE_MIN, PLATE_FONT_SIZE_MAX)

    font = ImageFont.truetype(FONT_PATH, font_size)

    canvas_w = IMAGE_WIDTH * 3
    canvas_h = IMAGE_HEIGHT * 3
    img = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)

    draw.text((canvas_w // 2, canvas_h // 2), text, font=font, fill=text_color, anchor="mm")
    img = crop_to_content(img, bg_color, margin=6)
    target_ratio = (IMAGE_WIDTH / IMAGE_HEIGHT)
    crop_ratio = (img.width / img.height)

    if crop_ratio > target_ratio:
        new_w = IMAGE_WIDTH
        new_h = max(1, int(IMAGE_WIDTH / crop_ratio))
    else:
        new_h = IMAGE_HEIGHT
        new_w = max(1, int(IMAGE_HEIGHT * crop_ratio))

    img = img.resize((new_w, new_h), Image.LANCZOS)
    final = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), bg_color)
    off_x = (IMAGE_WIDTH - new_w) // 2
    off_y = (IMAGE_HEIGHT - new_h) // 2
    final.paste(img, (off_x, off_y))

    return final

full_char_set = set()
full_char_set.update(THAI_LETTERS)
full_char_set.update(DIGITS)
full_char_set.add("-")

for province in PROVINCES:
    full_char_set.update(province)

with open(DICT_FILE, "w", encoding="utf-8") as f:
    for ch in sorted(full_char_set):
        f.write(ch + "\n")

print(f"Dictionary size: " f"{len(full_char_set)} characters " f"-> {DICT_FILE}")

train_lines = []
val_lines = []

for i in range(NUM_IMAGES):
    r = random.random()
    if r < 0.50:
        text, is_red = random_plate()
        is_province = False
    elif r < 0.70:
        text = random_letters_only()
        is_red = False
        is_province = False
    else:
        text = province_sampler.next()
        is_red = False
        is_province = True

    img = draw_text(text, is_red, is_province=is_province)
    filename = f"{i + 1:06}.png"
    is_train = (i / NUM_IMAGES) < TRAIN_RATIO

    if is_train:
        img.save(IMG_TRAIN_DIR / filename)
        train_lines.append(f"img/train/{filename}\t{text}")
    else:
        img.save(IMG_VAL_DIR / filename)
        val_lines.append(f"img/val/{filename}\t{text}")

with open(LABEL_TRAIN_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(train_lines))
with open(LABEL_VAL_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(val_lines))

print(f"\nDone." f" train={len(train_lines)}" f" val={len(val_lines)} images")

letter_sampler.report("พยัญชนะไทย")
digit_sampler.report("ตัวเลข")
province_sampler.report("จังหวัด")