import os
import random
import string

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# === CONFIG ===
img_width, img_height = 150, 50
font_size = 36
outline_width = 2
shadow_offset = (2, 2)
line_width = 3
line_count = 3
letters_per_image = 3
images_per_font = 1

characters = string.ascii_letters + string.digits
valid_classes = list(string.ascii_lowercase + string.digits)
char_to_class = {ch: i for i, ch in enumerate(valid_classes)}

fonts_dir = "fonts/"
dataset_dir = "datasets"

# === CREATE FOLDERS ===
os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)

def get_all_fonts():
    fonts = []
    for root, _, files in os.walk(fonts_dir):
        for f in files:
            if f.lower().endswith((".ttf", ".otf")):
                font_path = os.path.join(root, f)
                fonts.append(font_path)
    return fonts

def bbox_to_yolo(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def generate_image(index, font_path, split):
    letters = ''.join(random.choices(characters, k=letters_per_image))
    font = ImageFont.truetype(font_path, font_size)
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    # Calculate spacing
    letter_bboxes = [font.getbbox(letter) for letter in letters]
    letter_widths = [bbox[2] - bbox[0] for bbox in letter_bboxes]
    total_letter_width = sum(letter_widths)
    padding = 10
    space_between = max((img_width - total_letter_width - 2 * padding) // (len(letters) - 1), 0)

    x = padding
    hitboxes = []

    for i, letter in enumerate(letters):
        bbox = font.getbbox(letter)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        y = (img_height - h) / 2 - bbox[1]

        # Shadow
        # draw.text((x + shadow_offset[0], y + shadow_offset[1]), letter, font=font, fill="black")

        # Hollow or solid
        is_hollow = random.choice([True, False])
        if is_hollow:
            draw.text((x, y), letter, font=font, fill="white", stroke_width=outline_width, stroke_fill="black")
        else:
            draw.text((x, y), letter, font=font, fill="black")

        # Hitbox
        padding_px = 4
        x1 = max(0, int(x) - padding_px)
        y1 = max(0, int(y + bbox[1]) - padding_px)
        x2 = min(img_width, int(x + w) + padding_px)
        y2 = min(img_height, int(y + bbox[3]) + padding_px)
        box = (x1, y1, x2, y2)

        hitboxes.append((letter.lower(), box))
        x += w + space_between

    # === Distortions (Lines + Noise) ===
    used_y_positions = []
    for _ in range(line_count):
        style = random.choice(['line', 'curve', 'squiggle', 'dot', 'rect', 'blob', 'vline', 'hline'])
        x_start = random.randint(0, int(img_width * 0.2))
        length = random.randint(int(img_width * 0.7), int(img_width * 0.9))
        x_end = min(x_start + length, img_width)

        for _ in range(10):
            y = random.randint(0, img_height - 1)
            if all(abs(y - used) > 12 for used in used_y_positions):
                used_y_positions.append(y)
                break
        else:
            y = random.randint(0, img_height - 1)
            used_y_positions.append(y)

        if style == 'line':
            draw.line([(x_start, y), (x_end, y + random.randint(-1, 1))], fill="black", width=line_width)

        elif style == 'curve':
            points = [
                (x_start, y + random.randint(-2, 2)),
                (x_start + length // 3, y + random.randint(-4, 4)),
                (x_start + 2 * length // 3, y + random.randint(-4, 4)),
                (x_end, y + random.randint(-2, 2)),
            ]
            draw.line(points, fill="black", width=line_width, joint="curve")

        elif style == 'squiggle':
            step = 5
            amplitude = random.randint(3, 6)
            points = []
            for t in range(0, x_end - x_start, step):
                x = x_start + t
                y_offset = int(amplitude * random.uniform(-1, 1))
                points.append((x, y + y_offset))
            draw.line(points, fill="black", width=line_width)

        elif style == 'dot':
            for _ in range(random.randint(5, 15)):
                dot_x = random.randint(0, img_width - 1)
                dot_y = random.randint(0, img_height - 1)
                draw.point((dot_x, dot_y), fill="black")

        elif style == 'rect':
            for _ in range(random.randint(1, 3)):
                rx1 = random.randint(0, img_width - 10)
                ry1 = random.randint(0, img_height - 10)
                rx2 = rx1 + random.randint(3, 10)
                ry2 = ry1 + random.randint(3, 10)
                draw.rectangle([rx1, ry1, rx2, ry2], outline="black", width=1)

        elif style == 'blob':
            for _ in range(random.randint(1, 2)):
                bx = random.randint(0, img_width - 10)
                by = random.randint(0, img_height - 10)
                bw = random.randint(6, 12)
                bh = random.randint(6, 12)
                draw.ellipse([bx, by, bx + bw, by + bh], fill="black")

        elif style == 'vline':
            vx = random.randint(0, img_width - 1)
            draw.line([(vx, 0), (vx, img_height)], fill="black", width=1)

        elif style == 'hline':
            hy = random.randint(0, img_height - 1)
            draw.line([(0, hy), (img_width, hy)], fill="black", width=1)

    # Save image and label
    img_path = f"{dataset_dir}/images/{split}/{index}.jpg"
    label_path = f"{dataset_dir}/labels/{split}/{index}.txt"
    image.save(img_path)

    with open(label_path, "w") as f:
        for letter, box in hitboxes:
            if letter not in char_to_class:
                continue
            class_id = char_to_class[letter]
            x_c, y_c, w, h = bbox_to_yolo(box, img_width, img_height)
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


# === GENERATE DATA ===
all_fonts = get_all_fonts()
global_index = 0

print("🖋️ Generating dataset per font...")

for font_path in all_fonts:
    print(f"▶️ Using font: {os.path.basename(font_path)}")
    for i in tqdm(range(images_per_font), desc=f"{os.path.basename(font_path)[:20]}"):
        split = "train" if global_index < (len(all_fonts) * images_per_font * 0.8) else "val"
        generate_image(global_index, font_path, split)
        global_index += 1

print("✅ Dataset ready in 'datasets/'")
