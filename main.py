import os
import random
import string
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# === CONFIG ===
img_width, img_height = 150, 50
font_size = 36
outline_width = 2
line_width = 2
line_count = 2
letters_per_image = 200
images_per_font = 5
border_padding = 15       # padding from image edges
letter_spacing = 5       # spacing between letters

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
                fonts.append(os.path.join(root, f))
    return fonts

def bbox_to_yolo(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def boxes_overlap(b1, b2, margin=0):
    x1, y1, x2, y2 = b1
    a1, b1_, a2, b2_ = b2
    return not (x2 + margin <= a1 or x1 - margin >= a2 or y2 + margin <= b1_ or y1 - margin >= b2_)

def generate_image(index, font_path, split):
    letters = ''.join(random.choices(characters, k=letters_per_image))
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    placed_boxes = []
    hitboxes = []

    for letter in letters:
        placed = False
        for _ in range(100):
            font_size_rand = random.randint(int(font_size * 0.8), int(font_size * 1.1))
            font = ImageFont.truetype(font_path, font_size_rand)
            bbox = font.getbbox(letter)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            safe_x1 = 2
            safe_y1 = 2
            safe_x2 = img_width - w - border_padding
            safe_y2 = img_height - h - border_padding

            if safe_x2 < safe_x1 or safe_y2 < safe_y1:
                continue  # letter too big

            x = random.randint(safe_x1, safe_x2)
            y = random.randint(safe_y1, safe_y2)

            box = (x, y, x + w, y + h)

            if any(boxes_overlap(box, existing, margin=letter_spacing) for existing in placed_boxes):
                continue  # overlap

            placed_boxes.append(box)

            # Draw letter (no shadow)
            is_hollow = random.choice([True, False])
            if is_hollow:
                draw.text((x, y), letter, font=font, fill="white", stroke_width=outline_width, stroke_fill="black")
            else:
                draw.text((x, y), letter, font=font, fill="black")

            # Bounding box with padding
            pad = 3
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_width, x + w + pad)
            y2 = min(img_height, y + h + pad)

            hitboxes.append((letter.lower(), (x1, y1, x2, y2)))
            placed = True
            break

        if not placed:
            print(f"⚠️ Skipped letter '{letter}' (no place to fit)")
            continue

    # === Distortions (Reduced) ===
    for _ in range(line_count):
        style = random.choice(['line', 'dot', 'blob'])
        x_start = random.randint(0, img_width // 4)
        length = random.randint(img_width // 2, img_width - 10)
        x_end = min(x_start + length, img_width)
        y = random.randint(0, img_height - 1)

        if style == 'line':
            draw.line([(x_start, y), (x_end, y + random.randint(-1, 1))], fill="black", width=line_width)

        elif style == 'dot':
            for _ in range(random.randint(3, 6)):
                dot_x = random.randint(0, img_width - 1)
                dot_y = random.randint(0, img_height - 1)
                draw.point((dot_x, dot_y), fill="black")

        elif style == 'blob':
            for _ in range(random.randint(1, 2)):
                bx = random.randint(0, img_width - 8)
                by = random.randint(0, img_height - 8)
                bw = random.randint(4, 6)
                bh = random.randint(4, 6)
                draw.ellipse([bx, by, bx + bw, by + bh], fill="black")

    # === Save image and label ===
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

# === GENERATE DATASET ===
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

