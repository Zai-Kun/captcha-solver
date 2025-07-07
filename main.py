import os
import random
import string

from PIL import Image, ImageDraw, ImageFont, ImageOps

# === CONFIG ===
img_width, img_height = 150, 50
font_size_start = 30
outline_width = 2
line_width = 3
line_count = 7
letters_per_image = 3
images_per_font = 10000

characters = string.ascii_letters + string.digits

fonts_dir = "fonts/"
image_fonts_dir = "fonts-images/"
dataset_dir = "datasets"

# === CREATE FOLDERS ===
os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)


# === FONT LOADERS ===
def get_all_fonts():
    fonts = []
    for root, _, files in os.walk(fonts_dir):
        for f in files:
            if f.lower().endswith((".ttf", ".otf")):
                fonts.append(os.path.join(root, f))
    return fonts


def get_image_fonts():
    image_fonts = []
    for root, dirs, _ in os.walk(image_fonts_dir):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if any(
                os.path.isfile(os.path.join(dir_path, f"{c}.png"))
                for c in characters
            ):
                image_fonts.append(dir_path)
    return image_fonts


def bbox_to_yolo(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height


# === DISTORTIONS ===
def add_noise(draw):
    used_y_positions = []
    for _ in range(line_count):
        style = random.choice(
            ["line", "curve", "squiggle", "dot", "rect", "blob", "vline", "hline"]
        )
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

        if style == "line":
            draw.line(
                [(x_start, y), (x_end, y + random.randint(-1, 1))],
                fill="black",
                width=line_width,
            )
        elif style == "curve":
            points = [
                (x_start, y + random.randint(-2, 2)),
                (x_start + length // 3, y + random.randint(-4, 4)),
                (x_start + 2 * length // 3, y + random.randint(-4, 4)),
                (x_end, y + random.randint(-2, 2)),
            ]
            draw.line(points, fill="black", width=line_width, joint="curve")
        elif style == "squiggle":
            step = 5
            amplitude = random.randint(3, 6)
            points = []
            for t in range(0, x_end - x_start, step):
                x = x_start + t
                y_offset = int(amplitude * random.uniform(-1, 1))
                points.append((x, y + y_offset))
            draw.line(points, fill="black", width=line_width)
        elif style == "dot":
            for _ in range(random.randint(5, 15)):
                draw.point(
                    (
                        random.randint(0, img_width - 1),
                        random.randint(0, img_height - 1),
                    ),
                    fill="black",
                )
        elif style == "rect":
            for _ in range(random.randint(1, 3)):
                rx1 = random.randint(0, img_width - 10)
                ry1 = random.randint(0, img_height - 10)
                rx2 = rx1 + random.randint(3, 10)
                ry2 = ry1 + random.randint(3, 10)
                draw.rectangle([rx1, ry1, rx2, ry2], outline="black", width=1)
        elif style == "blob":
            for _ in range(random.randint(1, 2)):
                bx = random.randint(0, img_width - 10)
                by = random.randint(0, img_height - 10)
                bw = random.randint(6, 12)
                bh = random.randint(6, 12)
                draw.ellipse([bx, by, bx + bw, by + bh], fill="black")
        elif style == "vline":
            vx = random.randint(0, img_width - 1)
            draw.line([(vx, 0), (vx, img_height)], fill="black", width=1)
        elif style == "hline":
            hy = random.randint(0, img_height - 1)
            draw.line([(0, hy), (img_width, hy)], fill="black", width=1)


# === SAVE FUNCTION ===
def save_sample(image, hitboxes, index, split):
    img_path = f"{dataset_dir}/images/{split}/{index}.jpg"
    label_path = f"{dataset_dir}/labels/{split}/{index}.txt"
    image.save(img_path)

    with open(label_path, "w") as f:
        for letter, box in hitboxes:
            class_id = characters.index(letter)
            x_c, y_c, w, h = bbox_to_yolo(box, img_width, img_height)
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


# === GENERATE FROM TTF FONT ===
def generate_image(index, font_path, split):
    letters = "".join(random.choices(characters, k=letters_per_image))
    font = ImageFont.truetype(font_path, random.randint(font_size_start, font_size_start + 5))
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    letter_bboxes = [font.getbbox(letter) for letter in letters]
    letter_widths = [bbox[2] - bbox[0] for bbox in letter_bboxes]
    total_letter_width = sum(letter_widths)
    padding = 10
    space_between = max(
        (img_width - total_letter_width - 2 * padding) // (len(letters) - 1), 0
    )

    x = padding
    hitboxes = []

    for letter in letters:
        bbox = font.getbbox(letter)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        y = (img_height - h) / 2 - bbox[1]
        # shadow
        draw.text((x+3, y+3), letter, font=font, fill="black")
        # is_hollow = random.choice([True, False])
        # if is_hollow:
        draw.text(
            (x, y),
            letter,
            font=font,
            fill="white",
            stroke_width=outline_width,
            stroke_fill="black",
        )
        # else:
        # draw.text((x, y), letter, font=font, fill="black")
        box = (int(x) -4, int(y + bbox[1])-4, int(x + w)+4, int(y + bbox[3])+4)
        hitboxes.append((letter, box))
        x += w + space_between

    if random.choice([True, False]):
        add_noise(draw)
    save_sample(image, hitboxes, index, split)


# === GENERATE FROM IMAGE FONT ===
def generate_image_from_image_font(index, font_dir, split):
    letters = "".join(random.choices(characters, k=letters_per_image))
    canvas = Image.new("RGB", (img_width, img_height), "white")
    x = 10
    hitboxes = []

    for letter in letters:
        path = os.path.join(font_dir, f"{letter}.png")
        if not os.path.exists(path):
            path = os.path.join(font_dir, f"{letter}.png")
        if not os.path.exists(path):
            continue
        glyph = Image.open(path).convert("RGBA")
        glyph = ImageOps.contain(glyph, (font_size_start, font_size_start))
        w, h = glyph.size
        y = (img_height - h) // 2
        canvas.paste(glyph, (x, y), glyph)
        hitboxes.append((letter, (x, y, x + w, y + h)))
        x += w + 5

    draw = ImageDraw.Draw(canvas)
    add_noise(draw)
    save_sample(canvas, hitboxes, index, split)


# === MAIN EXECUTION ===
all_fonts = get_all_fonts()
all_image_fonts = get_image_fonts()
global_index = 0
total_fonts =  len(all_fonts) # len(all_image_fonts)
split_threshold = int(total_fonts * images_per_font * 0.8)

print("🖋️ Generating from TTF fonts...")
for font_path in all_fonts:
    for _ in range(images_per_font):
        split = "train" if global_index < split_threshold else "val"
        generate_image(global_index, font_path, split)
        global_index += 1

# print("🖼️ Generating from image fonts...")
# for font_dir in all_image_fonts:
#     print(f"▶️ {os.path.basename(font_dir)}")
#     for _ in tqdm(range(images_per_font)):
#         split = "train" if global_index < split_threshold else "val"
#         generate_image_from_image_font(global_index, font_dir, split)
#         global_index += 1

# print("✅ Dataset ready in 'datasets/'")
