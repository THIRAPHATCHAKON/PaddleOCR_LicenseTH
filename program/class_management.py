import requests

API_KEY = "tZlBsoX5SJChAioww39H"  # ใส่ API Key ของคุณ
WORKSPACE = "new-workspace-wnmam"
PROJECT = "model-pl3ai"

CLASSES_TO_REMOVE = ["plate_number", "province"]

def search_images(class_name, offset=0):
    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/search"
    res = requests.post(url, params={"api_key": API_KEY}, json={
        "class_name": class_name,
        "in_dataset": True,
        "offset": offset,
        "limit": 100,
        "fields": ["id", "name"]
    })
    return res.json()

def get_image_detail(image_id):
    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/images/{image_id}"
    res = requests.get(url, params={"api_key": API_KEY})
    return res.json()

def boxes_to_xml(boxes, image_name, width, height):
    xml = f"""<annotation>
    <filename>{image_name}</filename>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>"""
    for b in boxes:
        x, y, w, h = float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"])  # แปลงเป็น float
        xmin = int(x - w / 2)
        ymin = int(y - h / 2)
        xmax = int(x + w / 2)
        ymax = int(y + h / 2)
        xml += f"""
    <object>
        <name>{b["label"]}</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>"""
    xml += "\n</annotation>"
    return xml

def upload_annotation(image_id, image_name, xml_content):
    xml_filename = image_name.rsplit(".", 1)[0] + ".xml"
    url = f"https://api.roboflow.com/dataset/{PROJECT}/annotate/{image_id}"
    res = requests.post(url, params={"api_key": API_KEY, "name": xml_filename, "overwrite": "true"},  # เพิ่ม overwrite=true
                        data=xml_content, headers={"Content-Type": "text/plain"})
    return res.status_code, res.json()

# ดึง image id ทั้งหมด
all_image_ids = {}
for class_name in CLASSES_TO_REMOVE:
    print(f"ค้นหารูปที่มี class: {class_name}")
    offset = 0
    while True:
        data = search_images(class_name, offset)
        results = data.get("results", [])
        if not results:
            break
        for img in results:
            all_image_ids[img["id"]] = img.get("name", img["id"])
        if len(results) < 100:
            break
        offset += 100

print(f"\nพบรูปทั้งหมด {len(all_image_ids)} รูป\n")

total_updated = 0

for image_id, image_name in all_image_ids.items():
    detail = get_image_detail(image_id)
    ann = detail.get("image", {}).get("annotation", {})
    boxes = ann.get("boxes", [])
    width = ann.get("width", 640)
    height = ann.get("height", 640)

    filtered = [b for b in boxes if b.get("label") not in CLASSES_TO_REMOVE]
    removed = len(boxes) - len(filtered)

    if removed > 0:
        xml = boxes_to_xml(filtered, image_name, width, height)
        status, res = upload_annotation(image_id, image_name, xml)
        if status == 200:
            total_updated += 1
            print(f"✅ {image_name} → ลบ {removed} annotation")
        else:
            print(f"❌ {image_name} → {status}: {res}")

print(f"\nOk! อัปเดต {total_updated} รูป")