import json
import os
import shutil

# Paths for training data
train_coco_json = "FLIR_ADAS_v2/images_thermal_train/coco.json"
train_images_dir = "FLIR_ADAS_v2/images_thermal_train/data"

# Paths for validation data
val_coco_json = "FLIR_ADAS_v2/images_thermal_val/coco.json"
val_images_dir = "FLIR_ADAS_v2/images_thermal_val/data"

# Output directories for YOLO format
output_dir = "FLIR_ADAS_v2_yolo"
train_labels_dir = os.path.join(output_dir, "labels/train")
train_images_out_dir = os.path.join(output_dir, "images/train")
val_labels_dir = os.path.join(output_dir, "labels/val")
val_images_out_dir = os.path.join(output_dir, "images/val")

# Create directories if they don't exist
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(train_images_out_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(val_images_out_dir, exist_ok=True)

def convert_to_yolo(coco_json_path, images_dir, labels_dir, images_out_dir, category_map=None):
    """Convert COCO annotations to YOLO format."""
    # Load COCO annotations
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Create a category map if not provided
    if category_map is None:
        categories = coco_data["categories"]
        category_map = {cat["id"]: idx for idx, cat in enumerate(categories)}

    # Process images and annotations
    for image in coco_data["images"]:
        image_id = image["id"]
        file_name = image["file_name"]
        width = image["width"]
        height = image["height"]

        # Copy the image to the YOLO directory
        src_image_path = os.path.join(images_dir, file_name)
        dst_image_path = os.path.join(images_out_dir, file_name)
        shutil.copy(src_image_path, dst_image_path)

        # Prepare the YOLO annotation file
        annotation_file = os.path.splitext(file_name)[0] + ".txt"
        annotation_path = os.path.join(labels_dir, annotation_file)

        # Get annotations for this image
        annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
        yolo_annotations = []

        for ann in annotations:
            category_id = ann["category_id"]
            bbox = ann["bbox"]  # COCO format: [x_min, y_min, width, height]
            class_id = category_map[category_id]

            # Convert bbox to YOLO format: [center_x, center_y, width, height]
            x_min, y_min, box_width, box_height = bbox
            center_x = x_min + box_width / 2
            center_y = y_min + box_height / 2
            yolo_bbox = [
                center_x / width,
                center_y / height,
                box_width / width,
                box_height / height,
            ]
            yolo_annotations.append(f"{class_id} " + " ".join(map(str, yolo_bbox)))

        # Save YOLO annotations to file
        with open(annotation_path, "w") as f:
            f.write("\n".join(yolo_annotations))

    return category_map, coco_data["categories"]

# Convert training data
category_map, categories = convert_to_yolo(
    train_coco_json, train_images_dir, train_labels_dir, train_images_out_dir
)

# Convert validation data (use the same category map)
convert_to_yolo(
    val_coco_json, val_images_dir, val_labels_dir, val_images_out_dir, category_map
)

# Create the data.yaml file
num_classes = len(category_map)
class_names = [cat['name'] for cat in categories]
data_yaml = f"""
train: {os.path.abspath(train_images_out_dir)}
val: {os.path.abspath(val_images_out_dir)}

nc: {num_classes}
names: {class_names}
"""

# Save data.yaml
data_yaml_path = os.path.join(output_dir, "data.yaml")
with open(data_yaml_path, "w") as f:
    f.write(data_yaml)

print(f"YOLO dataset created at {output_dir}")