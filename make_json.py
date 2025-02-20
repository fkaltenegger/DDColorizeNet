from pathlib import Path

data_root = Path("data/colorization/")

flickr_path = data_root / "flickr30k_images/"
coco_path = data_root / "coco/train2017"
vg_path = data_root / "VG_100K"

with open("prompts.json", "a") as f:
    for path in flickr_path.glob("*/.jpg"):
        f.write(f"{{\"image\": \"{path.name}\", \"prompt\": \"\"}}\n")
    
    for path in coco_path.glob("*/.jpg"):
        f.write(f"{{\"image\": \"{path.name}\", \"prompt\": \"\"}}\n")
    
    for path in vg_path.glob("*/.jpg"):
        f.write(f"{{\"image\": \"{path.name}\", \"prompt\": \"\"}}\n")
    