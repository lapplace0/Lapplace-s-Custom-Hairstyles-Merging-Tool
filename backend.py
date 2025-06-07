import os
import shutil
import json
import random
import uuid
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool, cpu_count, freeze_support
import zipfile
import logging

# === Logging Setup ===
logging.basicConfig(filename="log.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# === Constants ===
input_folder = "hairstyle_sheets"
temp_dir = "temp_hairstyles"
export_root = "Lapplace Custom Hairs Export"
hairs_dir = os.path.join(export_root, "hairs")
name_file = "names.txt"
template_json_path = "hair.json"
manifest_path = "manifest.json"
hairstyle_width = 16
hairstyle_height = 96
settings_path = "settings.json"
default_settings = {"agg_threshold": 0.11, "merge_threshold": 0.15, "grouping": True}

def load_settings():
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return default_settings.copy()

def save_settings(agg_val, merge_val, grouping_enabled):
    settings = {
        "agg_threshold": round(float(agg_val), 4),
        "merge_threshold": round(float(merge_val), 4),
        "grouping": grouping_enabled
    }
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

def generate_unique_id():
    uid = uuid.uuid4().hex[:12].upper()
    return f"lapplacech.{uid}.FS"

def shifted_ssim_distance(img1, img2, max_shift=2):
    def normalize(x): return (x - np.mean(x)) / (np.std(x) + 1e-8)
    img1, img2 = normalize(img1), normalize(img2)
    return 1 - max(ssim(img1, np.roll(img2, shift, axis=0), data_range=1.0) for shift in range(-max_shift, max_shift+1))

def compute_pair(args): 
    i, j, arrays = args
    return (i, j, shifted_ssim_distance(arrays[i], arrays[j]))

def compute_distance_matrix(paths, arrays, progress_callback):
    n = len(arrays)
    D = np.zeros((n, n))
    pairs = [(i, j, arrays) for i in range(n) for j in range(i+1, n)]

    freeze_support()  # Important for Windows exe
    with Pool(cpu_count()) as pool:
        for count, (i, j, d) in enumerate(pool.imap_unordered(compute_pair, pairs), 1):
            D[i, j] = D[j, i] = d
            if count % 100 == 0:
                progress_callback(count, len(pairs))
    return D

def image_to_array(path):
    img = Image.open(path).convert("RGBA").crop((0, 0, 16, 32))
    gray = img.convert("L")
    alpha = np.array(img.getchannel("A")) / 255.0
    arr = np.array(gray) / 255.0
    binary = (arr > 0.05) & (alpha > 0.1)
    return binary.astype(float)

def fetch_fantasy_names(count):
    with open(name_file, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) < count:
        names = (names * ((count // len(names)) + 1))[:count]
    random.shuffle(names)
    return names[:count]

def run_clustering(agg_thresh, merge_thresh, do_grouping, progress_callback, done_callback, mod_display_name, debug_mode=False):
    try:
        for path in [temp_dir, export_root]:
            if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(temp_dir)
        os.makedirs(hairs_dir, exist_ok=True)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        sheet_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]
        temp_paths = []
        for file in sheet_files:
            sheet = Image.open(os.path.join(input_folder, file)).convert("RGBA")
            for row in range(sheet.height // hairstyle_height):
                for col in range(sheet.width // hairstyle_width):
                    crop = sheet.crop((col*16, row*96, col*16+16, row*96+96))
                    if all(p[3] == 0 for p in crop.getdata()): continue
                    path = os.path.join(temp_dir, f"{file}_r{row}_c{col}.png")
                    crop.save(path)
                    temp_paths.append(path)

        arrays = [image_to_array(p) for p in temp_paths]
        if not do_grouping:
            groups = {i: [p] for i, p in enumerate(temp_paths)}
        else:
            D = compute_distance_matrix(temp_paths, arrays, progress_callback)
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=agg_thresh,
                                                 metric="precomputed", linkage="average").fit(D)
            raw_groups = defaultdict(list)
            for label, path in zip(clustering.labels_, temp_paths):
                raw_groups[label].append(path)

            groups = {}
            large_groups = {k: v for k, v in raw_groups.items() if len(v) >= 3}
            small_groups = {k: v for k, v in raw_groups.items() if len(v) < 3}

            for k, v in large_groups.items():
                groups[k] = v

            for k, v in small_groups.items():
                best_match = None
                best_score = float("inf")
                for lk, lv in large_groups.items():
                    dist = np.mean([
                        shifted_ssim_distance(image_to_array(p1), image_to_array(p2))
                        for p1 in v for p2 in lv
                    ])
                    if dist < merge_thresh and dist < best_score:
                        best_score = dist
                        best_match = lk
                if best_match is not None:
                    groups[best_match].extend(v)
                else:
                    groups[k] = v

        manifest_data["Description"] = f"Contains {len(groups)} hairstyle groups and {len(temp_paths)} total hairstyles."
        manifest_data["UniqueID"] = generate_unique_id()
        manifest_data["Name"] = f"[FS][LCH] {mod_display_name}"
        with open(os.path.join(export_root, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        group_names = fetch_fantasy_names(len(groups))
        sorted_labels = sorted(groups.keys())
        label_to_name = {label: group_names[i] for i, label in enumerate(sorted_labels)}

        for label, paths in groups.items():
            group = label_to_name[label]
            for idx, path in enumerate(paths, 1):
                variant = f"{group}_v{idx:02d}"
                mod_dir = os.path.join(hairs_dir, f"lch_{variant}")
                os.makedirs(mod_dir, exist_ok=True)
                shutil.copy(path, os.path.join(mod_dir, "hair.png"))
                with open(template_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["Name"] = f"LCH {group.replace('_', ' ').capitalize()} v{idx:02d}"
                with open(os.path.join(mod_dir, "hair.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

        if debug_mode:
            debug_dir = "grouped_hairstyles"
            if os.path.exists(debug_dir): shutil.rmtree(debug_dir)
            os.makedirs(debug_dir, exist_ok=True)
            for label, paths in groups.items():
                group_folder = os.path.join(debug_dir, label_to_name[label])
                os.makedirs(group_folder, exist_ok=True)
                for idx, path in enumerate(paths, 1):
                    shutil.copy(path, os.path.join(group_folder, f"{idx}.png"))

        # === Create ZIP of the export folder ===
        zip_path = os.path.join(os.getcwd(), "Lapplace_Custom_Hairs.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_root):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, start=export_root)
                    zipf.write(full_path, arcname=os.path.join("Lapplace Custom Hairs Export", rel_path))

        shutil.rmtree(temp_dir, ignore_errors=True)
        done_callback(True, f"\u2705 Finished clustering {len(temp_paths)} hairstyles into {len(groups)} groups.")
    except Exception as e:
        logging.exception("Error during clustering")
        done_callback(False, f"\u274C Error: {str(e)}")


# === Main guard for PyInstaller + multiprocessing safety ===
if __name__ == "__main__":
    freeze_support()
    print("This module is intended to be imported by the UI, not run directly.")
