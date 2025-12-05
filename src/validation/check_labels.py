import os
import csv
from PIL import Image

# Rutas
from src.config import FRAMES_VIDEO_DIR, ANNOTATIONS_YOLO_DIR, BASE_DIR

# Rutas
images_dir = FRAMES_VIDEO_DIR
labels_dir = ANNOTATIONS_YOLO_DIR

# Extensiones de imagen aceptadas
image_exts = {".jpg", ".jpeg", ".png"}

def run():
    # Listar imágenes y etiquetas
    images = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in image_exts}
    labels = {os.path.splitext(f)[0]: f for f in os.listdir(labels_dir) if f.endswith(".txt")}

    errores = []
    imagenes_etiquetadas = []
    imagenes_wEtiqueta = []

    # 1. Iteracion Imagenes 
    for img_name, img_file in images.items():
        if img_name not in labels:
            errores.append(f"Imagen sin etiqueta: {img_file}")
            imagenes_wEtiqueta.append(img_file)
        else:
            errores.append(f"Imagen con  etiqueta: {img_file}")
            imagenes_etiquetadas.append(img_file)


    # 2. Etiqueta sin imagen
    for label_name, label_file in labels.items():
        if label_name not in images:
            errores.append(f"Etiqueta sin imagen: {label_file}")


    # 3. Validar coordenadas YOLO
    for label_name, label_file in labels.items():
        img_path = os.path.join(images_dir, images.get(label_name, ""))
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path)
            w, h = img.size
            with open(label_path, "r") as f:
                for line_num, line in enumerate(f, start=1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        errores.append(f"Formato inválido en {label_file}, línea {line_num}")
                        continue
                    cls, x, y, bw, bh = map(float, parts)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                        errores.append(f"Coordenadas fuera de rango en {label_file}, línea {line_num}")
        except Exception as e:
            errores.append(f"Error abriendo {img_path}: {e}")

    # Reporte imágenes c/s etiqueta en CSV
    output_csv = BASE_DIR / "reports/imgsWoutAnnotations.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["imagen", "motivo"])
        for img in imagenes_etiquetadas:
            writer.writerow([img, "etiquetada"])
        for img in imagenes_wEtiqueta:
            writer.writerow([img, "sin etiqueta"])


    print(f"Reporte de imágenes sin etiqueta guardado en: {output_csv}")

if __name__ == "__main__":
    run()

