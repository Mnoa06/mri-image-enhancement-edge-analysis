from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.metrics import structural_similarity
from skimage.filters import sobel
from skimage.feature import canny
from skimage import exposure
import json
import sqlite3

def calculate_metrics(reference, processed):
    mse = float(np.mean((processed - reference) ** 2))

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(10 * np.log10(1.0 / mse))

    ssim = float(structural_similarity(
        reference, processed, data_range=1.0
    ))

    return mse, psnr, ssim

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print(PROJECT_ROOT)

SERIES_FOLDER = PROJECT_ROOT / "data" / "original" / "brain-mri" / "flair"

print(SERIES_FOLDER)
print("Folder exists:", SERIES_FOLDER.is_dir())


files = sorted(path for path in SERIES_FOLDER.rglob("*") if path.is_file())

print("Number of files:", len(files))

for path in files[:5]:
    print(path.name)

# Here’s how it works:
# .rglob("*") searches the folder and its subfolders.
# .is_file() keeps files and excludes folders.
# sorted() puts the paths in filename order—not anatomical slice order.
# len(files) counts the files.
# files[:5] selects the first five.
# path.name prints just the filename.


# first_file = files[0]
middle_index = len(files) // 2
first_file = files[middle_index]

print("Selected file:", first_file.name)
dataset = pydicom.dcmread(first_file,stop_before_pixels=True)
# files[0] selects the first file because Python indexes start at zero.
# dcmread() reads the DICOM into a dataset object.
# stop_before_pixels=True reads the header without loading the pixel data.

print("Modality:", dataset.get("Modality", "Not provided"))
print("Series:", dataset.get("SeriesDescription", "Not provided"))
print("Rows:", dataset.get("Rows", "Not provided"))
print("Columns:", dataset.get("Columns", "Not provided"))\

#.get() retrieves a metadata value, returning "Not provided" 
# if it’s missing. Rows describe image height; columns describe width.

dataset = pydicom.dcmread(first_file)
image = dataset.pixel_array
#This time, we omit stop_before_pixels=True so the pixel data is included.
#  pixel_array decodes that data into a NumPy array—a grid of intensity values.

image_float = image.astype("float32")

minimum = image_float.min()
maximum = image_float.max()
if maximum > minimum:
    image_normalized = (image_float - minimum) / (maximum - minimum)
else:
    image_normalized = image_float * 0


print("Normalized minimum:", image_normalized.min())
print("Normalized maximum:", image_normalized.max())
print("processing data type:", image_float.dtype)
#.astype("float32") creates a copy that can hold decimal values.
#  This will let us normalize 
# intensities and apply filters while keeping your original image unchanged.

print("Image shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum intensity:", image.min())
print("Maximum intensity:", image.max())

#The decoder worked—you now have the MRI pixels loaded.
#Your image contains 288 rows × 288 columns, stored as uint16 
# (unsigned 16-bit integers). This slice’s values range from 0 to 536;
#  uint16 can hold values up to 65,535, but an image doesn’t have to use that entire range.

#examine its histogram. how many pixels fall into each intensity range.


# plt.imshow(image,cmap="gray") #Replaced
plt.imshow(image_normalized, cmap="gray", vmin=0, vmax=1)
#vmin=0 and vmax=1 fix the display scale: zero appears black and one appears white.
# plt.title("FLAIR - first file")
# plt.colorbar(label="Stored pixel intensity")
plt.title("Normalized FLAIR")
plt.colorbar(label="Normalized intensity")

# imshow() displays your pixel array.
# cmap="gray" maps lower values to darker shades and higher values to brighter shades.
# colorbar() shows the intensity scale.


#creates a new image with locally adjusted contrast.
# clip_limit=0.01 controls how much contrast amplification is allowed. 
# Higher values generally permit stronger enhancement.
# Unlike smoothing, CLAHE changes contrast rather than removing noise.
# It also differs from window/level: window/level applies one display mapping across the image, while CLAHE adjusts the mapping locally.
image_clahe = exposure.equalize_adapthist(
    image_normalized,
    clip_limit=0.01
)

rng = np.random.default_rng(42)
noise = rng.normal(loc=0, scale=0.03,size=image_normalized.shape)
# 42 is a seed: it makes the noise repeatable.
# loc=0 sets the average noise to zero.
# scale=0.03 sets its standard deviation to 3% of our normalized intensity range.
# size generates one noise value for each pixel.
print("Noise shape:", noise.shape)
print("Noise mean:", noise.mean())
print("Noise standard deviation:", noise.std())

image_noisy = image_normalized + noise
#This adds each noise value to its corresponding pixel. 
# Some values may fall below zero or above one. 
# For this experiment, keep those values so we preserve the noise we generated.

mse_noisy, psnr_noisy, ssim_noisy = calculate_metrics(
    image_normalized, image_noisy
)

image_gaussian = gaussian_filter(image_noisy, sigma=1.0)
#sigma=1.0 sets the smoothing width to one pixel. Larger values produce stronger smoothing.

mse_gaussian, psnr_gaussian, ssim_gaussian = calculate_metrics(
    image_normalized, image_gaussian
)

image_gaussian_light = gaussian_filter(image_noisy, sigma=0.5)
#sigma=0.5 uses a narrower 
# Gaussian kernel than sigma=1.0, so it blends neighboring pixels less strongly.

mse_light, psnr_light, ssim_light = calculate_metrics(
    image_normalized, image_gaussian_light
)

# mse_noisy = np.mean((image_noisy - image_normalized) **2)
# mse_gaussian = np.mean((image_gaussian - image_normalized) ** 2)

#Each line:
# 1. Subtracts the reference from the comparison image.
# 2. Squares each difference.
# 3. Averages those squared differences across all pixels.
# Lower MSE means a closer match to the reference. Our reference is the MRI 
# before adding simulated noise; it still contains its original acquisition noise.
# psnr_noisy = 10 * np.log10(1.0 / mse_noisy)
# psnr_gaussian = 10 * np.log10(1.0 / mse_gaussian)
#PSNR, which expresses the error on a decibel scale. 
# Higher PSNR means closer agreement with our original image

# ssim_light =structural_similarity(
#     image_normalized, image_gaussian_light, data_range=1.0
# )
# ssim_gaussian = structural_similarity(
#     image_normalized,image_gaussian, data_range=1.0
# )

#This compares the image before filtering with our original reference
# ssim_noisy = structural_similarity(image_normalized, image_noisy, data_range=1.0
# )


#size=3 selects a 3 × 3 neighborhood containing nine pixels. 
# The filter sorts their intensities and uses the fifth value.
image_median = median_filter(image_noisy, size=3)

mse_median, psnr_median, ssim_median = calculate_metrics(
    image_normalized, image_median
)

#The Sobel filter measures changes in intensity. Strong
#  changes produce brighter values in the edge image, highlighting boundaries.
edges_gaussian = sobel(image_gaussian_light)

edges_noisy = sobel(image_noisy)

#Canny, which produces a binary map: white pixels mark detected edges,
#  and black pixels mark everything else.
#Canny applies its own Gaussian smoothing. The high threshold selects strong edges;
#  the low threshold allows weaker edges connected to them.
edges_canny = canny(
    image_noisy,
    sigma=1.0,
    low_threshold=0.1,
    high_threshold=0.2
)

edges_canny_sensitive = canny(
    image_noisy,
    sigma=1.0,
    low_threshold=0.05,
    high_threshold=0.1
)

print(f"Noisy SSIM: {ssim_noisy:4f}")


print(f"Gaussian sigma 0.5 SSIM: {ssim_light:.4f}")
print(f"Gaussian sigma 1.0 SSIM: {ssim_gaussian:.4f}")

print(f"Light Gaussian SSM: {ssim_light:.4f}")
print(f"Gaussian SSIM: {ssim_gaussian:.4f}")
#data_range=1.0 tells SSIM that our reference image uses the normalized intensity range from 0 to 1.

print(f"Noisy MSE: {mse_noisy:.6f}")
print(f"Gaussian MSE: {mse_gaussian:.6f}")

print(f"Noisy PSNR: {psnr_noisy:.2f} dB")
print(f"Gaussian PSNR: {psnr_gaussian:.2f} dB")

#PSNR and SSIM measure how closely a processed image matches a reference image.
#  In our project, the reference is your original MRI before we added simulated noise.

#Peak SNR, or PSNR, measures similarity using pixel intensity errors.
# It comes directly from MSE: smaller errors produce higher PSNR.
#Measured in decibels (dB).
#Higher is better when matching the reference.
#Different than MRI SNR

#SSIM — Structural Similarity Index
#SSIM compares small regions of the images, looking at:
# - Brightness: average intensity.
# - Contrast: variation in intensity.
# - Structure: how local intensity patterns correspond.
#Closer to 1 means more similar, with 1 indicating an identical match.


#MSE - How large are the pixel errors?
#PSNR How small are those errors, expressed db?
#SSIM How similar are the local image patterns?


# mse_median = np.mean((image_median - image_normalized) ** 2)
# psnr_median = 10 * np.log10(1.0 / mse_median)
# ssim_median = structural_similarity(
#     image_normalized, image_median, data_range=1.0
# )

print(f"Median MSE: {mse_median:.6f}")
print(f"Median PSNR: {psnr_median:.2f} dB")
print(f"Median SSIM: {ssim_median:.4f}")


# mse_light = np.mean((image_gaussian_light - image_normalized) ** 2)
# psnr_light = 10 * np.log10(1.0 / mse_light)

print(f"Light Gaussian MSE: {mse_light:.6f}")
print(f"Light Gaussian PSNR: {psnr_light:.2f} dB")

plt.figure()
plt.hist(image.ravel(), bins=100)
plt.yscale("log")
plt.title("FLAIR intensity histogram")
plt.xlabel("Stored pixel intensity")
# plt.ylabel("Number of pixels")
plt.ylabel("Number of pixels (log scale)")
#A logarithmic scale spaces counts such as 10, 100, 1,000, and 10,000 evenly. 
# This makes both large and small counts visible.
#  It changes only the graph’s scale—the MRI values stay the same.


#Here’s what the new parts do:
# plt.figure() creates a separate figure.
# image.ravel() turns the 288 × 288 grid into a one-dimensional sequence of 82,944 pixel values.
# bins=100 groups those values into 100 intensity intervals.

plt.figure()
plt.imshow(image_noisy, cmap="gray", vmin=0, vmax=1)
plt.title("FLAIR with simulated Gaussian noise")
plt.colorbar(label="Normalized intensity")

plt.figure()
plt.imshow(image_gaussian, cmap="gray", vmin=0, vmax=1)
plt.title("Gaussian smoothing — sigma 1.0")
plt.colorbar(label="Normalized intensity")

# fig, axes = plt.subplots(1,3,figsize=(12,4))
filter_fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image_normalized,cmap="gray",vmin=0,vmax=1)
axes[0].set_title("Original")

axes[1].imshow(image_gaussian_light,cmap="gray",vmin=0,vmax=1)
axes[1].set_title("Gaussian: sigma 0.5")

# axes[2].imshow(image_gaussian,cmap="gray",vmin=0,vmax=1)
# axes[2].set_title("gausssian:sigma 1.0")
axes[2].imshow(image_median, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Median: 3 × 3")
plt.tight_layout()

plt.figure()
plt.imshow(edges_gaussian, cmap="gray")
plt.title("Sobel edges — Gaussian sigma 0.5")
plt.colorbar(label="Edge strength")


#edge_max gives both panels the same brightness scale, making the comparison fair.
edge_max = max(edges_noisy.max(),edges_gaussian.max())

# fig,axes = plt.subplots(1,2, figsize=(10,5))
sobel_fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(edges_noisy, cmap="gray", vmin=0, vmax=edge_max)
axes[0].set_title("Sobel — noisy image")

axes[1].imshow(edges_gaussian, cmap="gray", vmin=0, vmax=edge_max)
axes[1].set_title("Sobel — Gaussian sigma 0.5")

plt.tight_layout()

plt.figure()
plt.imshow(edges_canny, cmap="gray")
plt.title("Canny edges — sigma 1.0")


#We halved both thresholds, allowing weaker intensity changes to qualify as edges. 
# This can reveal more detail, but also more noise.
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

canny_fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(edges_canny, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Canny — thresholds 0.1 / 0.2")

axes[1].imshow(edges_canny_sensitive, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Canny — thresholds 0.05 / 0.1")

plt.tight_layout()

# fig, axes = plt.subplots(1,2, figsize=(10,5))
clahe_fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image_normalized, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original")

axes[1].imshow(image_clahe, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("CLAHE — clip limit 0.01")

plt.tight_layout()

#What: Create an output folder and save a PNG.
# Why: The plot window is temporary; 
# a saved image lets us review the result later and document the project.

OUTPUT_FOLDER = PROJECT_ROOT/ "outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

comparison_figures = {
    "comparison_filters": filter_fig,
    "comparison_sobel": sobel_fig,
    "comparison_canny": canny_fig
}

for name, figure in comparison_figures.items():
    figure.savefig(
        OUTPUT_FOLDER / f"{name}.png",
        dpi=180,
        bbox_inches="tight"
    )
    print("Saved comparison:", name)


clahe_path = OUTPUT_FOLDER / "flair_clahe.png"

plt.imsave(
    clahe_path,
    image_clahe,
    cmap="gray",
    vmin=0,
    vmax=1
)

print("Saved CLAHE image:", clahe_path)

#Save the numerical array. .npy file. PNG stores a display representation.
#  .npy preserves the intensity values and data type for later processing.
clahe_array_path = OUTPUT_FOLDER / "flair_clahe.npy"

#np.save() writes the array to disk. Later, np.load() can read it back.
np.save(clahe_array_path, image_clahe)

print("Saved CLAHE array:", clahe_array_path)

#dpi=180 sets the export resolution. 
# bbox_inches="tight" trims extra space around the figure.

comparison_path = OUTPUT_FOLDER / "comparison_clahe.png"

clahe_fig.savefig(
    comparison_path,
    dpi=180,
    bbox_inches="tight"
)

print("Saved comparison:", comparison_path)


#JSON
metrics_to_save = {
    "noisy": {
        "noise_std": 0.03,
        "seed": 42,
        "mse": float(mse_noisy),
        "psnr": float(psnr_noisy),
        "ssim": float(ssim_noisy)
    },
    "gaussian_0_5": {
        "sigma": 0.5,
        "mse": float(mse_light),
        "psnr": float(psnr_light),
        "ssim": float(ssim_light)
    },
    "gaussian_1_0": {
        "sigma": 1.0,
        "mse": float(mse_gaussian),
        "psnr": float(psnr_gaussian),
        "ssim": float(ssim_gaussian)
    },
    "median_3": {
        "size": 3,
        "mse": float(mse_median),
        "psnr": float(psnr_median),
        "ssim": float(ssim_median)
    }
}

metrics_path = OUTPUT_FOLDER / "filter_metrics.json"

metrics_path.write_text(
    json.dumps(metrics_to_save, indent=4),
    encoding="utf-8"
)

print("Saved metrics:", metrics_path)

results_to_save = {
    "flair_original": image_normalized,
    "flair_noisy": image_noisy,
    "flair_gaussian_0_5": image_gaussian_light,
    "flair_gaussian_1_0": image_gaussian,
    "flair_median_3": image_median,
    "flair_sobel": edges_gaussian,
    "flair_canny": edges_canny,
    "flair_canny_sensitive": edges_canny_sensitive
}

for name, array in results_to_save.items():
    output_path = OUTPUT_FOLDER / f"{name}.npy"
    np.save(output_path, array)
    print("Saved array:", output_path.name)

DATABASE_PATH = PROJECT_ROOT / "database" / "mri_processing.db"

print("Database exists:", DATABASE_PATH.is_file())

if DATABASE_PATH.is_file():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()

        print("Database tables:", tables)
        for table_name in ("images", "processing_results"):
            columns = connection.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()

            print(f"\n{table_name} columns:")
            for column in columns:
                print(column[1], column[2])

        source_path = str(first_file.resolve())

        connection.execute(
            """
            INSERT INTO images
                (filename, file_path, width, height, dtype)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO NOTHING
            """,
            (
                first_file.name,
                source_path,
                int(image.shape[1]),
                int(image.shape[0]),
                str(image.dtype)
            )
        )

        image_id = connection.execute(
            "SELECT image_id FROM images WHERE file_path = ?",
            (source_path,)
        ).fetchone()[0]

        print("Registered image ID:", image_id)

        gaussian_path = OUTPUT_FOLDER / "flair_gaussian_0_5.npy"

        parameters = {
            "sigma": 0.5,
            "noise_std": 0.03,
            "seed": 42
        }

        connection.execute(
            """
            INSERT INTO processing_results
                (image_id, method, parameters_json,
                 output_path, psnr, ssim)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(output_path) DO UPDATE SET
                parameters_json = excluded.parameters_json,
                psnr = excluded.psnr,
                ssim = excluded.ssim
            """,
            (
                image_id,
                "gaussian",
                json.dumps(parameters),
                str(gaussian_path.resolve()),
                float(psnr_light),
                float(ssim_light)
            )
        )

        print("Recorded Gaussian sigma 0.5 result")
        saved_result = connection.execute(
            """
            SELECT image_id, method, psnr, ssim
            FROM processing_results
            WHERE output_path = ?
            """,
            (str(gaussian_path.resolve()),)
        ).fetchone()

        print("Verified database result:", saved_result)

        other_results = [
            (
                "gaussian",
                {"sigma": 1.0, "noise_std": 0.03, "seed": 42},
                "flair_gaussian_1_0.npy",
                psnr_gaussian,
                ssim_gaussian
            ),
            (
                "median",
                {"size": 3, "noise_std": 0.03, "seed": 42},
                "flair_median_3.npy",
                psnr_median,
                ssim_median
            )
        ]

        for method, parameters, filename, psnr, ssim in other_results:
            result_path = OUTPUT_FOLDER / filename

            connection.execute(
                """
                INSERT INTO processing_results
                    (image_id, method, parameters_json,
                     output_path, psnr, ssim)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(output_path) DO UPDATE SET
                    parameters_json = excluded.parameters_json,
                    psnr = excluded.psnr,
                    ssim = excluded.ssim
                """,
                (
                    image_id,
                    method,
                    json.dumps(parameters),
                    str(result_path.resolve()),
                    float(psnr),
                    float(ssim)
                )
            )

            print("Recorded filter:", filename)
        analysis_results = [
            ("clahe", {"clip_limit": 0.01},
                "flair_clahe.npy"),

            ("sobel", {"input": "gaussian_sigma_0.5"},
                "flair_sobel.npy"),

            ("canny", {"sigma": 1.0, "low_threshold": 0.1,
                        "high_threshold": 0.2},
                "flair_canny.npy"),

            ("canny", {"sigma": 1.0, "low_threshold": 0.05,
                        "high_threshold": 0.1},
                "flair_canny_sensitive.npy")
        ]

        for method, parameters, filename in analysis_results:
            result_path = OUTPUT_FOLDER / filename

            connection.execute(
                """
                INSERT INTO processing_results
                    (image_id, method, parameters_json,
                        output_path, psnr, ssim)
                VALUES (?, ?, ?, ?, NULL, NULL)
                ON CONFLICT(output_path) DO UPDATE SET
                    parameters_json = excluded.parameters_json,
                    psnr = NULL,
                    ssim = NULL
                """,
                (
                    image_id,
                    method,
                    json.dumps(parameters),
                    str(result_path.resolve())
                )
            )

with sqlite3.connect(DATABASE_PATH) as check_connection:
    rows = check_connection.execute(
        """
        SELECT method, parameters_json, psnr, ssim
        FROM processing_results
        WHERE image_id = ?
        ORDER BY result_id
        """,
        (image_id,)
    ).fetchall()

    print("Stored filter records:", len(rows))

    for row in rows:
        print(row)



plt.show()

# show() opens the figure.
# Save and run: