import csv
from pathlib import Path

import numpy as np


OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "gender_height_weight_100.csv"


def generate_samples(seed=42, n_female=50, n_male=50):
    rng = np.random.default_rng(seed)

    female_height = rng.normal(loc=162, scale=5.5, size=n_female)
    female_weight = rng.normal(loc=55, scale=6.5, size=n_female)
    male_height = rng.normal(loc=175, scale=6.5, size=n_male)
    male_weight = rng.normal(loc=70, scale=8.0, size=n_male)

    female_height = np.clip(female_height, 150, 175)
    female_weight = np.clip(female_weight, 42, 75)
    male_height = np.clip(male_height, 160, 190)
    male_weight = np.clip(male_weight, 50, 95)

    heights = np.concatenate([female_height, male_height])
    weights = np.concatenate([female_weight, male_weight])
    labels_text = np.concatenate([np.array(["女"] * n_female), np.array(["男"] * n_male)])
    labels_num = np.concatenate([np.zeros(n_female), np.ones(n_male)]).astype(int)

    indices = rng.permutation(len(heights))

    rows = []
    for index in indices:
        rows.append(
            [
                round(float(heights[index]), 1),
                round(float(weights[index]), 1),
                str(labels_text[index]),
                int(labels_num[index]),
            ]
        )

    return rows


def save_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["height_cm", "weight_kg", "label_text", "label_num"])
        writer.writerows(rows)


def main():
    rows = generate_samples()
    save_csv(rows, OUTPUT_PATH)

    print(f"Generated file: {OUTPUT_PATH}")
    print(f"Total samples: {len(rows)}")
    print("\nFirst 5 rows:")
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
