import pandas as pd

from starmie.sdd.augment import augment
df = pd.read_csv("datasets/WDC/Test/T2DV2_253.csv")
augment(df,"highlight_cells")
augment(df,"replace_high_cells")