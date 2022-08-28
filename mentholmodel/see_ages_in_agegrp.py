import pandas as pd
import os

pop_file_name = os.path.join("..","..","Calibrated Population","Calibrated Population","PATH_Calibrate_18_64.xlsx")

pop_df = pd.read_excel(pop_file_name)

query = """
select distinct agegrp, age
"""

print(pop_df[["agegrp", "age"]].drop_duplicates().to_string(index=False))