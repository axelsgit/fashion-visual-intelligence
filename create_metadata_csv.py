import pandas as pd
from io import StringIO

# ==========================================================
# Dior metadata CSV
# ==========================================================
csv_data_dior = """collection_id,year,season,director,collection_name,era,location,category,folder_path,notes
54658,1998,FW,John Galliano,Dior FW1998 Ready-to-Wear,Galliano Era,Paris,Ready-to-Wear,data/FW1998_Galliano,Early theatrical fantasy
54690,2002,SS,John Galliano,Dior SS2002 Ready-to-Wear,Galliano Era,Paris,Ready-to-Wear,data/SS2002_Galliano,Exotic historical references
54672,2005,FW,John Galliano,Dior FW2005 Ready-to-Wear,Galliano Era,Paris,Ready-to-Wear,data/FW2005_Galliano,Baroque maximalism
54686,2007,SS,John Galliano,Dior SS2007 Ready-to-Wear,Galliano Era,Paris,Ready-to-Wear,data/SS2007_Galliano,Theatrical color explosion
54663,2010,FW,John Galliano,Dior FW2010 Ready-to-Wear,Galliano Era,Paris,Ready-to-Wear,data/FW2010_Galliano,Pre-departure refinement
54675,2012,FW,Raf Simons,Dior FW2012 Ready-to-Wear,Raf Simons Era,Paris,Ready-to-Wear,data/FW2012_Simons,Debut minimalist reinterpretation
32885,2013,SS,Raf Simons,Dior SS2013 Ready-to-Wear,Raf Simons Era,Paris,Ready-to-Wear,data/SS2013_Simons,Floral modern minimalism
38823,2014,FW,Raf Simons,Dior FW2014 Ready-to-Wear,Raf Simons Era,Paris,Ready-to-Wear,data/FW2014_Simons,Structured experimental silhouettes
41690,2015,FW,Raf Simons,Dior FW2015 Ready-to-Wear,Raf Simons Era,Paris,Ready-to-Wear,data/FW2015_Simons,Final futuristic couture
45903,2017,SS,Maria Grazia Chiuri,Dior SS2017 Ready-to-Wear,Maria Grazia Chiuri Era,Paris,Ready-to-Wear,data/SS2017_Chiuri,Feminist slogan debut
48945,2018,FW,Maria Grazia Chiuri,Dior FW2018 Ready-to-Wear,Maria Grazia Chiuri Era,Paris,Ready-to-Wear,data/FW2018_Chiuri,Feminine tailoring and empowerment
50493,2019,FW,Maria Grazia Chiuri,Dior FW2019 Ready-to-Wear,Maria Grazia Chiuri Era,Paris,Ready-to-Wear,data/FW2019_Chiuri,1950s-inspired silhouettes
54048,2021,SS,Maria Grazia Chiuri,Dior SS2021 Ready-to-Wear,Maria Grazia Chiuri Era,Paris,Ready-to-Wear,data/SS2021_Chiuri,Pandemic-era simplicity and nature
52527,2022,FW,Maria Grazia Chiuri,Dior FW2022 Ready-to-Wear,Maria Grazia Chiuri Era,Paris,Ready-to-Wear,data/FW2022_Chiuri,Tech x couture fusion
53106,2023,FW,Maria Grazia Chiuri,Dior FW2023 Ready-to-Wear,Maria Grazia Chiuri Era,Paris,Ready-to-Wear,data/FW2023_Chiuri,Reinterpretation of the Bar jacket
"""

meta_dior = pd.read_csv(StringIO(csv_data_dior))
meta_dior.to_csv("data/metadata_dior_womenswear.csv", index=False)

print("Dior metadata CSV created successfully.")
