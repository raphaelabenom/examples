# %%
import pandas as pd

# %%

df_censo = pd.read_csv(
    "./data/bronze/microdados_ed_basica_2024.csv",
    sep=";",
    encoding="Latin-1",
)
# %%
df_censo = df_censo[:50000]

df_censo.to_csv("senso_escolar_com_100mil_registros.csv", index=False)

# %%
df_censo.head()

# %%
for col in df_censo.columns:
    print(col)
# %%

de = df_censo["CO_ENTIDADE"] = 42103770
print(de)

# %%

df_enem = pd.read_csv(
    "./data/bronze/RESULTADOS_2024.csv",
    sep=";",
    encoding="Latin-1",
)

# %%

selecionados = df_enem[:50000]

selecionados.to_csv("out.csv", index=False)

# %%
for col in df_enem.columns:
    print(col)

# %%
df_enem = pd.read_csv(
    "./data/bronze/RESULTADOS_2024.csv",
    sep=";",
    encoding="Latin-1",
)

# %%

df_participantes = pd.read_csv(
    "./data/processed/DADOS/PARTICIPANTES_2024.csv",
    sep=";",
    encoding="Latin-1",
)

# %%
df_participantes.head()
