from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch

# Encontrar automáticamente datos_VAR.xlsx
archivos = list(Path.cwd().rglob("datos_VAR.xlsx"))
carpeta = archivos[0].parent
os.chdir(carpeta)

# Puntos 1, 2 y 3: Cargar datos y ordenar (primero y_t más exógena, luego x_t)
datos = pd.read_excel("datos_VAR.xlsx")
col_y = [c for c in datos.columns if "y" in c.lower()][0]
col_x = [c for c in datos.columns if "x" in c.lower()][0]
datos = datos[[col_y, col_x]]

# Punto 4: info() y head()
print("\n=== PUNTO 4: INFO Y HEAD ===")
print(datos.info())
print(datos.head())

# Punto 5: Índice temporal trimestral desde 1990Q1
datos.index = pd.period_range(start="1990Q1", periods=len(datos), freq="Q")
print("\n=== PUNTO 5: CON INDICE TEMPORAL ===")
print(datos.head())

# Punto 6: Gráfica de las series
datos.plot(subplots=True, figsize=(10, 6), title="Series de tiempo yt y xt")
plt.tight_layout()
plt.savefig("grafica_punto6_series.png", dpi=150)
plt.close()

# Puntos 7 y 8: Pruebas de Dickey-Fuller
print("\n=== PUNTOS 7 Y 8: TEST DICKEY-FULLER ===")
for col in [col_y, col_x]:
    res_adf = adfuller(datos[col])
    dec = "RECHAZA H0 (Es estacionaria)" if res_adf[1] < 0.05 else "NO RECHAZA H0 (No estacionaria)"
    print(f"Serie {col}: t-valor = {res_adf[0]:.4f} | p-valor = {res_adf[1]:.6f} -> {dec}")

# Punto 9: Rezagos óptimos
print("\n=== PUNTO 9: REZAGOS OPTIMOS ===")
modelo = VAR(datos)
print(modelo.select_order(maxlags=8).summary())

# Puntos 10 a 14: Estimación VAR(1), raíces, matriz A1, resumen y ecuaciones
res = modelo.fit(1)
print("\n=== PUNTO 11: RAICES Y ESTACIONARIEDAD ===")
print("Raices:", res.roots)
print("Modulo (valor absoluto):", np.abs(res.roots))
print("¿Es estacionario/estable?:", res.is_stable())

print("\n=== PUNTO 12: MATRIZ A1 ===")
print(res.coefs[0])

print("\n=== PUNTO 13: RESUMEN COMPLETO VAR(1) ===")
print(res.summary())

print("\n=== PUNTO 14: ECUACIONES ESTIMADAS ===")
c = res.params.loc["const"]
a = res.coefs[0]
print(f"{col_y}_t = {c[col_y]:.4f} + ({a[0,0]:.4f})*{col_y}_(t-1) + ({a[0,1]:.4f})*{col_x}_(t-1)")
print(f"{col_x}_t = {c[col_x]:.4f} + ({a[1,0]:.4f})*{col_y}_(t-1) + ({a[1,1]:.4f})*{col_x}_(t-1)")

# Puntos 15, 16 y 17: Diagnóstico de residuales
print("\n=== PUNTO 15: CORRELACION SERIAL ===")
print(res.test_whiteness(nlags=10).summary())

print("\n=== PUNTO 16: HETEROCEDASTICIDAD (ARCH) ===")
for col in [col_y, col_x]:
    arch = het_arch(res.resid[col])
    dec = "Hay heterocedasticidad (Rechaza H0)" if arch[1] < 0.05 else "NO hay heterocedasticidad (No rechaza H0)"
    print(f"Residuales {col}: Estadistico = {arch[0]:.4f} | p-valor = {arch[1]:.6f} -> {dec}")

print("\n=== PUNTO 17: NORMALIDAD ===")
print(res.test_normality().summary())

# Puntos 18, 19 y 20: Pronóstico 12 pasos adelante
print("\n=== PUNTOS 18, 19 Y 20: PRONOSTICOS (12 PASOS) ===")
pron, inf, sup = res.forecast_interval(y=datos.values[-1:], steps=12, alpha=0.05)
pasos = range(1, 13)
df_pron = pd.DataFrame(pron, columns=[col_y, col_x], index=pasos)
df_inf = pd.DataFrame(inf, columns=[col_y, col_x], index=pasos)
df_sup = pd.DataFrame(sup, columns=[col_y, col_x], index=pasos)
print("Pronostico puntual:\n", df_pron)
print("\nLimite inferior (95%):\n", df_inf)
print("\nLimite superior (95%):\n", df_sup)

ancho_medio_x3 = (df_sup.loc[3, col_x] - df_inf.loc[3, col_x]) / 2
lim_sup_y5 = df_sup.loc[5, col_y]
print(f"\nPUNTO 19 - Ancho medio intervalo x (paso 3): {ancho_medio_x3:.6f}")
print(f"PUNTO 20 - Limite superior intervalo y (paso 5): {lim_sup_y5:.6f}")

# Puntos 21, 22 y 23: IRFs y gráficas a 18 pasos
print("\n=== PUNTOS 21 Y 22: PRIMERAS 10 IRFs ===")
irf10 = res.irf(10)
print("21. NO ortogonalizadas (t=0 a t=10):\n", irf10.irfs)
print("\n22. ORTOGONALIZADAS (t=0 a t=10):\n", irf10.orth_irfs)

irf18 = res.irf(18)
fig1 = irf18.plot(orth=False)
fig1.savefig("grafica_punto23_IRF_no_ortogonalizada.png", dpi=150, bbox_inches="tight")
plt.close(fig1)

fig2 = irf18.plot(orth=True)
fig2.savefig("grafica_punto23_IRF_ortogonalizada.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

print("\n¡TERMINADO! Las 3 graficas quedaron guardadas en:", carpeta)
