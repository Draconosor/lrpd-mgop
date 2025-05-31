import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

# Cargar datos
df = pd.read_parquet(r"experiments\local_search_parameters\experimental_results.parquet")

# Asegurarse de que 'instance' sea de tipo string (si es necesario)
df['instance'] = df['instance'].astype('category')


# Formula para la regresión
# Utilizamos la fórmula de statsmodels para incluir la variable 'instance' como un bloque categórico
formula = 'final_emissions ~ n_iter + n_size + C(instance)'

# Ajustar el modelo
model = sm.OLS.from_formula(formula, data=df).fit()

# Imprimir los resultados
print("📌 Resultados del modelo global")
print(model.summary())

# Calcular residuos y valores ajustados
residuos = model.resid
valores_ajustados = model.fittedvalues

# Gráfico de residuos vs valores ajustados (para verificar homoscedasticidad)
stats.probplot(residuos, plot=plt)

# Prueba de normalidad de los residuos (Shapiro-Wilk)
shapiro_test = stats.shapiro(residuos)
print(f"Shapiro-Wilk test: Estadístico = {shapiro_test.statistic:.4f}, p-valor = {shapiro_test.pvalue:.4f}")

# Agrupar los valores ajustados por los diferentes niveles de la variable 'instance'
valores_ajustados_por_grupo = [valores_ajustados[df['instance'] == level] for level in df['instance'].unique()]

# Prueba de homoscedasticidad (Levene sobre valores ajustados agrupados por 'instance')
levene_test = stats.levene(*valores_ajustados_por_grupo)
print(f"Levene test: Estadístico = {levene_test.statistic:.4f}, p-valor = {levene_test.pvalue:.4f}")

# Crear el gráfico de influencia
fig, ax = plt.subplots(figsize=(8, 6))
sm.graphics.influence_plot(model, ax=ax)
plt.title('Gráfico de Influencia')
plt.show()


import statsmodels.formula.api as smf

# Ajustar un modelo de regresión robusta usando fórmula
formula = 'final_emissions ~ n_iter + n_size  + C(instance)'  # Ajusta la fórmula según tus variables
model_robusto = smf.rlm(formula, data=df).fit()

# Resumen del modelo robusto
print(model_robusto.summary())

r = model_robusto.resid

stats.probplot(r, plot=plt)
plt.show()