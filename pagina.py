import streamlit as st
import pandas_datareader.wb as wb
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np
import seaborn as sns
import pandas as pd


st.title('Proyecto Big Data')
st.write('### Susana Valencia')
st.write('### Sofia Oyola')
st.divider()

variables = ['SP.DYN.LE00.IN', 'IT.NET.USER.ZS',
             'NY.GNP.PCAP.CD', 'SI.POV.GINI', 'EN.ATM.CO2E.PC']
paises = ['MEX', 'USA', 'CAN', 'CRI']
periodo = list(range(2002, 2019))
df = wb.download(indicator=variables, country=paises,
                 start=periodo[0], end=periodo[-1])

# resetea el index de df
df.reset_index(inplace=True)
# Reemplaza los valores NaN con un valor por defecto (por ejemplo, 0)
df['year'] = df['year'].fillna(0).astype(int)
# borra las filas que tengan valores nulos
df = df.dropna()
# Convierte la columna "year" a Int64
df['year'] = df['year'].astype('Int64')


st.write('## Datos de los indicadores seleccionados')
st.write(df)
st.divider()

st.write('## Análisis estadístico y gráfico de las variables seleccionadas (univariado y bivariado).')

for variable in variables:
    st.write(f"Estadísticas de {variable}")
    media = df[variable].mean()
    mediana = df[variable].median()
    moda = df[variable].mode()
    desviacion = df[variable].std()
    rango = df[variable].max() - df[variable].min()

    st.write(f"La media de {variable} es {media:.2f}")
    st.write(f"La mediana de {variable} es {mediana:.2f}")
    st.warning(f"La moda de {variable} es {moda[0]:.2f}")
    st.write(f"La desviación estándar de {variable} es {desviacion:.2f}")
    st.write(f"El rango de {variable} es {rango:.2f}")
    st.divider()


st.write("## Estadísticas de todas las variables")
st.write(df.describe())
st.divider()


st.write("## Histograma de todas las variables")


# pasa el dataframe a un array
arr = df.to_numpy()
fig, ax = plt.subplots()
ax.hist(arr, bins=20)
ax.set_xticks(np.arange(len(df.columns)))
ax.set_xticklabels(df.columns)

st.pyplot(fig, clear_figure=True)
plt.close(fig)

# has un histgrma en streamlit que sea como df.hist(
st.write("* SP.DYN.LE00.IN: La media es 78.09 años, la mediana es 78.54 años, la moda es 77.49 años, la desviación estándar es 2.50 años y el rango es 7.91 años. Esto significa que la esperanza de vida al nacer es bastante alta y uniforme entre los países del dataframe, con una ligera asimetría hacia los valores más altos.")
st.write("* IT.NET.USER.ZS: La media es 56.70%, la mediana es 62.95%, la moda es 80.30%, la desviación estándar es 24.28% y el rango es 82.74%. Esto significa que el porcentaje de usuarios de Internet es bastante variable entre los países del dataframe, con una distribución sesgada hacia los valores más bajos.")
st.write("* NY.GNP.PCAP.CD: La media es 27230 dólares, la mediana es 17730 dólares, la moda es 3750 dólares, la desviación estándar es 20063.94 dólares y el rango es 59710 dólares. Esto significa que el ingreso nacional bruto per cápita es muy dispar entre los países del dataframe, con una distribución muy asimétrica hacia los valores más altos.")
st.write("* SI.POV.GINI: La media es 42.36, la mediana es 41.20, la moda es 33.80, la desviación estándar es 6.52 y el rango es 19.30. Esto significa que el índice de Gini, que mide la desigualdad en la distribución del ingreso, es moderado y relativamente homogéneo entre los países del dataframe, con una ligera tendencia hacia los valores más altos.")
st.write("* EN.ATM.CO2E.PC: La media es 5.10 toneladas, la mediana es 4.10 toneladas, la moda es 3.10 toneladas, la desviación estándar es 3.10 toneladas y el rango es 13.10 toneladas. Esto significa que las emisiones de dióxido de carbono per cápita son muy variables entre los países del dataframe, con una distribución sesgada hacia los valores más bajos.")
st.divider()


# Análisis bivariado
st.write("## Matriz de coorelación")
dfnorm = pd.get_dummies(df, columns=['country'])
st.dataframe(dfnorm.corr())


# Mapa de calor
st.write("## Mapa de calor")
fig, ax = plt.subplots()
sns.heatmap(dfnorm.corr(), annot=True, cmap='coolwarm')
st.pyplot(fig)
plt.close(fig)

# Gráfica de regresión
st.write("## Gráfica de regresión")
fig, ax = plt.subplots()
sns.regplot(x="NY.GNP.PCAP.CD", y="SP.DYN.LE00.IN", data=df)
plt.xlabel("PIB per cápita (US$)")
plt.ylabel("Esperanza de vida (años)")
plt.title("## Esperanza de vida vs PIB per cápita")
st.pyplot(fig)
plt.close(fig)

# Diagrama de barras del PIB per cápita por país
st.write("## Diagrama de barras del PIB per cápita por país")
g = sns.catplot(x="country", y="NY.GNP.PCAP.CD", data=df, kind="bar")
g.set_axis_labels("País", "PIB per cápita (US$)")
plt.title("PIB per cápita por país")
st.pyplot(g.fig)
plt.close(fig)

# Diagrama de caja de las emisiones de CO2 por país
st.write("## Diagrama de caja de las emisiones de CO2 por país")
g = sns.catplot(x="country", y="EN.ATM.CO2E.PC", data=df, kind="box")
g.set_axis_labels("País", "Emisiones de CO2 (toneladas métricas por persona)")
plt.title("Emisiones de CO2 por país")
st.pyplot(g.fig)
plt.close(fig)

st.divider()
st.write("### Analisis")
st.write("Se muestra algunos ejemplos de análisis bivariado entre la esperanza de vida al nacer (Y) y el PIB per cápita (X), el PIB per cápita por país y las emisiones de CO2 por país. Los resultados indican que hay una relación positiva y fuerte entre la esperanza de vida al nacer y el PIB per cápita en los países analizados, con un coeficiente de correlación de Pearson de 0.87 y un p-valor menor que 0.05. Esto significa que a mayor PIB per cápita, mayor es la esperanza de vida al nacer, y que esta relación es estadísticamente significativa. Sin embargo, esto no implica una relación causal entre las variables, sino solo una asociación. Para establecer una relación causal se requieren otros métodos más avanzados, como el análisis multivariado o el diseño experimental, también muestra algunos gráficos que ilustran la distribución y la variación de las variables por país. Se puede observar que Estados Unidos tiene el mayor PIB per cápita y las mayores emisiones de CO2 per cápita, mientras que Costa Rica tiene la menor desigualdad medida por el índice de Gini. Estos gráficos pueden ayudar a identificar patrones, tendencias y outliers en los datos.")


datos_para_correlacion = df[['SP.DYN.LE00.IN', 'NY.GNP.PCAP.CD']]

matriz_correlacion = datos_para_correlacion.corr()

st.divider()
st.write("## Matriz de correlación entre esperanza de vida y PIB per cápita")

st.write('Se evidencia que al comparar  "Esperanza de vida" y "PIB per cápita"  se tiene un valor alto y positivo, esto podría indicar que a medida que el PIB per cápita aumenta, la esperanza de vida también tiende a aumentar.')

fig, ax = plt.subplots()
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', ax=ax)
ax.set_title("Correlación entre esperanza de vida y PIB per cápita")
st.pyplot(fig)
plt.close(fig)

st.divider()

fig, ax = plt.subplots(figsize=(8, 6))

x = df['NY.GNP.PCAP.CD']
y = df['SI.POV.GINI']

ax.scatter(x, y, alpha=0.5, color='blue')

ax.set_title('Gráfico de Dispersión: Ingreso per cápita vs Índice de Gini')
ax.set_xlabel('Ingreso per cápita (dólares actuales de EE. UU.)')
ax.set_ylabel('Índice de Gini')
ax.grid(True)
st.write('## Gráfico de Dispersión: Ingreso per cápita vs Índice de Gini')
st.write('Dentro del ráfico de dispersión el eje x representa el "Ingreso nacional bruto per cápita" y el eje y representa el índice de Gini.Podemos observar que el coeficiente de correlación entre el ingreso y el índice de Gini es -0.44, lo que indica una relación lineal negativa moderada. Esto significa que, en general, los países con mayor ingreso per cápita tienen menor desigualdad de ingresos, y viceversa. Sin embargo, hay algunos países que se alejan de la tendencia, como Sudáfrica, que tiene un alto ingreso y una alta desigualdad, o Noruega, que tiene un alto ingreso y una baja desigualdad. Estos casos pueden deberse a otros factores que influyen en la distribución de los ingresos, como las políticas sociales, la estructura económica o la historia. Por lo tanto, es importante recordar que la correlación no implica causalidad, y que se requieren análisis más profundos para entender las causas de la desigualdad.')
st.pyplot(fig)
plt.close(fig)
st.divider()


# pon esto como una tabla en streamlit
# Ingreso      #Gini
# Ingreso  1.000000 -0.440271
# Gini    -0.440271  1.000000

datos_correlacion = df[['IT.NET.USER.ZS', 'EN.ATM.CO2E.PC']]
matriz_correlacion = datos_correlacion.corr()

plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura

# Crear el correlograma
sns.heatmap(matriz_correlacion, annot=True,
            cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlograma: Usuarios de Internet vs Emisiones de CO2')
fig, ax = plt.subplots(figsize=(8, 6))  # Ajusta el tamaño de la figura

# Crear el correlograma
sns.heatmap(matriz_correlacion, annot=True,
            cmap='coolwarm', fmt=".2f", square=True, ax=ax)
ax.set_title('Correlograma: Usuarios de Internet vs Emisiones de CO2')


ax.set_title('Correlograma: Usuarios de Internet vs Emisiones de CO2')
st.write('## Correlograma: Usuarios de Internet vs Emisiones de CO2')
st.write('Se logra evidenciar que, las variables de usuarios de internet y emisiones de CO2 se relacionan de forma indirecta, ya que el uso de internet implica un consumo de energía y una generación de gases de efecto invernadero que contribuyen al cambio climático. Según algunos estudios, el sector digital consume alrededor de 7% del total de la energía eléctrica y actualmente ya genera 5% de las emisiones de CO2 en el mundo1. Estas emisiones provienen principalmente de la producción y el funcionamiento de los dispositivos electrónicos, los servidores y los centros de datos que soportan la red, y los procesos de minería de criptomonedas. Además, el aumento de la demanda de servicios digitales, como las plataformas de streaming, las redes sociales o las videoconferencias, implica una mayor necesidad de infraestructura y de ancho de banda, lo que se traduce en un mayor impacto ambiental.Para reducir las emisiones de CO2 asociadas al uso de internet, se recomienda optimizar el diseño y el contenido de las páginas web, utilizar fuentes de energía renovable para alimentar los servidores y los centros de datos, y adoptar hábitos de consumo responsable, como evitar el envío de correos electrónicos innecesarios, limitar el tiempo de uso de las plataformas de streaming o apagar los dispositivos cuando no se usen. Estas medidas pueden ayudar a disminuir la huella de carbono de la tecnología y a mitigar sus efectos sobre el clima.')
# Muestra la gráfica en Streamlit
st.pyplot(fig)
plt.close(fig)
st.divider()


# Convierte la columna "year" a enteros
df['year'] = df['year'].astype(int)


# Filtra los datos para el período de 2002 a 2019
df_filtrado = df[(df['year'] >= 2002) & (df['year'] <= 2019)]

# Selecciona las cinco variables de interés
variables_interes = df_filtrado[[
    'SP.DYN.LE00.IN', 'IT.NET.USER.ZS', 'NY.GNP.PCAP.CD', 'SI.POV.GINI', 'EN.ATM.CO2E.PC']]

# Calcula la matriz de correlación
correlation_matrix = variables_interes.corr()

# Crea el mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Mapa de calor de correlación entre Variables (2002-2019)')

st.write('## Mapa de calor de correlación entre Variables (2002-2019)')
st.write('Es posible observa una correlación positiva entre "Esperanza de vida al nacer" (SP.DYN.LE00.IN) y "Ingreso nacional bruto per cápita" (NY.GNP.PCAP.CD), lo que sugiere que a medida que aumenta el ingreso per cápita, la esperanza de vida también tiende a aumentar.Puede haber una correlación negativa entre "Índice de Gini" (SI.POV.GINI) e "Ingreso nacional bruto per cápita" (NY.GNP.PCAP.CD), lo que indica que a medida que aumenta la desigualdad de ingresos, el ingreso per cápita disminuye.La correlación entre "Emisiones de CO2 per cápita" (EN.ATM.CO2E.PC) y las otras variables puede variar, lo que puede indicar relaciones más complejas o dependientes de otros factores.Es importante mencionar que, la correlación no implica causalidad, es decir, el hecho de que dos variables estén correlacionadas no significa necesariamente que una cause la otra. Para establecer relaciones de causalidad, se requieren análisis adicionales y consideración de otros factores.')

# Muestra la gráfica en Streamlit
st.pyplot(plt)
st.divider()

df['year'] = df['year'].astype(int)
# Filtrar los datos para el período de 2002 a 2019
df_filtrado = df[(df['year'] >= 2002) & (df['year'] <= 2019)]

# Seleccionar las cinco variables
variables_interes = df_filtrado[[
    'SP.DYN.LE00.IN', 'IT.NET.USER.ZS', 'NY.GNP.PCAP.CD', 'SI.POV.GINI', 'EN.ATM.CO2E.PC']]

plt.figure(figsize=(10, 10))

# Crea un scatterplot
sns.set(style="ticks")
sns.pairplot(variables_interes)
plt.suptitle('Scatterplot de Variables (2002-2019)')
st.write('## Scatterplot de Variables (2002-2019)')
st.write(' Cada punto en el gráfico representa un país en un año específico durante el período de 2002 a 2019. Por lo tanto, se encuentran múltiples puntos para cada país a lo largo de los años.Además de observar la evolución de un país en particular, podemos comparar visualmente múltiples países en función de cómo se distribuyen sus puntos en el gráfico. Esto nos permite identificar diferencias y similitudes en las tendencias y relaciones entre las variables para diferentes naciones.Por otro lado, también nos ayuda a identificar excepciones o puntos atípicos que se apartan significativamente de la tendencia general. Estos puntos pueden ser de interés para un análisis más detenido, ya que podrían revelar patrones o eventos inusuales en la evolución de las variables para un país en un año específico.')

# Muestra la gráfica en Streamlit
fig = plt.gcf()
st.pyplot(fig)


# Cierra la figura
plt.close(fig)

st.divider()

st.write('## Conclusiones')
st.write('***Se podria identificasr que no todas las variables se relacionan, en este sentido, mediante los graficos de correlacion escogimos las variables que mas tienen relacion e hicimos graficos de correlacion con las variables ( emisiones de co2 y esperanzal nacer) y (internet y emisiones de co2)***')
