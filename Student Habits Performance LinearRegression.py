'''Nome Script: Analisi sulle malattie da inquinamento del suolo tramite la LinearRegression.py
Descrizione: Questo script elabora e analizza i dati di di un database e cerca di predire l'insorgenza di malattie
dovute all'inquinamento in sogetti adulti.

Autore: [Francisco J. Scognamiglio]
Versione: 1.0
Data: [1 maggio 2025]
Copyright: © [Francisco J. Scognamiglio] [2025]
Licenza: [MIT]
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('C:\\Users\\hp\\Documents\\Corso AI Cefi\\file .csv\\student_habits_performance.csv')
'''
df_1 = df.drop(columns=['student_id','gender', 'part_time_job', 'diet_quality','parental_education_level', 'internet_quality', 'extracurricular_participation'])
corr = df_1.corr()

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0, annot=True,
    #cmap=sns.light_palette("blue", 12),
    square=True
    
)
plt.show()
'''

'''
#df_filtrato = df[df['Top AI Tools Used'] == 'ChatGPT']
#print(df_filtrato)

plt.scatter(df['exam_score'], df['Consumer Trust in AI (%)'])
plt.xlabel('exam_score')
plt.ylabel('Market Share of AI Companies (%)')
plt.title('exam_score vs Consumer Trust in AI (%)')
plt.grid(True)
plt.show()
'''


# Calcolo della correlazione tra le variabili X e y
df1 = df[['exam_score', 'study_hours_per_day']]
corr = df1.corr()
print(corr)

correlazione1 = df['exam_score'].corr(df['study_hours_per_day'], method='spearman')
correlazione2 = df['exam_score'].corr(df['study_hours_per_day'], method='kendall')
print("Correlazione tra exam_score\n e study_hours_per_day (spearman):\n", correlazione1)
print("Correlazione tra exam_score\n e study_hours_per_day (kendall):\n", correlazione2)



#Regressione Lineare tramite un polinomio
#Specifica il grado del polinomio che vuoi adattare
grado_polinomio = 2

# Crea un oggetto PolynomialFeatures
polinomio = PolynomialFeatures(degree=grado_polinomio)

# Trasforma le variabili indipendenti originali in termini polinomiali
X_poly = polinomio.fit_transform(df['exam_score'].values.reshape(-1, 1))

# Ora puoi usare LinearRegression sui termini polinomiali trasformati
modello_polinomiale = LinearRegression()
modello_polinomiale.fit(X_poly, df['study_hours_per_day'])

# Puoi ottenere i coefficienti del modello polinomiale
print("Coefficienti del modello polinomiale:", modello_polinomiale.coef_)
print("Intercetta del modello polinomiale:", modello_polinomiale.intercept_)



#Creo il grafico per visualizzare i dati originali e polinomio 

x_range = np.linspace(df['exam_score'].min(), df['exam_score'].max(), 1000).reshape(-1, 1)
X_range_poly = polinomio.transform(x_range)
y_pred = modello_polinomiale.predict(X_range_poly)


#Valutazione
mse = mean_squared_error(df['exam_score'].values, y_pred)
r2 = r2_score(df['exam_score'].values, y_pred)

print(f"Mean Squared Error (con 2 variabili): {mse:.2f}")
print(f"R^2 Score (con 2 variabili): {r2:.2f}")



plt.figure(figsize=(6, 5))

#Plot dei dati originali
plt.scatter(df['exam_score'], df['study_hours_per_day'], color='blue', label='Dati Originali')

#Plot del polinomio
plt.plot(x_range, y_pred, color='red', label=f'Regressione Polinomiale (Grado {grado_polinomio})')


plt.xlabel ("Punteggio d'esame")
plt.ylabel('Ore di studio per giorno')
plt.title('Regressione Polinomiale')
plt.legend()
plt.grid(True)
plt.show()
