'''
Nome Script: Analisi tipo di corpo celesti tramite i dati acquisiti utilizzando il metodo Logistic Regression.py
Descrizione: Questo script elabora e analizza i dati di di un database e cerca di classificare a quele tipologia
di corpo celeste appartiene.
Autore: [Francisco J.Scognamiglio]
Versione: 1.0
Data: [12.5.25]
Copyright: © [Francisco J.Scognamiglio] [2025]
Licenza: [MIT]
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



df = pd.read_csv('C:\\Users\\hp\\Documents\\Corso AI Cefi\\file .csv\\star_classification.csv')

X = df[["alpha","delta","u","g","r","i","z"]]
y = df[['class']].values.ravel()
class_name = ['Galassia', 'Stella', 'QSO'] 

feature_names=["alpha","delta","u","g","r","i","z"]


# Dividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Standardizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit + Transform sul training set
X_test = scaler.transform(X_test) 

# Inizializza e allena il modello LogisticRegression per la classificazione multiclasse
# Per impostazione predefinita, multi_class='ovr'

logistic_model_ovr = LogisticRegression(random_state=42)
logistic_model_ovr.fit(X_train, y_train)
y_pred_ovr = logistic_model_ovr.predict(X_test)
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
print(f"Accuratezza con strategia One-vs-Rest (OvR): {accuracy_ovr:.4f}")
print("\nClassification Report (One-vs-Rest):")
print(classification_report(y_test, y_pred_ovr, target_names=class_name))

# Utilizza la strategia multinomiale (Softmax)
logistic_model_multinomial = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs', max_iter=1000)
logistic_model_multinomial.fit(X_train, y_train)
y_pred_multinomial = logistic_model_multinomial.predict(X_test)
accuracy_multinomial = accuracy_score(y_test, y_pred_multinomial)
print(f"Accuratezza con strategia Multinomiale (Softmax): {accuracy_multinomial:.4f}")
