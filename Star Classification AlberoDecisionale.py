'''Nome Script:  Analisi tipo di corpo celesti tramite i dati acquisiti utilizzando l'albero decisionale.py
Descrizione: Questo script elabora e analizza i dati di di un database e cerca di classificare a quele tipologia
di corpo celeste appartiene.

Autore: [Francisco J. Scognamiglio]
Versione: 1.0
Data: [2 maggio 2025]
Copyright: © [Francisco J. Scognamiglio] [2025]
Licenza: [MIT]
'''

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import pandas as pd

df = pd.read_csv('star_classification.csv')

X = df[["alpha","delta","u","g","r","i","z"]]
y = df[['class']

class_name = ['Galassia', 'Stella', 'QSO'] 
feature_names=["alpha","delta","u","g","r","i","z"]


# 2. Divisione dei Dati in Training e Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Creazione e Addestramento del Modello dell'Albero Decisionale
# Puoi sperimentare con diversi criteri ('gini' o 'entropy') e profondità massima
albero_decisionale = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
albero_decisionale.fit(X_train, y_train)

# 4. Previsioni sul Set di Test
y_pred = albero_decisionale.predict(X_test)

# 5. Valutazione del Modello
print("Risultati dell'Albero Decisionale:")
print("\nMatrice di Confusione:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels= class_name, yticklabels=class_name)
plt.ylabel('Valori Reali')
plt.xlabel('Valori Predetti')
plt.title('Matrice di Confusione (Albero Decisionale)')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_name, zero_division=1))

# 6. (Opzionale) Visualizzazione dell'Albero Decisionale
plt.figure(figsize=(10, 8))
tree.plot_tree(albero_decisionale, feature_names=feature_names, class_names=class_name, filled=True)
plt.title("Albero Decisionale Addestrato")
plt.show()
