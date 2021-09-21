# ML_Project

# 1 Problema
La classificazione di serie temporali (TSC) è un task molto diffuso nell'ambito del machine learning. Negli ultimi anni, grazie al recente successo del deep learning e alla grande disponibilità di dati storici, le reti neurali deep hanno prodotto risultati allo stato dell'arte in molti domini applicativi di TSC [1]. Rispetto ad un normale task di classificazione che si svolge su dati cross-sectional, l'obiettivo di un classificatore di serie temporali è assegnare un'etichetta a ciascuna serie considerando le eventuali dipendenze temporali locali e globali. Il problema da risolvere è un problema di Hand Gesture Classification usando come input i dati storici generati dall'accelerometro a tre assi di un Wii Remote.

## 2 Dataset
Il training set comprende tre le in formato csv (train gesture x, train gesture y, train gesture z). Ciascun file contiene rispettivamente, per ogni gesture etichettata (train label), l'evoluzione lungo i 3 assi X, Y, Z. Ogni asse contiene 5000 time series di 315 osservazioni [2]. Il dataset contiene 8 possibili gesture (indicizzate da 0 a 7):
