# ML_Project

## 1 Problema
La classificazione di serie temporali (TSC) è un task molto diffuso nell'ambito del machine learning. Negli ultimi anni, grazie al recente successo del deep learning e alla grande disponibilità di dati storici, le reti neurali deep hanno prodotto risultati allo stato dell'arte in molti domini applicativi di TSC [1]. Rispetto ad un normale task di classificazione che si svolge su dati cross-sectional, l'obiettivo di un classificatore di serie temporali è assegnare un'etichetta a ciascuna serie considerando le eventuali dipendenze temporali locali e globali. Il problema da risolvere è un problema di Hand Gesture Classification usando come input i dati storici generati dall'accelerometro a tre assi di un Wii Remote.

![Cattura](https://user-images.githubusercontent.com/54946553/134172966-e8f4429d-79fc-4a38-bad3-c593c21cffd9.PNG)


## 2 Dataset
Il training set comprende tre file in formato csv (train gesture x, train gesture y, train gesture z). Ciascun file contiene rispettivamente, per ogni gesture etichettata (train label), l'evoluzione lungo i 3 assi X, Y, Z. Ogni asse contiene 5000 time series di 315 osservazioni [2]. Il dataset contiene 8 possibili gesture (indicizzate da 0 a 7):
![Cattura](https://user-images.githubusercontent.com/54946553/134173314-cf99dc4d-ed9d-4248-87ce-30fc543cc6db.PNG)

## 3 Valutazione
Il progetto può essere svolto individualmente o in gruppi composti da al massimo 2 componenti. Deve essere prodotta una breve relazione in cui si giustificano le scelte fatte. La metrica di valutazione del progetto è l'accuracy sul test set.

## 4 Modalità di consegna
Al fine di poter valutare nel migliore dei modi i progetti è importante che gli script siano chiari e parzialmente commentati. Oltre alla relazione, bisogna consegnare il codice sorgente con gli script per l'addestramento della rete neurale. Il codice deve inoltre contenere una routine di test che ha la funzione di leggere il test set (test gesture x, test gesture y, test gesture z), le etichette di test (test label.csv), il modello serializzato ed infine valutare in modo automatico l'accuracy sul test set.

## 5 Riferimenti
[1] https://arxiv.org/pdf/1809.04356.pdf

[2] https://we.tl/t-ZuVRsoOd1j
