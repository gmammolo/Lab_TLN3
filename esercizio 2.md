# Esercizio di laboratorio: complessità definizionale [LAB-2]

Misurazione dell’overlap lessicale tra una serie di definizioni per concetti generici/specifici e concreti/astratti.
*Si vedano le slides ed il materiale su Moodle per maggiori dettagli.*
Questo esercizio ha l’obiettivo di esplorare in modo pratico la complessità insita nella formulazione di definizioni. Attraverso un’attività collaborativa, i partecipanti saranno guidati nella produzione e nell’analisi di definizioni per concetti concreti e astratti, evidenziando le differenze nella rappresentazione linguistica e nella densità semantica.

## Fase 1: Creazione condivisa di definizioni
L’attività si articola nella creazione, collettivamente, di definizioni per quattro concetti distinti:
- Due concetti concreti:
  - Uno generico, ad esempio: Pantalone
  - Uno specifico, ad esempio: Microscopio
- Due concetti astratti:
  - Uno generico, ad esempio: Pericolo
  - Uno specifico, ad esempio: Euristica

Ogni definizione dovrà essere pensata per essere comprensibile a un lettore generico, e potrà seguire strategie diverse (genus-differentia, esempi, parafrasi, ecc.).

## Fase 2: Analisi collettiva
Dopo la redazione delle definizioni, si procederà a una breve analisi condivisa. I partecipanti saranno invitati a riflettere su:
- le scelte linguistiche adottate;
- il grado di specificità;
- le difficoltà incontrate nel definire concetti astratti rispetto a quelli concreti;
- l’uso (o assenza) di strategie definitorie canoniche.

## Fase 3: Task di analisi lessicale e semantica
Come estensione dell’esercizio, verrà assegnato un task di confronto tra le definizioni create. I partecipanti dovranno calcolare:
- il grado di sovrapposizione lessicale tra definizioni dello stesso concetto;
- il grado di sovrapposizione semantica, attraverso tecniche a scelta (sinonimia, uso di risorse come WordNet, embedding, ecc.).

Questo compito ha lo scopo di stimolare una riflessione critica sulla variabilità definizionale, nonché sull’efficacia delle definizioni nel trasmettere significati in modo univoco o condiviso.

## Fase 4: Calcolo della similarità lessicale e semantica

In questa fase, l’attenzione si sposta sull’analisi quantitativa delle definizioni prodotte. L’obiettivo è misurare quanto le definizioni condividano elementi linguistici e concettuali, considerando due dimensioni principali: **similarità lessicale (simlex) e similarità semantica (simsem)**

Procedimento:
1. Calcolare la similarità lessicale (simlex) tra tutte le definizioni:
   - Una possibile formula è il conteggio dei termini in comune tra due definizioni, dopo aver applicato tecniche di stopword removal, lemmatizzazione, e normalizzazione.

2. Calcolare la similarità semantica (simsem):
   - Una possibile strategia consiste nel rappresentare ciascuna definizione con un vettore (ad es. tramite media di word embeddings) e calcolare la cosine similarity tra i vettori.
   - È possibile scegliere liberamente il modello semantico (es. Word2Vec, GloVe, BERT, ecc.).
  
3. Aggregare i valori di simlex e simsem in base a due dimensioni:
   - Concretezza: concetti concreti vs. concetti astratti
   - Specificità: concetti generici vs. concetti specifici
  
4. Calcolare i valori medi (o altre statistiche a scelta) di simlex e simsem tra le definizioni prese due a due, all’interno di ciascun gruppo (es. tutte le definizioni per concetti concreti, ecc.)


## Fase 5: Analisi e riflessione sui risultati
Sulla base dei risultati ottenuti, provate a rispondere alle seguenti domande: 

- La sovrapposizione lessicale tra le definizioni è risultata più alta o più bassa rispetto alle vostre aspettative?
- La sovrapposizione semantica riflette meglio la vostra intuizione di sovrapposizione delle definizioni?
- Avete notato differenze significative tra concetti concreti e astratti, o tra quelli generici e specifici?
- Alcune definizioni vi sono sembrate più “prototipiche” o condivise di altre? Perché?

Questa attività mira a stimolare la consapevolezza delle variazioni definizionali, nonché a introdurre strumenti computazionali per l’analisi del significato e della formulazione del
linguaggio