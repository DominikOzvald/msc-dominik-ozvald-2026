GHA-AIOps
 
AIOps-based anomaly detection for GitHub Actions pipelines.

Author: Dominik Ožvald · Supervisor: Vlado Sruk
Institution: University of Zagreb Faculty of Electrical Engineering and Computing · Year: 2026


 This repository contains the Python source code for the architecture, training procedure, and evaluation of deep learning models intended for anomaly detection in logs generated within GitHub Actions pipeline workflows. 
 The repository was created as part of the master's thesis "Application of AIOps methodology for anomaly detection in GitHub Actions pipelines" by Dominik Ožvald (2026), the full contents of which can be found in the docs/ directory. 
 Key categories of anomalies defined and analyzed in the paper are: flaky tests, configuration drift, silent failures and security anomalies.

Mian features:
- preprocessing of GitHub Actions logs;
- synthetic dataset generation;
- multiclass anomaly detection;
- reconstruction-based anomaly detection;
- model training and evaluation;
- GitHub Actions integration;
- anomaly reporting and notification.

## Struktura repozitorija
Programsko rješenje podijeljeno je u nekoliko glavnih direktorija. Cjelokupni izvorni kod nalazi se unutar direktorija src, dok su podaci, trenirani modeli i rezultati organizirani u zasebne cjeline radi modularnosti i lakše reproduktivnosti rezultata. \
Direktorij src predstavlja jezgru implementacije i podijeljen je na tri primarna poddirektorija: GHA_AIOps (GHA-AIOps pristup), data_generation (generiranje umjetnih podataka) i integration (Integracija GHA-AIOps pristupa u radni tok)\
Src/GHA_AIOps sadrži logiku za definiranje arhitektura dubokih modela, njihovo učenje i ispitivanje. U njemu se nalaze sljedeće datoteke i direktoriji.
- GHA_AIOps/models/ - Arhitektura modela
  - convlstm.py: Sadrži implementaciju autoenkodera
  - transformer.py: Sadrži definicije razrede (engl. class) za višeklasni i rekonstrukcijski model transformera
  - embedder.py: Sadrži klasu omotača (engl. wrapper) oko autoenkodera koja služi za pravilno oblikovanje i pripremu tenzora prije prosljeđivanja koderu autoenkodera.
- GHA_AIOps/utils/ - Pomoćne funkcije i procesiranje zapisa:
  - data.py: Sadrži funkcije za pretprocesiranje tekstualnih zapisa
  - datasets.py: Implementira prilagođene razrede za učitavanje podataka
  - embeddings.py: Sadrži klasu zaduženu za dodjeljivanje numeričkih vrijednosti ASCII znakovima unutar zapisa
  - train.py: Sadrži petlje za učenje modela
- GHA_AIOps/training/ - skripte za pokretanje učenja 
  - convlstm.py: Pokreće proces učenja autoenkodera.
  - transformer.py: Koristi se za treniranje rekonstrukcijskog modela transformera.
  - tag_transformer.py: Pokreće učenje višeklasnog modela transformera
  - tuning.py: Sadrži logiku za dodatno učenje višeklasnog modela transformera na stvarnim podacima
- GHA_AIOps/testing/ Evaluacijske skripte:
  - transformer.py: Služi za ispitivanje rekonstrukcijskog modela.
  - tag_transformer.py: Služi za ispitivanje višeklasnog modela.

Src/data_generation/ sadrži skripte korištene za generiranje umjetnog skupa podataka. U njemu se nalaze sljedeće datoteke
- build.py: Simulira fazu izgradnje aplikacije
- test.py: Simulira izvođenje jediničnih testova
- package.json: Definira pakete koji se pojavljuju u konfiguracijskom pomaku.
- test_values.json: Sadrži ispitne vrijednosti koje se koriste za simulaciju nestabilnih ispita

Src/integration/ sadrži skripte za dohvat zapisa i integriranje njihove obrade u radni tok.  U njemu se nalaze sljedeće datoteke
- action.yml: Definira GitHuba akciju i njezine korake
- fetch_logs.py: Dohvaća zapise i sprema hi u virtualno okruženje
- detect.py: Učitava model i koristi ih za detekciju anomalija i izradu izvještaja

Direktorij data organiziran je u dvije cjeline s obzirom na podrijetlo i namjenu podataka. Data/dummy/ sadrži umjetno generirane podatke podijeljene na skupove za učenje, ispitivanje i učenje rekonstrukcijskog modela. Skup namijenjen rekonstrukcijskom modelu ne sadrži anomalije. Data/gha/ sadrži stvarne, označene zapise iz GHALogs skupa podataka, podijeljene na dio za učenje i ispitivanje.\
Unutar direktorija models nalaze se sačuvane težine obučenih modela u .pt formatu, nazvane prema arhitekturi. U ovom direktoriju nalaze se sljedeći artefakti:
- ConvLSTM_E_32_H_196_L_128.pt: Sačuvani model autoenkodera
- RecTransformer_DE_2_H_2_F_1024.pt: Rekonstrukcijski model transformera
- TaggedTransformer_E_2_H_2_F_1024_D_128.pt: Višeklasni model prije dodatnog učenja.
- TaggedTransformer_E_2_H_2_F_1024_D_128_tuned.pt: Višeklasni model nakon dodatnog učenja (finog podešavanja) na GHALogs skupu podataka

Direktorij results sadrži rezultate evaluacije modela. Datoteke u svojem nazivu sadrže ime pripadajućeg modela radi lakše identifikacije. U direktoriju se nalaze slike koje prikazuju matrice zabune za višeklasne modele, kao i ROC krivulje za rekonstrukcijski model. Osim slika u direktoriju se nalaze .txt datoteke s numeričkim vrijednostima preciznosti, odziva, F1 mjere i njihovim makro prosjekom.

    
## Requirements
- Python 3.10 or newer
- Dependencies listed in  [`requirements.txt`](requirements.txt).

## Instaltion
Dependency installation 
```
$ pip install torch numpy scikit-learn matplotlib
```
Clone repository
```
$ git clone https://github.com/DominikOzvald/msc-dominik-ozvald-2026.git
```
## Dataset
 The data directory is organized into two sections based on data origin and purpose. 
 The `Data/dummy/` directory contains synthetically generated data divided into sets for training, testing, and training the reconstruction model; the set intended for the reconstruction model contains no anomalies. 
 Anomalies form the four key categories were inserted and automatically labeled. This set was generated using scripts form `src/data_generation` in a GitHub Action workflow.\
 The `Data/gha/` directory contains real, labeled logs from the GHALogs dataset, divided into training and testing subsets. This data was taken from real public GitHub repositories and labeled by hand for the use in this thesis.
 This only a subset containing only workflow runs with anomalies.
 
## Models
- **autoencoder**:
  - takes: preprocessed logs
  - outputs: fixed length vector representations
  - purpose: extract features form logs
  - architecture: `src/GHA_AIOps/models/convlstm.py`
  - training: `src/GHA_AIOps/training/convlstm.py`
- **multiclass transformer based model**:
  - takes: sliding window of vector representations form the autoencoder
  - outputs: vector representing the probability that a log line belongs to an anomaly class or is normal
  - purpose: detection and classification of anomalies
  - architecture: `src/GHA_AIOps/models/transformer.py`
  - training: `src/GHA_AIOps/training/tag_transformer.py`
  - evaluation: `src/GHA_AIOps/evaluating/tag_transformer.py`
- **reconstruction based transformer**
  - takes: sliding window of vector representations form the autoencoder
  - outputs: reconstructed window of vector representations
  - purpose: detecting anomalies by their deviation form normal log patterns
  - architecture: `src/GHA_AIOps/models/transformer.py`
  - training: `src/GHA_AIOps/training/transformer.py`
  - evaluation: `src/GHA_AIOps/evaluating/transformer.py`


## Evaluation
To evaluate the trained models first move to repo root directory
```
$ cd msc-dominik-ozvald-2026
```
To evaluate the multiclass transformer model run the following command: 
```
$ python -m src.GHA_AIOps.evaluating.tag_transformer
```
The result of this command should be the following confusion matrix: ![TaggedTransformer_E_2_H_2_F_1024_D_128_matrix.png](results%2FTaggedTransformer_E_2_H_2_F_1024_D_128_matrix.png)  
Stdout should show the following metrics:\
Standard : {'precision': 0.9943, 'recall': 0.99818, 'f1-score': 0.99623, 'support': 7165.0}\
"Flaky" : {'precision': 0.96590, 'recall': 0.94095, 'f1-score': 0.95327, 'support': 271.0}\
Pomak : {'precision': 1.0, 'recall': 0.93636, 'f1-score': 0.96713, 'support': 220.0}\
Sigurnost : {'precision': 0.97810, 'recall': 0.8815, 'f1-score': 0.9273, 'support': 152.0}\
Tihi : {'precision': 0.76086, 'recall': 0.9210, 'f1-score': 0.8333, 'support': 38.0}\
accuracy : 0.9918429773132806 \
Anomaly accuracy : 0.9251101613044739

o evaluate the reconstruction transformer model run the following command: 

```
$ python -m src.GHA_AIOps.evaluating.transformer
```
The result of this command should be the following ROC curve:  ![RecTreansformer_DE_2_H_2_F_1024_ROC.png](results%2FRecTreansformer_DE_2_H_2_F_1024_ROC.png)

## GitHub Actions integration 

## Citation
 
### Software
 
To cite the software, use the citation metadata provided in
[`citation.cff`](citation.cff).
 
### Master's thesis
 
D. Ožvald, “Primjena AIOps metodologije za detekciju anomalija u GitHub
Actions cjevovodima” [“Application of the AIOps Methodology for Anomaly
Detection in GitHub Actions Pipelines”], M.S. thesis, University of Zagreb
Faculty of Electrical Engineering and Computing, Zagreb, Croatia, 2026.

## License
 
The software in this repository is licensed under the MIT License.
See [`LICENSE`](LICENSE).