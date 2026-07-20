# Primjena AIOps metodologije za detekciju anomalija u GitHub Actions cjevovodima
 Dominik Ožvald, Vlado Sruk\
 Ovaj repozitorij sadrži izvorni Python kod za implementaciju arhitekture, postupak treniranja i evaluaciju modela dubokog učenja namijenjenih detekciji anomalija u zapisima (engl. logs) nastalim unutar radnih tokova GitHub Actions cjevovoda. 
 Repozitorij je nastao u sklopu diplomskog rada „Primjena AIOps metodologije za detekciju anomalija u GitHub Actions cjevovodima” Dominika Ožvalda (2026.), čiji se cjeloviti tekst nalazi u direktoriju docs/. 
 U radu su definirane i analizirane četiri ključne kategorije anomalija: nestabilni ispit (engl. flaky tests), konfiguracijski pomak (engl. configuration drift), tihi neuspjesi (engl. silent failures) i sigurnosne anomalije.
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

Direktorij data organiziran je u dvije cjeline s obzirom na podrijetlo i namjenu podataka. Data/dummy/ sadrži umjetno generirane podatke podijeljene na skupove za učenje, ispitivanje i učenje rekonstrukcijskog modela. Skup namijenjen rekonstrukcijskom modelu ne sadrži anomalije. Data/gha/ sadrži stvarne, označene zapise iz GHALogs skupa podataka, podijeljene na dio za učenje i ispitivanje.\
Unutar direktorija models nalaze se sačuvane težine obučenih modela u .pt formatu, nazvane prema arhitekturi. U ovom direktoriju nalaze se sljedeći artefakti:
- ConvLSTM_E_32_H_196_L_128.pt: Sačuvani model autoenkodera
- RecTransformer_DE_2_H_2_F_1024.pt: Rekonstrukcijski model transformera
- TaggedTransformer_E_2_H_2_F_1024_D_128.pt: Višeklasni model prije dodatnog učenja.
- TaggedTransformer_E_2_H_2_F_1024_D_128_tuned.pt: Višeklasni model nakon dodatnog učenja (finog podešavanja) na GHALogs skupu podataka

Direktorij results sadrži rezultate evaluacije modela. Datoteke u svojem nazivu sadrže ime pripadajućeg modela radi lakše identifikacije. U direktoriju se nalaze slike koje prikazuju matrice zabune za višeklasne modele, kao i ROC krivulje za rekonstrukcijski model. Osim slika u direktoriju se nalaze .txt datoteke s numeričkim vrijednostima preciznosti, odziva, F1 mjere i njihovim makro prosjekom.

    
## Preduvjeti
Sva kod u ovom repozitorjiju je piasan u porgramskogm jeziku Python pa je za njegovo pokretanje dovoljno instalirati Python i potrebene python pakete.
Preduvjeti za pokretanje koda su python 3.10+, torch minimalno 2.5.1, numpy minimalno 2.2.6, scikit-learn minimalno 1.5.2 i matplotlib minimalno 3.9.2
## Instalacija i pokretanje
Instalacija potrebnih paketa 
```
$ pip install torch numpy scikit-learn matplotlib
```
Kloniranje repozitorija
```
$ git clone https://github.com/DominikOzvald/msc-dominik-ozvald-2026.git
```
Pokretanje ispitivanja 
```
$ cd msc-dominik-ozvald-2026
```
```
$ python -m src.GHA_AIOps.evaluating.tag_transformer
```
```
$ python -m src.GHA_AIOps.evaluating.transformer
```
## Minimalni radni primjer i Reprodukcija rezultata
Pokretanjem navedenih naredba instalirat će se potrebni python paket za ispitivanje. Rezultat ispitivanja višeklasnog će biti matrica zabune kao ona u direktoriju results zajedno s ispisom preciznosti, odziva, F1 mjere i njihovog makro prosjeka na standardni izlaz. Kod ispitivanja rekosntrukcijskog modela rezultat će biti slika ROC krivulje kao ona u direktoriju results.
## Licence
Na ovaj kod se odnosi MIT licenca, a na GHALogs skup podataka se odnosi Creative Commons Attribution Share Alike 4.0 international licenca. Skup podatak se može pronaći na poveznici https://zenodo.org/records/10154920