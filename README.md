# Primjena AIOps metodologije za detekciju anomalija u GitHub Actions cjevovodima
 Dominik Ožvald, Vlado Sruk\
 Ovaj repozitorij sadrži python kod za arhitekturu, učenje i ispitivanje dubokih modela za detekciju anomalija u zapisima nastalim u GitHub Actions cjevovodu. 
 Osim koda za modele sadrži i kod koji je korišten za generiranje umjetnog skupa podataka na kojem su modeli učeni.
## Struktura repozitorija
Direktorij src sadrži pod direktorije ML i data_generation. Direktorij ML sadrži kod za arhitekturu, učenje i ispitivanje dubokih modela, a data_generation za generiranje podataka.
Direktorij data sadrži generirani umjetni skup podataka i označani pod skup podtak iz GHALogs skupa podataka. Direkotirj mode sadrži podučene modele, a direktorij results rezultate ispitivanja tih modela.
## Preduvjeti
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
$ python -m src.ML.testing.tag_transformer
```
```
$ python -m src.ML.testing.transformer
```
## Minimalni radni primjer i Reprodukcija rezultata
pokretanjem gore navedenih naredbi će se pokrenuti ispitivanje naučenih modela i prikazati rezultati tog ispitivanja koji su jednaki rezultatima u direktoriju results 
## Licence
Na ovaj kod se odnosi MIT licenca, a na GHALogs skup podataka se odnosi Creative Commons Attribution Share Alike 4.0 international licenca. Skup podatak se može pronaći na poveznici https://zenodo.org/records/10154920
