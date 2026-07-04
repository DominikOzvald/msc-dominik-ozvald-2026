# Primjena AIOps metodologije za detekciju anomalija u GitHub Actions cjevovodima
 Dominik Ožvald, Vlado Sruk\
 Ovaj repozitorij sadrži python kod za arhitekturu, učenje i ispitivanje dubokih modela za detekciju anomalija u zapisima nastalim u GitHub Actions cjevovodu. 
 Takvi zapisi nastaju tijekom izvođenja CI/CD cjevovoda i mogu sadržavati obrasce koji upućuju na kvarove, degradacije pouzdanosti, konfiguracijske probleme ili sigurnosne rizike, čak i kada radni tok ne završava nužno neuspjehom.
 Osim koda za modele sadrži i kod koji je korišten za generiranje umjetnog skupa podataka na kojem su modeli učeni. Ovaj kod je vazan uz diplomski rad koji se nalazi u direkotirju docs.
Definirane su četiri vrste anomalija promatrane u radu: nestabilni ispiti, konfiguracijski pomak, tihi neuspjesi i sigurnosne anomalije.. Za potrebe učenja i ispitivanja izrađen je skup podataka koji simulira izgradnju i ispitivanje poslužiteljske aplikacije u koji su ubačeni različiti tipovi anomalija, uz kontrolirano umetanje različitih tipova anomalija. 
Dodatno je pristup provjeren na označenom podskupu stvarnih zapisa iz GHALogs skupa podataka. Izlučivanje značajki iz zapisa se radi pomoću autoenkodera koji koristi LSTM mrežu i jednodimenzionalnu konvoluciju za kodiranje zapisa u vektore jednake duljine. Dobiveni vektori se grupiraju u pomoću klizni prozora te prosljeđuju u dva modela temeljena na transformeru. Jedan model uči na označenim podacima i razvrstava anomalije po tipovima, a drugi rekonstruira ulazne vektore i određuje anomaliju po razlici između ulaznih i izlaznih vektora. 
## Struktura repozitorija
Izvroni python kod se  nalazi u drekotirju src. Direktorij src sadrži pod direktorije ML i data_generation. Direktorij ML sadrži kod za arhitekturu, učenje i ispitivanje dubokih modela, a data_generation za generiranje podataka.
ML/models sadrži razrede koji opisuju arhrekturu dubokih modela, ML/models/convlsrtm.py ospisuje Autoenkoder, ML/models/transformer.py opisjuje višeklasni i rekostukciski model transformera,
a ML/models/embedder.py sadrži razred omotača oko autoenkodera koji služi za prvilno oblikovanje tenzora prije nego se proslijedi autoenkoderu. 
ML/utils sadrži razrede za porcesiranje podatak i petlje za učenje modela. ML/utils/data.py sadrži funkcije za pred procesitanje zapisa. ML/utils/datasets.py sadrži razrede za učitavanje zapisa. 
ML/utils/ebeddings.py sadrži razred koji dodjlejuje numerićke virjednosti ascii znakovima zapisa.
ML/utils/train.py sadrži petelje za učenje modela.
ML/training sadrži skrpte za pokertanje učenja modela. ML/training/transformer.py uči rekosnturkcijsi model dok ML/training/tag_transfomer uči višeklasni model, a ML/training/convlstm.py autoenkoder. ML/training/tunning.py dodatno uči model transformera prije ispitvanja na GHALogs skupu podatak.
ML/testing sadrži skripte za ispitavnaje modela. ML/testing/transformer.py isputije reksontukcijski model, a ML/testing/tag_transformer višeklasni model.
U poddirektoriku data_generation nalazi se kod za generiranje umjetnih podataka. Data_generation/build.py simulira fazu igradnje aplikacije, a data_generation/test.py simulira jediničo ispitvanje. Paketi koji pojvljuju u konfiguracijskom pomaku su definirani u data_generation/package.json, a ispitne vrijednosti za nestabilne ispite u data_generation/test_valures.json
Direktorij data sadrži generirani umjetni skup podataka i označani pod skup podatak iz GHALogs skupa podataka. Data/dummy sadrži umjetno generirane podatake podjeljene u skup za učenje, ispitivanje i učenje rekonstukcijskog modela. Podaci za učenje rekonstrukcijskog modela ne sadrže anomalije. Data/gha sadrži stvarne zapise podijljen u podatke za ispitvanje i učenje. 
Direkotirj models sadrži učene modele. Spremljeni su višeklansi model prije i poslje doatanog učenja na GHALogs skupu podataka, nazvani TaggedTransformer_E_2_H_2_F_1024_D_128.pt i TaggedTransformer_E_2_H_2_F_1024_D_128_tuned.pt. 
Rekosrukcijski model je spremljen kao RecTransformer_DE_2_H_2_F_1024.pt, a autoenkoder ConvLSTM_E_32_H_196_L_128.pt. 
Rezultati ispitavanja ovih modela se nalaze u direktoriju results. Rezultati su prikazani kao slike i txt dataoteke. Slike prikazuju matrice zabune višekalsnog modela i ROC krivulju rekonstukcijskog modela. U .txt datotekama se nalze preciznost, odziv, F1 mjera i njihov makro prosjek. Slike i .txt datoteke u svojem imenu sadrže ime mopdela na kojeg se odnose.
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
$ python -m src.ML.testing.tag_transformer
```
```
$ python -m src.ML.testing.transformer
```
## Minimalni radni primjer i Reprodukcija rezultata
pokretanjem gore navedenih naredbi će se pokrenuti ispitivanje naučenih modela i prikazati rezultati tog ispitivanja koji su jednaki rezultatima u direktoriju results 
## Licence
Na ovaj kod se odnosi MIT licenca, a na GHALogs skup podataka se odnosi Creative Commons Attribution Share Alike 4.0 international licenca. Skup podatak se može pronaći na poveznici https://zenodo.org/records/10154920
