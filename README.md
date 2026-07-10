# Primjena AIOps metodologije za detekciju anomalija u GitHub Actions cjevovodima
 Dominik Ožvald, Vlado Sruk\
 Ovaj repozitorij sadrži python kod za arhitekturu, učenje i ispitivanje dubokih modela za detekciju anomalija u zapisima nastalim u GitHub Actions cjevovodu koji je dio diplomksog rada Primjena AIOps metodologije za detekciju anomalija u GitHub Actions cjevovodima Dominika Ožvalda godine 2026. 
 Takvi zapisi nastaju tijekom izvođenja CI/CD cjevovoda i mogu sadržavati obrasce koji upućuju na kvarove, degradacije pouzdanosti, konfiguracijske probleme ili sigurnosne rizike, čak i kada radni tok ne završava nužno neuspjehom.
 Osim koda za modele sadrži i kod koji je korišten za generiranje umjetnog skupa podataka na kojem su modeli učeni. Ovaj kod je vazan uz diplomski rad koji se nalazi u direkotirju docs.
Definirane su četiri vrste anomalija promatrane u radu: nestabilni ispiti, konfiguracijski pomak, tihi neuspjesi i sigurnosne anomalije. Za potrebe učenja i ispitivanja izrađen je skup podataka koji simulira izgradnju i ispitivanje poslužiteljske aplikacije u koji su ubačeni različiti tipovi anomalija, uz kontrolirano umetanje različitih tipova anomalija. 
Dodatno je pristup provjeren na označenom podskupu stvarnih zapisa iz GHALogs skupa podataka. Izlučivanje značajki iz zapisa se radi pomoću autoenkodera koji koristi LSTM mrežu i jednodimenzionalnu konvoluciju za kodiranje zapisa u vektore jednake duljine. Dobiveni vektori se grupiraju u pomoću klizni prozora te prosljeđuju u dva modela temeljena na transformeru. Jedan model uči na označenim podacima i razvrstava anomalije po tipovima, a drugi rekonstruira ulazne vektore i određuje anomaliju po razlici između ulaznih i izlaznih vektora. 
## Struktura repozitorija
Direktorij src predstavlja jezgru implementacije i podijeljen je na dva primarna poddirektorija: ML (strojno učenje) i data_generation (generiranje umjetnih podataka).\
Src/ML sadrži logiku za definiranje arhitektura dubokih modela, njihovo učenje i ispitivanje. U njemu se nalaze sljedeće datoteke i direktoriji.
- ML/models/ - Arhitektura modela
  - convlstm.py: Sadrži implementaciju autoenkodera
  - transformer.py: Sadrži definicije rezrede (engl. class) za višeklasni i rekonstrukcijski model transformera
  - embedder.py: Sadrži klasu omotača (engl. wrapper) oko autoenkodera koja služi za pravilno oblikovanje i pripremu tenzora prije prosljeđivanja koderu autoenkodera.
- ML/utils/ - Pomoćne funkcije i procesiranje zapisa:
  - data.py: Sadrži funkcije za pretprocesiranje tekstualnih zapisa
  - datasets.py: Implementira prilagođene razrede za učitavanje podataka
  - embeddings.py: Sadrži klasu zaduženu za dodjeljivanje numeričkih vrijednosti ASCII znakovima unutar zapisa
  - train.py: Sadrži petlje za učenje modela
- ML/training/ - skripte za pokretanje učenja 
  - convlstm.py: Pokreće proces učenja autoenkodera.
  - transformer.py: Koristi se za treniranje rekonstrukcijskog modela transformera.
  - tag_transformer.py: Pokreće učeanje višeklasnog modela transformera
  - tuning.py: Sadrži logiku za dodatno učenje višeklasnog modela transformera na stvarnim podacima
- ML/testing/ Evaluacijske skripte:
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
$ python -m src.ML.testing.tag_transformer
```
```
$ python -m src.ML.testing.transformer
```
## Minimalni radni primjer i Reprodukcija rezultata
Pokretanjem navedenih naredba instalirati će se potrebni python paket za ispitivanje. Rezultat ispitivanja višeklasnog će biti matrica zabune kao ona u direktoriju results zajedno s ispisom preciznosti, odziva, F1 mjere i njihovog makro prosjeka na standardni izlaz. Kod ispitivanja rekosntrukcijskog modela rezultat će bit slika ROC krivulje kao ona u direktoriju results. 
## Licence
Na ovaj kod se odnosi MIT licenca, a na GHALogs skup podataka se odnosi Creative Commons Attribution Share Alike 4.0 international licenca. Skup podatak se može pronaći na poveznici https://zenodo.org/records/10154920
