# Impróżmenty — 2026-07-17

Zestaw poprawek wydajności i czytelności nakładki. Zmienione pliki:
`src/detection/detector.py` oraz `scripts/run_on_webcam.py`. Logika
pipeline'u (AU → walencja/pobudzenie → emocje) pozostała bez zmian.

---

## 1. Wydajność — `src/detection/detector.py`

**Problem:** każda klatka przechodziła podwójną konwersję kolorów
(BGR → RGB → BGR), była zapisywana jako stratny JPEG do pliku
tymczasowego, a plik był zapisywany, gdy uchwyt `NamedTemporaryFile`
wciąż był otwarty (na Windowsie ponowne otwarcie otwartego pliku bywa
zawodne).

**Zmiany:**

- Usunięta podwójna konwersja kolorów — `cv2.imwrite` i tak oczekuje
  BGR, więc klatka jest zapisywana bezpośrednio, bez żadnej konwersji.
- JPEG → **PNG** (bezstratny, kompresja na poziomie 1): ekstrakcja AU
  nie jest już zaburzana artefaktami kompresji, a szybki poziom
  kompresji utrzymuje czas kodowania w pojedynczych milisekundach.
- Uchwyt pliku tymczasowego jest **zamykany przed** `cv2.imwrite`
  (`tmp.close()` najpierw, potem zapis) — poprawne i przewidywalne
  zachowanie na Windowsie.

## 2. Wydajność — pomijanie klatek w `scripts/run_on_webcam.py`

**Problem:** pełna detekcja py-feat trwa na CPU rzędu sekund na klatkę,
przez co podgląd zamierał.

**Zmiany:**

- Nowy argument `--every N` (domyślnie `3`): detekcja uruchamia się co
  N-tą klatkę, a pomiędzy nimi nakładka jest rysowana z ostatniego
  wyniku. Podgląd jest płynniejszy, a wskaźniki aktualizują się
  okresowo.
- `--every 1` przywraca poprzednie zachowanie (analiza każdej klatki).
- Przed pierwszym wynikiem nakładka pokazuje `[warming_up]` zamiast
  śmieci.

## 3. Czytelność nakładki — `scripts/run_on_webcam.py`

**Problem:** procenty rysowane kolorem emocji — szary „Neutral" na
szarym panelu był prawie niewidoczny (widoczne na zrzucie ekranu);
tekst V/A w kolorze (130,130,130) również ginął na jasnym tle.

**Zmiany:**

- Panel przyciemniony: krycie 0.45 → **0.70** i ciemniejsze tło
  (20,20,20) → (15,15,15) — jasny tekst nie zlewa się z prześwietloną
  sceną za panelem.
- Procenty rysowane **białym tekstem z cienkim ciemnym obrysem**
  (dwuwarstwowe `putText`) — czytelne niezależnie od koloru emocji.
  Kolor emocji nadal widać na pasku.
- Tekst `V:… A:…` rozjaśniony (130,130,130) → (220,220,220), również
  z ciemnym obrysem.

## 4. Porządki — `scripts/run_on_webcam.py`

- Usunięty **zduplikowany `import math`** wewnątrz `draw_overlay`
  (moduł jest już importowany na górze pliku).

---

## Weryfikacja

- `python -m py_compile` przechodzi dla obu zmienionych plików.
- Zmiany nie dotykają interfejsów: `FaceDetector.detect(img_bgr) → preds`
  oraz `pipeline.run_on_frame(frame) → dict` bez zmian, więc
  `run_on_image.py` działa jak dotychczas.

## Poza zakresem (do zrobienia następnym razem)

- ~~**Wagi modelu**: użyć wbudowanych `preds.emotions` z py-feat.~~
  ✅ Zrobione — patrz część 2 poniżej.
- Detekcja w osobnym wątku (zamiast pomijania klatek) — pełna płynność
  podglądu.
- Wyjście z pętli po serii pustych klatek (obecnie `continue` może się
  zapętlić po odłączeniu kamery).

---

# Część 2 — prawdziwe emocje z py-feat (2026-07-17, później)

**Problem:** py-feat przy każdej detekcji liczy pełny, **wytrenowany**
klasyfikator emocji (`preds.emotions`), ale pipeline te dane wyrzucał
i pokazywał emocje wyliczone z V/A pochodzącego z **niewytrenowanego**
regresora (losowe wagi → wskaźniki były szumem skupionym wokół
Neutral).

Zmienione pliki: `src/aus/au_extractor.py`, `src/pipeline.py`,
`scripts/run_on_webcam.py`.

## 1. `src/aus/au_extractor.py` — nowa metoda `extract_emotions()`

- Wyciąga `preds.emotions` (klasyfikator py-feat) i mapuje nazwy kolumn
  na wyświetlane: anger→Angry, disgust→Disgusted, fear→Fearful,
  happiness→Happy, sadness→Sad, surprise→Surprised,
  neutrality/neutral→Neutral.
- Zwraca `dict[str, float]` z prawdopodobieństwami albo `None`
  (brak twarzy / NaN / brak danych).

## 2. `src/pipeline.py` — emocje w wyniku

- Wynik `run_on_frame()` / `run_on_image()` zawiera teraz dodatkowo:
  - `emotions` — realne prawdopodobieństwa z py-feat,
  - `va_trained` — `True` tylko gdy załadowano wytrenowane wagi
    regresora (informuje konsumenta, czy V/A ma sens).
- `run_on_image.py` automatycznie wypisuje te pola w JSON-ie
  (bez zmian w samym skrypcie).

## 3. `scripts/run_on_webcam.py` — nakładka na realnych danych

- Słupki pokazują teraz **realne prawdopodobieństwa** z py-feat
  (7 emocji), a nie rozmytą przynależność gaussowską z V/A.
- **V/A bez wag regresora**: wyliczane jako średnia ważona centrów
  emocji (centroid prawdopodobieństw) — kropka na cyrkumpleksie
  reaguje na prawdziwe emocje zamiast dryfować przy zerze.
- Etykieta źródła przy odczycie: `[emotions]` (centroid z klasyfikatora),
  `[regressor]` (wytrenowany MLP) lub `[untrained!]` (fallback na losowe
  wagi, gdy emocji brak).
- Stary tryb rozmyty V/A→emocje pozostał jako fallback, gdy `emotions`
  jest niedostępne.

## Weryfikacja

- Testy jednostkowe `extract_emotions()` na sztucznych obiektach Fex
  (pandas): mapowanie nazw, wariant kolumny `neutral`, ignorowanie
  nieznanych kolumn, NaN→None, pusty wynik→None — wszystkie przeszły.
- Matematyka centroidu sprawdzona: klatka zdominowana przez Happy
  (0.71) daje V=+0.38, A=+0.20 — sensowna pozycja na cyrkumpleksie.
- `python -m py_compile` przechodzi dla wszystkich trzech plików.

## Czego się spodziewać po uruchomieniu (część 2)

- Słupki będą **zdecydowane** (np. Happy 70–90% przy uśmiechu) zamiast
  płaskiego rozkładu 8–35%.
- Pojawi się maks. 7 emocji (klasyfikator py-feat nie zna Calm
  i Excited — te wrócą po wytrenowaniu własnego regresora V/A).
- Po wytrenowaniu i podaniu `--weights models/emotion_regressor.pth`
  V/A przełączy się automatycznie na regresor (`[regressor]`).

---

# Część 3 — logowanie nastroju i analiza w czasie (2026-07-17)

Nowa funkcjonalność: każdy pomiar trafia do lokalnej bazy SQLite, dzięki
czemu nastrój można analizować dzień po dniu i korelować z czynnikami
zewnętrznymi (sen, kofeina, sport, stres...).

Nowe pliki: `src/emolog/logger.py`, `scripts/log_factors.py`,
`scripts/analyze_history.py`, `.gitignore`.
Zmieniony: `scripts/run_on_webcam.py`.

## 1. Baza danych — `data/emolog.db` (SQLite, WAL)

- **`sessions`** — jedna sesja przed kamerą = jeden wiersz (start, koniec,
  źródło, czy regresor miał wagi).
- **`measurements`** — pełny wektor każdego pomiaru: timestamp, V/A,
  7 prawdopodobieństw emocji, emocja dominująca oraz **20 surowych AU
  (JSON)** — te ostatnie posłużą w przyszłości do treningu własnego
  regresora AU→V/A.
- **`daily_factors`** — jeden wiersz na dzień: sen (godziny + jakość),
  kofeina, alkohol, minuty ruchu, stres 1–5, samoocena nastroju 1–5,
  notatka.
- Folder `data/` jest w `.gitignore` — dane osobiste nie trafią do repo.

## 2. Logowanie z kamery — `run_on_webcam.py`

- Logowanie **domyślnie włączone**; `--no-log` wyłącza, `--db` zmienia
  ścieżkę bazy. Zapis tylko przy faktycznej detekcji (nie przy
  przerysowaniach z pomijania klatek); klatki bez twarzy są pomijane.
- Sesja domykana w `finally` — działa też przy Ctrl+C.

## 3. Dziennik czynników — `scripts/log_factors.py`

- Interaktywnie: `python scripts/log_factors.py` (Enter pomija pytanie).
- Flagami: `python scripts/log_factors.py --sleep 7.5 --stress 2 --note "..."`.
- Można uzupełniać wstecz: `--date 2026-07-16`. Wartości nadpisują tylko
  podane pola (upsert).

## 4. Analiza — `scripts/analyze_history.py`

- Tabela dzienna: liczba pomiarów, średnia walencja/pobudzenie,
  **mood_index** = P(Happy) − P(Sad+Angry+Fearful+Disgusted),
  **volatility** = odchylenie std walencji w ciągu dnia (stabilność
  emocjonalna), emocja dominująca.
- Korelacje Pearsona: metryki dzienne × czynniki (wymaga ≥3 wspólnych
  dni; oznaczenia ** |r|≥0.5, *** |r|≥0.7, z ostrzeżeniem o małej próbie).
- Wykres PNG (`data/mood_report.png`): przebieg walencji/pobudzenia/
  mood_index z pasmem zmienności + dzienne złożenie emocji (słupki).
- `--days N` ogranicza do ostatnich N dni.

## Weryfikacja

- Test end-to-end na 7 dniach danych syntetycznych (280 pomiarów,
  3 czynniki dziennie, zapis przez realne API loggera i CLI dziennika):
  wszczepiona zależność sen→szczęście odzyskana z r=+1.00,
  stres→walencja r=−0.97; wyniki błędne (`no_face_detected`) poprawnie
  ignorowane; wykres wygenerowany i obejrzany.
- `python -m py_compile` przechodzi dla wszystkich plików.

## Sugerowany rytuał

1. Rano/wieczorem: `python scripts/log_factors.py` (30 sekund).
2. Normalnie używaj `run_on_webcam.py` — loguje sam.
3. Raz na kilka dni: `python scripts/analyze_history.py` i rzut oka na
   `data/mood_report.png`.

Po ~2–3 tygodniach korelacje zaczną być wiarygodne (n≥14).

---

# Część 4 — odporność na zawieszenie kamery (2026-07-17, wieczorem)

**Problem (wystąpił w praktyce):** backend MSMF na Windowsie przestał
oddawać klatki (`can't grab frame. Error: -1072873822` / 0xC00D3EA2 —
typowo gdy inna aplikacja przejmie kamerę albo sterownik się zatnie),
a pętla główna wpadła w nieskończone `[warning] Empty frame received` —
trzeba było zabić proces Ctrl+C.

**Zmiany w `scripts/run_on_webcam.py`:**

- Puste klatki nie spamują już logiem — jedno ostrzeżenie na serię,
  z krótkim odczekaniem (0.1 s) zamiast ciasnej pętli.
- Po ~3 s ciągłych pustych klatek następuje **automatyczny reconnect**
  (release + ponowne otwarcie kamery), maksymalnie 3 próby.
- Po wyczerpaniu prób program **wychodzi czysto** — sesja logowania
  emocji i tak jest domykana w `finally`, więc dane nie przepadają.
- Nowa funkcja `open_camera()`: gdy domyślny backend (MSMF) nie otwiera
  kamery, próbowany jest **DirectShow** (`CAP_DSHOW`) — stabilniejszy
  na wielu konfiguracjach Windows.

**Weryfikacja:** `python -m py_compile` OK. Scenariusz zawieszenia
symulowany logicznie (streak → reconnect → wyjście); realny test wymaga
fizycznej kamery.
