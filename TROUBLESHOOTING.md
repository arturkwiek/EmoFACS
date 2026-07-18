# Troubleshooting

Znane problemy środowiskowe przy uruchamianiu EmoFACS i sposoby ich obejścia.

---

## Zalecana konfiguracja (skrót)

Jeśli pracujesz w WSL i chcesz tylko, żeby zadziałało:

```powershell
# Windows, z katalogu repo
.\.venv\Scripts\python.exe scripts\serve_camera_windows.py
```

```bash
# WSL
ip route show default | awk '{print $3}'       # adres hosta Windows
python scripts/run_on_webcam.py --cam http://<IP_HOSTA>:8080/video
```

Kamerę obsługuje Windows, cały pipeline ML zostaje w WSL. Szczegóły i
uzasadnienie — [sekcja 3](#3-select-timeout--kamera-podłączona-ale-bez-klatek).

**Zastrzeżenie:** detekcja i zapis do bazy działają, ale **okno podglądu
w WSL nie wyświetla obrazu** ([sekcja 7](#7-podgląd-w-wsl-nie-wyświetla-obrazu)).
Jeśli potrzebujesz podglądu, uruchom całość natywnie — na Windows
([sekcja 4](#4-za-nowe-zależności-dla-py-feat-062)) albo na
natywnym Linuksie, gdzie żaden z tych problemów nie występuje.

**`usbipd` nie jest rozwiązaniem tego problemu** — przekaże urządzenie do
WSL, ale klatki i tak nie popłyną. Sekcje 1 i 2 opisują tę drogę, bo
prowadzą przez realne błędy, które po niej zobaczysz; sekcja 3 wyjaśnia,
dlaczego kończy się ślepym zaułkiem.

---

## 1. Kamera niedostępna w WSL2

### Objaw

Uruchomienie w WSL bez `--cam` (albo z indeksem kamery) kończy się
natychmiastowym błędem:

```
[error] --cam 0 is a local device index, which cannot work under WSL.
```

Skrypt wykrywa WSL i odmawia startu, zamiast pozwalać na kilkanaście
sekund pozornego działania. Przed dodaniem tego zabezpieczenia objaw
wyglądał tak — taki output można jeszcze zobaczyć przy starszych wersjach
skryptu albo gdy kamera nie jest przekazana przez `usbipd`:

```
[ WARN:0@13.734] global cap_v4l.cpp:913 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
[ERROR:0@13.734] global obsensor_uvc_stream_channel.cpp:158 getStreamChannelGroup Camera index out of range
[warning] Default backend failed — trying DirectShow.
...
[error] Cannot open camera index 0.
```

### Przyczyna

WSL2 **nie udostępnia urządzeń USB jako `/dev/video*`**. Kamera jest
widoczna wyłącznie po stronie Windows i musi zostać jawnie przekazana do
WSL przez `usbipd-win`. Dodatkowo `cv2.CAP_DSHOW` (DirectShow) to backend
wyłącznie windowsowy — w Linuksie fallback nie ma żadnych szans zadziałać,
więc komunikat „trying DirectShow” jest w WSL mylący.

### Diagnostyka

Na Windowsie (PowerShell jako administrator):

```powershell
usbipd list
```

```
BUSID  VID:PID    DEVICE                                      STATE
2-11   0c45:6a09  Integrated Webcam                           Shared
```

### Rozwiązanie A — przekazanie kamery do WSL (usbipd)

```powershell
usbipd bind   --busid 2-11
usbipd attach --wsl --busid 2-11
```

Weryfikacja w WSL:

```bash
ls /dev/video*
```

Jeśli urządzenia są widoczne, ale otwarcie nadal zawodzi:

```bash
sudo modprobe uvcvideo
sudo usermod -aG video "$USER"   # wymaga ponownego zalogowania sesji
```

> **Uwaga:** pojawienie się `/dev/video0` **nie oznacza działającej
> kamery.** Urządzenie da się otworzyć, ale strumień wideo i tak nie
> ruszy — patrz [sekcja 3](#3-select-timeout--kamera-podłączona-ale-bez-klatek).

### Rozwiązanie B — streaming z Windows do WSL (zalecane)

Windows czyta klatki i wystawia je jako MJPEG po HTTP, WSL je konsumuje.
Patrz [sekcja 3](#3-select-timeout--kamera-podłączona-ale-bez-klatek).

### Rozwiązanie C — uruchomienie całości natywnie na Windows

```powershell
py scripts/run_on_webcam.py
```

Omija cały problem przekazywania USB. Wymaga jednak działającego
środowiska po stronie Windows — patrz [sekcja 4](#4-za-nowe-zależności-dla-py-feat-062).

---

## 2. `Device busy (exported)` przy `usbipd attach`

### Objaw

```
WSL usbip: error: Attach Request for 2-11 failed - Device busy (exported)
usbipd: warning: The device appears to be used by Windows; stop the software
        using the device, or bind the device using the '--force' option.
usbipd: error: Failed to attach device with busid '2-11'.
```

### Przyczyna

Kamera jest w danym momencie **zajęta przez oprogramowanie Windows**.
Urządzenie może być formalnie „Shared”, a mimo to pozostawać zablokowane.

Najczęstsi winowajcy:

- **Windows Hello — logowanie twarzą** (na laptopach Dell to zdecydowanie
  najczęstsza przyczyna; trzyma kamerę praktycznie stale i nie zwalnia jej
  w sposób przewidywalny),
- Teams, Zoom, Slack, przeglądarka z aktywną zakładką korzystającą z kamery,
- aplikacja Aparat / Camera,
- warstwa Dell ControlVault.

### Rozwiązanie

1. Zamknij wszystkie aplikacje korzystające z kamery.
2. Wymuś przejęcie urządzenia:

   ```powershell
   usbipd unbind --busid 2-11
   usbipd bind   --busid 2-11 --force
   usbipd attach --wsl --busid 2-11
   ```

3. Jeżeli `bind --force` zwróci:

   ```
   usbipd: warning: A reboot may be required before the changes take effect.
   ```

   — **to nie jest ostrzeżenie do zignorowania.** Wymuszony bind podmienia
   sterownik na stub USB/IP, a podmiana nie wchodzi w życie, dopóki system
   trzyma otwarty uchwyt do urządzenia. Kolejny `attach` przed restartem
   zakończy się dokładnie tym samym `Device busy (exported)`.

   Zrestartuj Windows, następnie:

   ```powershell
   usbipd attach --wsl --busid 2-11
   ```

4. Jeśli po restarcie błąd wraca — wyłącz logowanie twarzą:
   **Ustawienia → Konta → Opcje logowania → Rozpoznawanie twarzy (Windows Hello)**.

> **Uwaga:** `bind --force` podmienia sterownik urządzenia na poziomie
> systemu. Aby przywrócić stan wyjściowy, użyj `usbipd unbind --busid 2-11`.

### Powrót kamery do Windows

Po zabawie z usbipd Windows przestaje widzieć kamerę — aplikacje zgłaszają
`Cannot open camera index 0`, choć w WSL nic już nie jest podłączone.

**`detach` nie wystarcza.** Odłącza urządzenie od WSL, ale sterownik
pozostaje podmieniony na stub USB/IP, więc Windows nadal go nie obsłuży.
Potrzebne jest `unbind`, w PowerShell **jako administrator**:

```powershell
usbipd unbind --busid 2-11
usbipd list
```

W kolumnie `STATE` musi pojawić się `Not shared`. Dopóki widnieje tam
`Shared (forced)`, żadna aplikacja Windows kamery nie otworzy — ani
OpenCV, ani Aparat.

---

## 3. `select() timeout` — kamera podłączona, ale bez klatek

### Objaw

`usbipd attach` kończy się powodzeniem, `/dev/video0` istnieje, OpenCV
otwiera urządzenie bez błędu i skrypt startuje — po czym nie dostaje ani
jednej klatki:

```
[emolog] Logging to .../data/emolog.db (session #3). Use --no-log to disable.
Press 'q' to quit.
[ WARN:0@24.325] global cap_v4l.cpp:1048 tryIoctl VIDEOIO(V4L2:/dev/video0): select() timeout.
[warning] Empty frame received — waiting for the camera to recover...
[ WARN:0@34.436] global cap_v4l.cpp:1048 tryIoctl VIDEOIO(V4L2:/dev/video0): select() timeout.
...
```

### Przyczyna

To **ograniczenie protokołu USB/IP, nie błąd konfiguracji.**

Kamery UVC przesyłają obraz **transferami izochronicznymi** — trybem USB
zaprojektowanym pod stały strumień danych z gwarantowaną przepustowością,
bez retransmisji. USB/IP tuneluje przez sieć transfery typu control, bulk
i interrupt; izochroniczne albo nie są przenoszone wcale, albo rozsypują
się przy przepustowości typowej dla wideo.

Stąd ten konkretny obraz błędu: enumeracja urządzenia i odczyt deskryptorów
korzystają z transferów control, więc działają — `/dev/video0` powstaje,
`VIDIOC_QUERYCAP` przechodzi, `open()` się udaje. Dopiero właściwy
strumień potrzebuje izochronicznych i nigdy nie dociera, przez co
`select()` czeka do timeoutu w nieskończoność.

Żadna zmiana po stronie WSL tego nie naprawi. Obniżenie przepustowości
(MJPG zamiast YUYV, mniejsza rozdzielczość, niższy fps) bywa skuteczne na
prostych kamerach USB, ale przy kamerach zintegrowanych w laptopach
szanse są niewielkie.

### Rozwiązanie — streaming po sieci zamiast po USB

Zamiast przekazywać urządzenie USB, przekaż gotowy obraz. Windows odczytuje
klatki i wystawia strumień MJPEG po HTTP, WSL go konsumuje. Przez sieć idzie
skompresowany JPEG po TCP, więc transfery izochroniczne przestają być
potrzebne.

Podział odpowiedzialności:

| | Windows | WSL |
| --- | --- | --- |
| rola | odczyt klatek + serwer MJPEG | py-feat, AU, emocje, overlay, SQLite |
| zależności | wyłącznie `cv2` | bez zmian |

Po stronie Windows **nie jest potrzebny ani `torch`, ani `py-feat`**, więc
konflikt wersji z [sekcji 4](#4-za-nowe-zależności-dla-py-feat-062)
przestaje mieć znaczenie.

**Krok 1 — serwer na Windows** (PowerShell, z katalogu repo):

```powershell
.\.venv\Scripts\python.exe scripts\serve_camera_windows.py
```

Oczekiwany output:

```
[info] Camera 0 opened at 640x480.
[info] Waiting for the first frame...
[info] Serving MJPEG on http://0.0.0.0:8080/video
[info]   reachable at http://172.22.128.1:8080/video
[info] Press Ctrl+C to stop.
```

Serwer startuje dopiero po zakodowaniu pierwszej klatki, więc pojawienie
się linii `Serving MJPEG` oznacza, że kamera faktycznie działa. Zostaw
proces uruchomiony.

Jeśli zamiast tego zobaczysz `Cannot open camera index 0`, kamerę wciąż
trzyma stub USB/IP — patrz [sekcja 2](#powrót-kamery-do-windows).

**Krok 2 — adres hosta widziany z WSL:**

```bash
ip route show default | awk '{print $3}'
```

**Krok 3 — uruchomienie pipeline'u w WSL:**

```bash
python scripts/run_on_webcam.py --cam http://<IP_HOSTA>:8080/video
```

**Krok 4 — reguła zapory** (tylko jeśli krok 3 nie łączy).

Najpierw sprawdź, czy w ogóle jest co blokować — jeśli nikt nie nasłuchuje,
zapora nie ma z tym nic wspólnego:

```powershell
Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue |
    Format-Table LocalAddress, LocalPort, State
```

Pusty wynik oznacza, że serwer nie działa — wróć do kroku 1. Dopiero gdy
port nasłuchuje, a WSL nadal nie dociera, dodaj regułę. PowerShell
**jako administrator**:

```powershell
New-NetFirewallRule -DisplayName "EmoFACS MJPEG (WSL)" -Direction Inbound `
    -Protocol TCP -LocalPort 8080 -RemoteAddress 172.16.0.0/12 -Action Allow
```

Usunięcie reguły:

```powershell
Remove-NetFirewallRule -DisplayName "EmoFACS MJPEG (WSL)"
```

### Szybka diagnostyka

Test samego strumienia, bez angażowania py-feat:

```bash
curl -s -m 5 -o /dev/null -w "%{http_code}\n" http://<IP_HOSTA>:8080/video
```

| Wynik | Interpretacja |
| --- | --- |
| `200` | strumień działa — problem leży po stronie pipeline'u |
| zwis / `000` | ruch blokowany przez zaporę — patrz krok 4 |
| `connection refused` | serwer na Windows nie działa — patrz krok 1 |

Strumień można też obejrzeć w przeglądarce na Windows: `http://localhost:8080/`.

### Uwagi

- `--cam` przyjmuje indeks urządzenia **albo** URL. Przy URL-u skrypt
  celowo **nie** próbuje w razie niepowodzenia lokalnych indeksów 0–4 —
  ciche przełączenie się na inną kamerę ukrywałoby fakt, że strumień
  jest niedostępny.
- Streaming dokłada opóźnienie rzędu kilkudziesięciu ms (kompresja plus
  przesył). Przy `--every 3` i detekcji trwającej sekundy jest to
  pomijalne, ale nie zerowe.
- Serwer utrzymuje tylko najświeższą klatkę, więc wolny klient nie buduje
  zaległości — kosztem gubienia klatek, co przy tym zastosowaniu jest
  zachowaniem pożądanym.

---

## 4. Za nowe zależności dla `py-feat` 0.6.2

### Objaw

Dwa błędy, pojawiające się kolejno — drugi dopiero po naprawieniu
pierwszego:

```
ImportError: cannot import name 'read_video' from 'torchvision.io'
```

```
ImportError: cannot import name 'simps' from 'scipy.integrate'
```

### Przyczyna

`py-feat` 0.6.2 to wydanie sprzed kilku lat i korzysta z API, które
w nowszych wersjach zależności zostały usunięte:

- `torchvision.io.read_video` — usunięte po `torchvision` 0.21,
- `scipy.integrate.simps` — usunięte w SciPy 1.14 (zastąpione `simpson`).

Źródłem obu był `requirements.txt` bez górnych ograniczeń: `torch>=2.0`
i `scipy>=1.7.0` pozwalały pipowi pobrać najnowsze wydania.

### Rozwiązanie — sprawdzony zestaw wersji

Ten zestaw został **zweryfikowany** i uruchamia pipeline na Windows:

| Pakiet        | Wersja     |
| ------------- | ---------- |
| torch         | 2.6.0+cpu  |
| torchvision   | 0.21.0+cpu |
| scipy         | 1.13.1     |
| numpy         | 1.23.5     |
| opencv-python | 4.11.0.86  |
| py-feat       | 0.6.2      |

Limity są już zapisane w `requirements.txt`, więc świeża instalacja
powinna trafić w te wersje sama. Ręcznie:

```powershell
.\.venv-win\Scripts\python.exe -m pip install "torch==2.6.0" "torchvision==0.21.0" "scipy<1.14"
```

Weryfikacja:

```powershell
.\.venv-win\Scripts\python.exe -c "import cv2, torch, torchvision, feat; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| py-feat', feat.__version__)"
```

> **Upgrade `py-feat` nie jest opcją.** `0.6.2` to najnowsze dostępne
> wydanie — `pip index versions py-feat` kończy listę na tej wersji.
> Dlatego to zależności trzeba trzymać w ryzach, a nie py-feat podnosić.

**Alternatywa — pozostanie przy WSL**

Środowisko WSL ma komplet zależności w działających wersjach: `py-feat`,
`torch` i `cv2` importują się poprawnie, a po skonfigurowaniu streamingu
z [sekcji 3](#3-select-timeout--kamera-podłączona-ale-bez-klatek) detekcja
działa i zapisuje pomiary do bazy. Serwer kamery potrzebuje po stronie
Windows tylko `cv2`, więc konfliktu wersji nie trzeba wtedy naprawiać.

**Ograniczenie:** podgląd w WSL nie renderuje obrazu — patrz
[sekcja 7](#7-podgląd-w-wsl-nie-wyświetla-obrazu). Jeśli zależy Ci na
oknie z podglądem, wybierz opcję 1 lub 2 i pracuj na Windows.

---

## 5. Brakujące biblioteki Qt w WSL (`libSM.so.6`)

### Objaw

Detekcja przebiega, po czym program przerywa działanie przy próbie
otwarcia okna:

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in
"/home/vision/.local/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized.

Available platform plugins are: xcb.
Aborted (core dumped)
```

Komunikat jest mylący: plugin **został znaleziony**, ale nie daje się
załadować, bo brakuje biblioteki systemowej, od której zależy.

### Diagnostyka

Ta komenda wskazuje konkretną brakującą bibliotekę:

```bash
QT_DEBUG_PLUGINS=1 python scripts/run_on_webcam.py --cam http://<IP_HOSTA>:8080/video 2>&1 \
    | grep -iE "cannot load|cannot open shared|not found" | head -20
```

Przykładowy wynik:

```
Cannot load library .../cv2/qt/plugins/platforms/libqxcb.so:
(libSM.so.6: cannot open shared object file: No such file or directory)
```

### Rozwiązanie

```bash
sudo apt install -y libsm6 libice6 libxext6
```

Jeśli po instalacji pojawi się kolejna brakująca biblioteka — to normalne.
`libqxcb.so` zgłasza tylko **pierwszą** nieznalezioną zależność, więc
diagnostykę trzeba powtórzyć. Pozostałe biblioteki, które bywają potrzebne:

```bash
sudo apt install -y libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0
```

Sprawdź też, czy WSLg wystawia serwer X — powinno zwrócić `:0`:

```bash
echo $DISPLAY
```

---

## 6. `cv2` zainstalowane poza `.venv`

W tym środowisku `cv2` ładuje się z katalogu użytkownika, nie z
wirtualnego środowiska:

```bash
python -c "import cv2; print(cv2.__file__, cv2.__version__)"
# /home/vision/.local/lib/python3.10/site-packages/cv2/__init__.py 4.11.0
```

Samo w sobie działa, ale warto o tym wiedzieć: `pip install` wewnątrz
`.venv` nie wpłynie na to, która wersja OpenCV jest faktycznie używana,
bo katalog użytkownika ma pierwszeństwo w `sys.path`. Przy diagnozowaniu
problemów z OpenCV zawsze sprawdzaj `cv2.__file__`, a nie `pip list`.

---

## 7. Podgląd w WSL nie wyświetla obrazu

### Objaw

Po naprawieniu Qt ([sekcja 5](#5-brakujące-biblioteki-qt-w-wsl-libsmso6))
okno się otwiera, ale pozostaje puste — widać ramkę, pasek narzędzi
OpenCV i mały czarny prostokąt w rogu, natomiast nie ma obrazu z kamery.
W terminalu detekcja przebiega normalnie, kolejno po ~2 s na klatkę.

### Co zostało sprawdzone

- Strumień jest poprawny. Odczyt pięciu klatek bezpośrednio przez
  `cv2.VideoCapture(url)` zwraca `ret: True` i `shape: (480, 640, 3)`.
- Serwer działa — obraz jest widoczny w przeglądarce na Windows pod
  `http://localhost:8080/`.
- `DISPLAY` jest ustawione na `:0`, plugin `xcb` ładuje się poprawnie.

Czyli klatki docierają w całości, a mimo to `cv2.imshow` ich nie rysuje.

### Status

**Nierozwiązane.** Problem leży w warstwie GUI OpenCV pod WSLg, nie
w pipelinie. Możliwe kierunki, żadnego nie zweryfikowano:

- inna wersja OpenCV (`opencv-python` z `~/.local` — patrz
  [sekcja 6](#6-cv2-zainstalowane-poza-venv)),
- wymuszenie backendu: `QT_QPA_PLATFORM=xcb` albo `wayland`,
- zewnętrzny serwer X (VcXsrv) zamiast WSLg.

### Obejście

Analiza nie wymaga podglądu — pomiary trafiają do bazy niezależnie od
tego, czy okno cokolwiek pokazuje. Jeśli podgląd jest potrzebny, prościej
uruchomić całość natywnie: na Windows (patrz
[sekcja 4](#4-za-nowe-zależności-dla-py-feat-062)) albo na
natywnym Linuksie, gdzie ani ten problem, ani przekazywanie kamery przez
USB/IP w ogóle nie występują.

---

## 8. Współdzielony katalog `.venv` (Windows + Linux)

### Kontekst

Katalog `.venv/` zawiera **dwa nakładające się środowiska wirtualne**:

```
.venv/
├── bin/                          # launchery Linux/WSL
├── Scripts/                      # launchery Windows (python.exe, pip.exe)
├── lib/
│   ├── site-packages/            # pakiety Windows  (Windows widzi to jako Lib/)
│   └── python3.10/site-packages/ # pakiety Linux/WSL
└── pyvenv.cfg                    # obecnie wskazuje na interpreter Windows
```

Pakiety obu systemów trafiają do rozdzielnych podkatalogów, więc oba
środowiska działają obok siebie. Na Windowsie `lib` i `Lib` to jednak
**ta sama ścieżka** (system plików jest niewrażliwy na wielkość liter),
przez co układ jest mylący przy przeglądaniu.

Wybór interpretera:

```powershell
# Windows
.\.venv\Scripts\python.exe scripts/run_on_webcam.py
```

```bash
# WSL
./.venv/bin/python scripts/run_on_webcam.py
```

### Ryzyko

Wspólny `pyvenv.cfg` jest jeden i wskazuje na ostatnio utworzone
środowisko. Ponowne wywołanie `python -m venv .venv` z któregokolwiek
systemu nadpisze go i może zdezorientować drugą stronę.

### Stan obecny

Środowisko dla Windows zostało odtworzone w osobnym katalogu
**`.venv-win/`** i to jego należy używać:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_on_webcam.py --every 10
```

Stary `.venv/` nadal zawiera działające środowisko Linux/WSL
(`./.venv/bin/python`). Jego część windowsowa jest zepsuta — patrz
sekcja 9.

**Zalecenie:** przenieść też stronę linuksową do `.venv-wsl/`
i dodać wszystkie trzy katalogi do `.gitignore`.

---

## 9. Zepsute launchery po przemianowaniu katalogu projektu

### Objaw

`pip` zgłasza ścieżkę do katalogu, którego nie ma:

```
Fatal error in launcher: Unable to create process using
'"...\Facial-Emotion-Recognition\1\.venv\Scripts\python.exe"
 "...\Facial-Emotion-Recognition\EmoFACS\.venv\Scripts\pip.exe"':
Nie można odnaleźć określonego pliku.
```

Zwróć uwagę na katalog `1\` — projekt nazywał się tak w momencie
tworzenia środowiska i został później przemianowany na `EmoFACS`.

Drugi, cichszy objaw: mimo `(.venv)` w znaku zachęty polecenie `python`
uruchamia **interpreter systemowy**, nie ten ze środowiska:

```
C:\Users\Dell\AppData\Local\Programs\Python\Python310\python.exe
```

### Przyczyna

Pliki `.exe` w `Scripts\` (`pip.exe`, `python.exe` i pozostałe) mają
ścieżkę do interpretera **wkompilowaną w plik binarny**. Przenosiny ani
zmiana nazwy katalogu ich nie aktualizują. Skrypt aktywacyjny również
zawiera ścieżki bezwzględne, więc `activate` przestaje działać poprawnie
— stąd znak zachęty sugerujący aktywne środowisko, mimo że `PATH`
prowadzi gdzie indziej.

### Dlaczego to groźne

Ten błąd potrafi **cicho unieważnić instalację pakietów**. Wywołanie
`pip install "torch==2.6.0"` w takim środowisku albo padnie na launcherze,
albo zainstaluje pakiet do interpretera systemowego — a `pip list`
uruchomiony poprawną ścieżką pokaże niezmienione wersje. Wygląda to jak
instalacja, która „nie zadziałała", choć w rzeczywistości trafiła w inne
miejsce.

### Rozwiązanie

**Zawsze wywołuj interpreter pełną ścieżką**, nigdy przez `pip` czy
`python`:

```powershell
.\.venv-win\Scripts\python.exe -m pip install <pakiet>
.\.venv-win\Scripts\python.exe -m pip list --format=freeze
```

Środowiska po przenosinach **nie da się naprawić w miejscu** — trzeba je
odtworzyć:

```powershell
py -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install --upgrade pip
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
```

### Weryfikacja

Ta komenda musi wskazywać na katalog projektu, a nie na `AppData`:

```powershell
.\.venv-win\Scripts\python.exe -c "import sys; print(sys.executable)"
```
