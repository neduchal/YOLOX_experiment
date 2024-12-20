# Popis úprav oproti originálnímu repozitáři

Repozitář obsahuje upravenou verzi sítě YOLOX vracející kromě výstupu i featury pro každou detekci. Používá se identickým způsobem jako původní repozitář. Navíc má skript na generování INT8 TensorRT optimalizované sítě, docker kontejner na testování a skript, který provede inferenci všech obrázků v zadané cestě a výsledky uloží do textového souboru. 

Úprava by měla fungovat na všechny verze sítě YOLOX.

## Upravené soubory:

```
yolox/models/yolox_head.py
yolox/utils/boxes.py
tools/demo.py
```

## Přidané soubory
```
tools/trt_int8.py
tools/process.py
docker/Dockerfile
```

# Instalace a spuštění základní sítě 
<details>
<summary>Installation</summary>

Step1. Install YOLOX from source.
```shell
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .  # or  python3 setup.py develop
```

</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table.

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python tools/demo.py image -n yolox-l -c /path/to/your/yolox_l.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu] [--trt] [--fp16]
```
</details>
<details>
<summary>Process</summary>

Step1. Download a pretrained model from the benchmark table.

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python tools/process.py process -n yolox-l -c /path/to/your/yolox_l.pth --path /path/to/image/directory --conf 0.25 --nms 0.45 --tsize 640 --device [cpu/gpu] [--trt] [--fp16]
```
</details>

# Převod do TRT

   If you want to convert our model, use the flag -n to specify a model name:
   ```shell
   python tools/trt.py -n <YOLOX_MODEL_NAME> -c <YOLOX_CHECKPOINT>
   ```
   For example:
   ```shell
   python tools/trt.py -n yolox-s -c your_ckpt.pth
   ```
   <YOLOX_MODEL_NAME> can be: yolox-nano, yolox-tiny. yolox-s, yolox-m, yolox-l, yolox-x.

# Převod do TRT INT8

If you want to convert our model, use the flag -n to specify a model name:
```shell
python tools/trt.py -n <YOLOX_MODEL_NAME> -c <YOLOX_CHECKPOINT> -d <cesta_k_datasetu–
```
For example:
```shell
python tools/trt.py -n yolox-s -c your_ckpt.pth
```
<YOLOX_MODEL_NAME> can be: yolox-nano, yolox-tiny. yolox-s, yolox-m, yolox-l, yolox-x.

<cesta_k_datasetu> je cesta ke slozce s obrazky.

## Co síť vrací:

Síť vrací pole tensorů o velikosti [pocet_detekci, 263]
Jednotlive itemy jsou postupne

| Pozice | Význam | Popis            |
|--------|--------|------------------|
| 0      | x1     | pozice x1 bboxu  |
| 1      | y1     | pozice y1 bboxu  |
| 2      | x2     | pozice x2 bboxu  |
| 3      | y2     | pozice y2 bboxu  |
| 4      | obj_score | objektove score | 
| 5      | cls_score | score tridy |
| 6      | class | trida |
| 7-262      | features | featury pro danou detekci |

## Co vraci process.py

Výstupem je soubor, který má na jednotlivých řádkách data oddělená čárkou.

image_name(bez typu souboru), x1, y1, x2, y2, obj_score, cls_score, class, time

POZOR!: Čas zpracování obrázku se ukládá ke všem detekcím. Je tedy potřeba ho při načítání všech detekcí daného obrázku zpracovat jen 1.


## Výsledky na testovacim datasetu:

Testováno na 279 obrázcích 640x640 celkově obsahujících 1911 detekcí v GT datech.

| Síť | Precision | Recall | F1 | AVG time [s] | FPS | Správných detekcí |
| --- | ---------- | ------ | -- | -------- | ----- | ---------------- |
| Original | 0.546 | 0.801  | 0.650 | 0.0278 | 36 | 1530
| FP16 | 0.547 | 0.801  | 0.650 | 0.0295 | 34 | 1531
| TRT FP32 | 0.546 | 0.801  | 0.650 | 0.0203 | 49 | 1530
| TRT INT8 | 0.661 | 0.679  | 0.670 | 0.0108 | 93 | 1297

## Build a spuštení dockeru 

Docker image se vytvoří ve složce docker pomocí příkazu 

```shell
docker build -t yolox_feats:latest . 
```

Případně pokud nechceme použít cache kvůli znovu zbuildění repozitáře:

```shell
docker build -t yolox_feats:latest . 
```

Tento krok většinou ale není nutný, jelikož stačí kontejner zapnout a ve složce YOLOX zavolat 

```shell
git pull
```

Samotné spuštění vypadá následovně:

```shell
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --privileged --name yolox -e DISPLAY=$DISPLAY -it --rm -v /cesta/k/modelum/:/workspace/models/ -v /cesta/k/datum:/workspace/data/ yolox_feats:latest /bin/bash
```

Případně je možné Docerfile upravit a modely nechat stahnout při jeho buildu. Stačí odkomentovat řádky 15-19.

Vše v kontejneru je pak k dispozici pod /workspace. Zejména pak složka YOLOX, kde jsou k dispozici výše uvedené skripty.