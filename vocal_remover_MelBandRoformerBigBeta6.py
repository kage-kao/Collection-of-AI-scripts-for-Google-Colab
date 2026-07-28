#!/usr/bin/env python3
"""
Colab-скрипт: удаление вокала Mel-Band Roformer Big Beta 6 + загрузка на tempshare.

Как использовать:
  1. Открой https://colab.research.google.com
  2. Меню Runtime → Change runtime type → T4 GPU (бесплатно)
  3. Вставь всё содержимое этого файла в одну ячейку
  4. Нажми ▶ (Shift+Enter)
  5. Вставь прямую ссылку на скачивание исходного аудиофайла
  6. В конце получишь ссылку на минус, действующую 7 дней
"""
import subprocess, os, glob, sys, zipfile, re
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import yaml as _yaml

# ============ НАСТРОЙКИ ============
MODEL = "melband-roformer-big-beta6"  # лучший по качеству
# Запасной вариант (если Big Beta 6 упадёт по памяти): "melband-roformer-kim-vocals"
TEMPSHARE_DURATION = 7  # срок действия ссылки, дней
OUTPUT_BITRATE = "320k"
# ==================================


def run(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        sys.exit(f"❌ Команда завершилась с ошибкой: {cmd}")
    return r


print("=" * 60)
print("  Mel-Band Roformer — удаление вокала (Big Beta 6)")
print("=" * 60)

# 1. Установка зависимостей
print("\n[1/8] Установка пакетов...")
run("pip install -q -U melband-roformer-infer soundfile")

# 2. Запрашиваем и скачиваем трек
print("\n[2/8] Скачивание трека...")
while True:
    TRACK_URL = input("Вставь прямую ссылку на скачивание трека (mp3/wav/flac/m4a или ZIP): ").strip()
    parsed_url = urlparse(TRACK_URL)
    if parsed_url.scheme in ("http", "https") and parsed_url.netloc:
        break
    print("❌ Нужна полноценная ссылка, начинающаяся с http:// или https://")

try:
    request = Request(TRACK_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response, open("input_track", "wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
except Exception as error:
    sys.exit(f"❌ Не удалось скачать трек: {error}")

assert os.path.getsize("input_track") > 1000, "Трек не скачался или файл слишком маленький"
print(f"   ✓ Скачано: {os.path.getsize('input_track') // 1024} KB")

# Если это ZIP — распаковываем, иначе берём как есть
if zipfile.is_zipfile("input_track"):
    with zipfile.ZipFile("input_track") as z:
        z.extractall("input_extracted")
    mp3s = glob.glob("input_extracted/**/*", recursive=True)
    mp3s = [m for m in mp3s if m.lower().endswith((".mp3", ".wav", ".flac", ".m4a"))]
else:
    mp3s = ["input_track"]
assert mp3s, "В архиве нет аудио"
run(f"cp '{mp3s[0]}' track_source")
print(f"   Найден файл: {mp3s[0]}")

# 3. Конвертация в WAV 44.1kHz стерео
print("\n[3/8] Конвертация в WAV...")
os.makedirs("input_wav", exist_ok=True)
run("ffmpeg -y -i track_source -ar 44100 -ac 2 input_wav/track.wav -loglevel error")

# 4. Скачиваем модель
print(f"\n[4/8] Скачивание модели {MODEL}...")
run(f"melband-roformer-download --model {MODEL} --output-dir ./models")

# 5. Находим конфиг и веса, нормализуем структуру (Big Beta 6 кладёт chunk_size
#    под audio:, а код ждёт его под inference:; плюс !!python/tuple ломает safe_load)
model_dir = f"models/{MODEL}"
ckpts = glob.glob(f"{model_dir}/*.ckpt")
yamls = glob.glob(f"{model_dir}/*.yaml")
assert ckpts and yamls, f"❌ Файлы модели не найдены в {model_dir}"
cfg_src, weights = yamls[0], ckpts[0]

with open(cfg_src) as f:
    raw = f.read().replace("!!python/tuple", "")
data = _yaml.safe_load(raw) or {}

inf = data.setdefault("inference", {})
audio = data.get("audio", {})
if "chunk_size" not in inf and "chunk_size" in audio:
    inf["chunk_size"] = audio["chunk_size"]
    print(f"   Скопирован audio.chunk_size={audio['chunk_size']} → inference.chunk_size")
if "num_overlap" not in inf:
    inf["num_overlap"] = 2
    print("   Добавлен inference.num_overlap=2 (по умолчанию)")

cfg = cfg_src.replace(".yaml", "_normalized.yaml")
with open(cfg, "w") as f:
    _yaml.safe_dump(data, f, sort_keys=False)
print(f"   Конфиг нормализован → {cfg}\n   Веса:   {weights}")

# 6. Инференс на GPU
print("\n[6/8] Разделение (Big Beta 6 на GPU)...")
run(
    f"melband-roformer-infer "
    f"--config_path '{cfg}' --model_path '{weights}' "
    f"--input_folder input_wav --store_dir out --device cuda:0"
)

# 7. Конвертация инструментала в MP3 320 kbps
print("\n[7/8] Конвертация в MP3 320 kbps...")
inst = glob.glob("out/*_instrumental.wav")
assert inst, "❌ instrumental.wav не найден в выводе"
run(f"ffmpeg -y -i '{inst[0]}' -c:a libmp3lame -b:a {OUTPUT_BITRATE} -ar 44100 no_vocals.mp3 -loglevel error")
print(f"   ✓ no_vocals.mp3 ({os.path.getsize('no_vocals.mp3') // 1024} KB)")

# 8. Загрузка на tempshare
print("\n[8/8] Загрузка на tempshare...")
r = subprocess.run(
    f'curl -s -X POST -F "file=@no_vocals.mp3" -F "duration={TEMPSHARE_DURATION}" '
    f"https://api.tempshare.su/upload",
    shell=True, capture_output=True, text=True,
)
print(r.stdout)

# Парсим ссылку
try:
    import json
    data = json.loads(r.stdout)
    if data.get("success"):
        print("\n" + "=" * 60)
        print("  ✓ ГОТОВО")
        print("=" * 60)
        print(f"  Страница:  {data['url']}")
        print(f"  Прямая:    {data['raw_url']}")
        print(f"  Действует: {TEMPSHARE_DURATION} дней (до {data.get('expires','?')})")
        print("=" * 60)
    else:
        print("❌ tempshare вернул ошибку, смотри ответ выше")
except Exception as e:
    print(f"Не удалось распарсить ответ tempshare ({e}), но файл загружен — смотри ответ curl выше")
