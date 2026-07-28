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
import os, sys
from urllib.parse import urlparse

if len(sys.argv) != 2:
    sys.exit("Запусти скрипт из Colab-ячейки через launcher: ссылка передаётся вторым аргументом.")
TRACK_URL = sys.argv[1].strip()
parsed_url = urlparse(TRACK_URL)
if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
    sys.exit("❌ Нужна полноценная ссылка, начинающаяся с http:// или https://")

import subprocess, glob, zipfile, re, shutil, shlex
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

# Если это ZIP — распаковываем все аудиофайлы, иначе обрабатываем один трек
if zipfile.is_zipfile("input_track"):
    shutil.rmtree("input_extracted", ignore_errors=True)
    with zipfile.ZipFile("input_track") as z:
        z.extractall("input_extracted")
    audio_files = sorted(
        m for m in glob.glob("input_extracted/**/*", recursive=True)
        if os.path.isfile(m) and m.lower().endswith((".mp3", ".wav", ".flac", ".m4a"))
    )
else:
    audio_files = ["input_track"]
assert audio_files, "В архиве нет аудиофайлов"
print(f"   Найдено треков: {len(audio_files)}")

# 3. Конвертация всех треков в WAV 44.1kHz стерео
print("\n[3/8] Конвертация всех треков в WAV...")
shutil.rmtree("input_wav", ignore_errors=True)
os.makedirs("input_wav", exist_ok=True)
input_wavs = []
used_names = set()
for index, source in enumerate(audio_files, 1):
    original_name = os.path.splitext(os.path.basename(source))[0]
    safe_name = re.sub(r"[^A-Za-z0-9А-Яа-я._-]+", "_", original_name).strip("._") or f"track_{index}"
    candidate = safe_name
    suffix = 2
    while candidate.lower() in used_names:
        candidate = f"{safe_name}_{suffix}"
        suffix += 1
    used_names.add(candidate.lower())
    target = os.path.join("input_wav", f"{candidate}.wav")
    print(f"   [{index}/{len(audio_files)}] {os.path.basename(source)}")
    run(
        f"ffmpeg -y -i {shlex.quote(source)} -ar 44100 -ac 2 "
        f"{shlex.quote(target)} -loglevel error"
    )
    input_wavs.append(target)

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
print("\n[6/8] Разделение всех треков (Big Beta 6 на GPU)...")
shutil.rmtree("out", ignore_errors=True)
run(
    f"melband-roformer-infer "
    f"--config_path '{cfg}' --model_path '{weights}' "
    f"--input_folder input_wav --store_dir out --device cuda:0"
)

# 7. Конвертация всех инструменталов в MP3 320 kbps и упаковка в ZIP
print("\n[7/8] Конвертация всех инструменталов в MP3 320 kbps...")
inst = sorted(glob.glob("out/*_instrumental.wav"))
assert inst, "❌ instrumental.wav не найден в выводе"
if len(inst) != len(input_wavs):
    print(f"⚠️ Модель вернула {len(inst)} инструменталов из {len(input_wavs)} треков")
shutil.rmtree("no_vocals", ignore_errors=True)
os.makedirs("no_vocals", exist_ok=True)
mp3_outputs = []
for index, wav_file in enumerate(inst, 1):
    stem = os.path.basename(wav_file)
    if stem.lower().endswith("_instrumental.wav"):
        stem = stem[:-len("_instrumental.wav")]
    output_file = os.path.join("no_vocals", f"{stem}_instrumental.mp3")
    run(
        f"ffmpeg -y -i {shlex.quote(wav_file)} -c:a libmp3lame "
        f"-b:a {OUTPUT_BITRATE} -ar 44100 {shlex.quote(output_file)} -loglevel error"
    )
    mp3_outputs.append(output_file)
    print(f"   ✓ [{index}/{len(inst)}] {os.path.basename(output_file)} ({os.path.getsize(output_file) // 1024} KB)")

archive_name = "no_vocals.zip"
with zipfile.ZipFile(archive_name, "w", zipfile.ZIP_DEFLATED) as archive:
    for output_file in mp3_outputs:
        archive.write(output_file, os.path.basename(output_file))
print(f"   ✓ Архив создан: {archive_name} ({os.path.getsize(archive_name) // 1024} KB)")

# 8. Загрузка архива на tempshare
print("\n[8/8] Загрузка архива на tempshare...")
r = subprocess.run(
    f'curl -s -X POST -F "file=@{shlex.quote(archive_name)}" -F "duration={TEMPSHARE_DURATION}" '
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
