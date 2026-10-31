import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # фикс нумба

import math
import time
import tkinter as tk
import librosa
import numpy as np
from tkinter import filedialog
import pygame
from scipy.signal import find_peaks, medfilt
import threading
import tempfile
import requests
import yt_dlp

#звук
pygame.mixer.init()

#общ. переменные
selected_audio_path = ""
tracks_history = []  #оцененные треки

text_top = "RRH " * 80
text_right = "RRH " * 80
text_bottom = "RRH " * 80
text_left = "RRH " * 80

#онлайн поиск (YouTube)
YOUTUBE_API_KEY = "AIzaSyBwJ-bD9w9APR6On9tiTK-7GIa2lyvZ7iA"
YT_SEARCH = "https://www.googleapis.com/youtube/v3/search"
FFMPEG_DIR = r"C:\ffmpeg\bin"  # папка с ffmpeg.exe
tmp_dir = tempfile.mkdtemp(prefix="rhythm_")
PLACEHOLDER = "🔎 Название песни..."
online_results = []

#цвета
BG = "#0a0a18"
PANEL = "#12122b"
PANEL2 = "#1a1a3d"
CYAN = "#00e5ff"
PINK = "#ff2e88"
PURPLE = "#9b51e0"
TEXT = "#e6e6ff"
MUTED = "#7a7aa0"
GOLD = "#ffd166"
RED = "#ff3b5c"
GREEN = "#2cff9a"

#анимки
SPEC_BANDS = 48
SPEC_W = 780
SPEC_H = 90
BAR_W = 380
spec_frames = None
spec_sr = 44100
spec_hop = 1024
spec_levels = np.zeros(SPEC_BANDS)
spec_peaks = np.zeros(SPEC_BANDS)
gauge_cur = 0.0
gauge_tar = 0.0
gauge_active = False
bar_cur = [0.0, 0.0, 0.0]
bar_tar = [0.0, 0.0, 0.0]
paused = False

#цветовые переходы
def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

#цвет результата
def score_color(v):
    if v < 5:
        return lerp_color(RED, GOLD, v / 5.0)
    return lerp_color(GOLD, GREEN, (v - 5.0) / 5.0)

#изменение кнопки
def add_hover(widget, normal, hover):
    widget.config(activebackground=hover)
    widget.bind("<Enter>", lambda e: widget.config(bg=hover))
    widget.bind("<Leave>", lambda e: widget.config(bg=normal))

#передача оценок в индикаторы
def set_scores(b_score, c_score, g_score, total):
    global gauge_tar, gauge_active
    gauge_active = True
    gauge_tar = float(total)
    bar_tar[0] = float(b_score)
    bar_tar[1] = float(c_score)
    bar_tar[2] = float(g_score)

#визуализатор
def compute_spectrum(y, sr):
    n_fft = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=spec_hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    edges = np.geomspace(40, min(sr / 2 - 1, 12000), SPEC_BANDS + 1)
    out = np.zeros((S.shape[1], SPEC_BANDS))
    for b in range(SPEC_BANDS):
        mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if mask.any():
            out[:, b] = S[mask].mean(axis=0)
    out = np.sqrt(out)
    out = out / (np.percentile(out, 95, axis=0) + 1e-9)
    return np.clip(out, 0.0, 1.0)

#ррх контур
def animate_border():
    global text_top, text_right, text_bottom, text_left
#движение контура
    text_top = text_top[1:] + text_top[0]
    text_right = text_right[1:] + text_right[0]
    text_bottom = text_bottom[-1] + text_bottom[:-1]
    text_left = text_left[-1] + text_left[:-1]
#вывод контура
    lbl_top.config(text=text_top[:140])
    lbl_right.config(text="\n".join(text_right[:50]))
    lbl_bottom.config(text=text_bottom[:140])
    lbl_left.config(text="\n".join(text_left[:50]))
#зацикливание через 120мс
    root.after(120, animate_border)

#плавность
def ui_tick():
    global gauge_cur, spec_levels, spec_peaks
    now = time.time()
#переливание контура
    t = (math.sin(now * 2.0) + 1) / 2
    lbl_title.config(fg=lerp_color(CYAN, PINK, t))
    lbl_top.config(fg=lerp_color("#2a2fd0", "#00d9ff", t))
    lbl_bottom.config(fg=lerp_color("#2a2fd0", "#00d9ff", 1 - t))
    lbl_left.config(fg=lerp_color("#ff2e88", "#7a2cff", t))
    lbl_right.config(fg=lerp_color("#ff2e88", "#7a2cff", 1 - t))
#круговой индикатор
    d = gauge_tar - gauge_cur
    gauge_cur += d * 0.10 if abs(d) > 0.02 else d
    frac = max(0.0, min(1.0, gauge_cur / 10.0))
    col = score_color(gauge_cur)
    if frac < 0.004:
        gauge_canvas.itemconfig(gauge_arc, state="hidden")
    else:
        gauge_canvas.itemconfig(gauge_arc, state="normal", extent=-359.9 * frac, outline=col)
    gauge_canvas.itemconfig(gauge_text, text=f"{gauge_cur:.1f}" if gauge_active else "--",
                            fill=col if gauge_active else MUTED)
#полоски критериев
    for i in range(3):
        dd = bar_tar[i] - bar_cur[i]
        bar_cur[i] += dd * 0.12 if abs(dd) > 0.02 else dd
        w = max(0.0, min(1.0, bar_cur[i] / 10.0)) * (BAR_W - 16)
        if w < 1:
            bar_canvases[i].itemconfig(bar_fills[i], state="hidden")
        else:
            bar_canvases[i].coords(bar_fills[i], 8, 8, 8 + w, 8)
            bar_canvases[i].itemconfig(bar_fills[i], state="normal", fill=score_color(bar_cur[i]))
#запуск трека
    playing = False
    try:
        playing = pygame.mixer.music.get_busy()
    except Exception:
        pass

    if playing:
        target = None
        if spec_frames is not None:
            try:
                pos = pygame.mixer.music.get_pos() / 1000.0
                idx = int(pos * spec_sr / spec_hop)
                if 0 <= idx < len(spec_frames):
                    target = spec_frames[idx]
                else:
                    target = np.zeros(SPEC_BANDS)
            except Exception:
                target = None
        if target is None:
            target = 0.15 + 0.6 * np.random.rand(SPEC_BANDS)
    else:
        target = np.array([0.05 + 0.04 * math.sin(now * 2.0 + i * 0.4) for i in range(SPEC_BANDS)])
    rising = target > spec_levels
    spec_levels = spec_levels + (target - spec_levels) * np.where(rising, 0.55, 0.15)
    spec_peaks = np.maximum(spec_levels, spec_peaks - 0.012)
    slot = SPEC_W / SPEC_BANDS

    for i in range(SPEC_BANDS):
        x0 = i * slot + 2
        x1 = (i + 1) * slot - 2
        h = max(2.0, float(spec_levels[i]) * (SPEC_H - 8))
        spec_canvas.coords(spec_bars[i], x0, SPEC_H - h, x1, SPEC_H)
        ph = max(4.0, float(spec_peaks[i]) * (SPEC_H - 8)) + 3
        spec_canvas.coords(spec_caps[i], x0, SPEC_H - ph - 2, x1, SPEC_H - ph)
    root.after(33, ui_tick)

#рейтинг оцененных песен
def update_rating_display():
#сортировка по рейтингу
    sorted_tracks = sorted(tracks_history, key=lambda x: x["score"], reverse=True)
#названия топ 5
    rating_text = "🏆 РЕЙТИНГ ТРЕКОВ:\n" + "-" * 25 + "\n"
    if not sorted_tracks:
        rating_text += "История пуста...\nПроанализируйте трек"
    else:
        medals = ["🥇", "🥈", "🥉"]
        for idx, track in enumerate(sorted_tracks[:5], 1):
            short_name = track['name'] if len(track['name']) <= 18 else track['name'][:15] + "..."
            prefix = medals[idx - 1] if idx <= 3 else f"{idx}."
            rating_text += f"{prefix} {short_name} — {track['score']}/10\n"
    lbl_rating.config(text=rating_text)

#пауза / продолжить
def toggle_pause():
    global paused
    try:
        if paused:
            pygame.mixer.music.unpause()
            btn_pause.config(text="⏸")
            paused = False
        else:
            pygame.mixer.music.pause()
            btn_pause.config(text="▶")
            paused = True
    except Exception as e:
        print(f"❌ Ошибка паузы: {e}")

#анализ трека
def analyze_file(file_path):
    global selected_audio_path, spec_frames, spec_sr, paused
    selected_audio_path = file_path
    filename = os.path.basename(file_path)
    lbl_status.config(text=f"⏳ Анализируется:\n{filename}", fg="#2f80ed")
    set_scores(0, 0, 0, 0)
    root.update()

    try:
        evaluator = TrackRhythmEvaluator(selected_audio_path)
        total_score = evaluator.print_full_report()

        try:
            spec_frames = compute_spectrum(evaluator.y, evaluator.sr)
            spec_sr = evaluator.sr
        except Exception as e:
            spec_frames = None
            print(f"⚠️ Спектр не посчитан, включена запасная анимация: {e}")

        if not any(t['name'] == filename for t in tracks_history):
            tracks_history.append({"name": filename, "score": total_score})
            update_rating_display()
        lbl_status.config(text=f"▶️ Сейчас играет: {filename}", fg="#27ae60")
    except Exception as e:
        print(f"❌ Ошибка анализа файла: {e}")
        lbl_status.config(text="❌ Ошибка анализа файла", fg="#eb5757")
        return

    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        paused = False
        btn_pause.config(text="⏸")
    except Exception as e:
        print(f"❌ Ошибка воспроизведения pygame: {e}")

#выбор локального файла
def open_and_analyze():
    file_path = filedialog.askopenfilename(
        title="Выберите аудиофайл",
        filetypes=[("Аудио файлы", "*.mp3 *.wav *.ogg")]
    )
    if file_path:
        analyze_file(file_path)

#короткое понятное описание ошибки (без URL и ключа)
def short_error(e):
    if isinstance(e, requests.HTTPError) and e.response is not None:
        try:
            msg = e.response.json()["error"]["message"]
        except Exception:
            msg = f"HTTP {e.response.status_code}"
        return msg[:120]
    if isinstance(e, requests.ConnectionError):
        return "нет соединения с интернетом"
    return str(e)[:120]

#онлайн поиск (YouTube)
def search_online(event=None):
    query = search_var.get().strip()
    if query == PLACEHOLDER or not query:
        query = "популярные песни"

    if YOUTUBE_API_KEY.startswith("ВСТАВЬ"):
        lbl_status.config(text="❌ Вставь YOUTUBE_API_KEY в код", fg="#eb5757")
        return

    lbl_status.config(text="🔎 Поиск...", fg="#2f80ed")

    def worker():
        try:
            params = {
                "part": "snippet",
                "type": "video",
                "videoCategoryId": "10",
                "maxResults": 25,
                "q": query,
                "key": YOUTUBE_API_KEY,
            }
            r = requests.get(YT_SEARCH, params=params, timeout=15)
            r.raise_for_status()
            data = []
            for it in r.json().get("items", []):
                vid = it.get("id", {}).get("videoId")
                if not vid:
                    continue
                sn = it["snippet"]
                data.append({
                    "id": vid,
                    "name": sn["title"],
                    "artist_name": sn["channelTitle"],
                })
            root.after(0, lambda: show_search_results(data))
        except Exception as e:
            err = short_error(e)
            root.after(0, lambda: lbl_status.config(text=f"❌ Поиск: {err}", fg="#eb5757"))

    threading.Thread(target=worker, daemon=True).start()

#результаты поиска
def show_search_results(data):
    online_results.clear()
    online_results.extend(data)
    lst_results.delete(0, "end")

    if not data:
        lbl_status.config(text="Ничего не найдено", fg="#828282")
        hide_results()
        return
    for t in data:
        lst_results.insert("end", f"{t['artist_name']} — {t['name']}")
    lbl_status.config(text=f"Найдено: {len(data)}", fg="#828282")

    root.update_idletasks()
    x_right = search_entry.winfo_rootx() - root.winfo_rootx() + search_entry.winfo_width()
    y = search_entry.winfo_rooty() - root.winfo_rooty() + search_entry.winfo_height() + 2
    lst_results.place(x=x_right, y=y, anchor="ne")
    lst_results.lift()

#скрыть список
def hide_results(event=None):
    lst_results.place_forget()

#скачивание трека (звук из YouTube, через yt-dlp и ffmpeg)
def download_selected(event=None):
    sel = lst_results.curselection()
    if not sel:
        return
    track = online_results[sel[0]]
    hide_results()
    lbl_status.config(text="⬇️ Скачивание...", fg="#2f80ed")

    def worker():
        try:
            safe = "".join(c for c in track["name"] if c.isalnum() or c in " -_")[:60] or "track"
            base = os.path.join(tmp_dir, f"{track['id']}_{safe}")
            opts = {
                "format": "bestaudio/best",
                "outtmpl": base + ".%(ext)s",
                "ffmpeg_location": FFMPEG_DIR,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
                "quiet": True,
                "noplaylist": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={track['id']}"])
            path = base + ".mp3"
            root.after(0, lambda: analyze_file(path))
        except Exception as e:
            err = str(e)[:120]
            root.after(0, lambda: lbl_status.config(text=f"❌ Загрузка: {err}", fg="#eb5757"))
    threading.Thread(target=worker, daemon=True).start()

def on_search_focus_in(event):
    if search_var.get() == PLACEHOLDER:
        search_entry.delete(0, "end")
        search_entry.config(fg="white")

def on_search_focus_out(event):
    if not search_var.get().strip():
        search_entry.insert(0, PLACEHOLDER)
        search_entry.config(fg=MUTED)

class TrackRhythmEvaluator:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path, sr=None)

#дельта по бпм
    def evaluate_bpm_spread(self):
        o_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        local_bpms = librosa.feature.tempo(onset_envelope=o_env, sr=self.sr, aggregate=None)
        local_bpms_smoothed = medfilt(local_bpms, kernel_size=31) if len(local_bpms) > 31 else local_bpms
        q75, q25 = np.percentile(local_bpms_smoothed, [75, 25])
        bpm_spread = q75 - q25

        if bpm_spread < 2.0:
            bpm_spread = 0.0
        if bpm_spread <= 2.0:
            score = 10.0
            verdict = "Стабильный темп (электронная сетка / игра под клик)"
        elif bpm_spread <= 8.0:
            score = np.clip(10.0 - (bpm_spread - 2.0) * 0.5, 7.0, 9.5)
            verdict = "Естественный «живой» темп (микро-сдвиги группы)"
        else:
            score = np.clip(7.0 - (bpm_spread - 8.0) * 0.2, 1.0, 7.0)
            verdict = "Динамичный темп (осознанная смена частей трека)"
        return round(bpm_spread, 1), round(score, 1), verdict

#контрастность песни
    def evaluate_structural_contrast(self):
        rms = librosa.feature.rms(y=self.y)[0]
        hop_len_rms = 512
        frames_per_5sec = int((self.sr * 5) / hop_len_rms)
        chunks = [rms[i: i + frames_per_5sec] for i in range(0, len(rms), frames_per_5sec)]
        chunk_loudness = [np.mean(chunk) for chunk in chunks if len(chunk) > 0]

        if len(chunk_loudness) > 0:
            loud_peak = np.max(chunk_loudness)
            quiet_drop = np.percentile(chunk_loudness, 10) + 1e-4
            contrast = loud_peak / quiet_drop
        else:
            contrast = 1.0
        score = np.clip(contrast * 3.0, 1.0, 10.0)

        if contrast > 2.2:
            verdict = "Яркие перепады энергии"
        else:
            verdict = "Ровная, монотонная динамика"
        return round(contrast, 2), round(score, 1), verdict

#так называемый грув
    def evaluate_groove(self):
        o_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        beat_frames, _ = find_peaks(o_env, distance=20, prominence=0.2)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        if len(beat_times) > 2:
            intervals = np.diff(beat_times)
            timing_deviation = np.std(intervals)
        else:
            timing_deviation = 0.0
        score = np.clip(timing_deviation * 150, 1.0, 10.0)

        if timing_deviation > 0.04:
            verdict = "«Живой» кач/свинг (микро-сдвиги относительно сетки)"
        else:
            verdict = "Роботизированный квантованный бит (идеальная сетка)"
        return round(timing_deviation, 4), round(score, 1), verdict

#вывод результатов оценки
    def print_full_report(self):
        b_spread, b_score, b_verd = self.evaluate_bpm_spread()
        c_val, c_score, c_verd = self.evaluate_structural_contrast()
        g_val, g_score, g_verd = self.evaluate_groove()

# защита от сплошного шума
        is_probably_noise = (c_val < 1.6) and (g_val > 0.06)
        if is_probably_noise:
            g_score = np.clip(g_score - 6.0, 1.0, 10.0)
            b_score = np.clip(b_score - 5.0, 1.0, 10.0)
            b_verd = "Нестабильная частотная пульсация шума"
            g_verd = "Хаотичные всплески частот (генератор шума)"
        total_score = round((b_score + c_score + g_score) / 3, 1)

        lbl_metric1.config(text=f"1. Вариативность темпа: {b_spread} BPM  |  Оценка: {b_score}/10\n👉 {b_verd}")
        lbl_metric2.config(text=f"2. Структурный контраст: {c_val} коэф.  |  Оценка: {c_score}/10\n👉 {c_verd}")
        lbl_metric3.config(text=f"3. Грув: {g_val}   |  Оценка: {g_score}/10\n👉 {g_verd}")
        lbl_total.config(text=f"🌟 ОБЩИЙ БАЛЛ: {total_score} / 10")

        if is_probably_noise:
            lbl_total.config(text=f"🌟 ОБЩИЙ БАЛЛ: {total_score} / 10\n⚠️ Обнаружен монотонный шум!")
        set_scores(b_score, c_score, g_score, total_score)
        return total_score


#само окно
root = tk.Tk()
root.title("Анализатор Ритмики")
root.geometry("960x500")
root.resizable(False, False)
root.configure(bg=BG)

#настройки цветов
font_cfg = ("Courier New", 9, "bold")
bg_color = BG
fg_color1 = "#2a2fd0"
fg_color2 = "#ff2e88"

#рамка контура
lbl_top = tk.Label(root, text="", font=font_cfg, bg=bg_color, fg=fg_color1, anchor="w")
lbl_top.pack(side="top", fill="x", padx=5)

lbl_bottom = tk.Label(root, text="", font=font_cfg, bg=bg_color, fg=fg_color1, anchor="e")
lbl_bottom.pack(side="bottom", fill="x", padx=5)

lbl_left = tk.Label(root, text="", font=font_cfg, bg=bg_color, fg=fg_color2, justify="center")
lbl_left.pack(side="left", fill="y", pady=0, padx=(5, 0))

lbl_right = tk.Label(root, text="", font=font_cfg, bg=bg_color, fg=fg_color2, justify="center")
lbl_right.pack(side="right", fill="y", pady=0, padx=(0, 5))

content_frame = tk.Frame(root, bg=BG)
content_frame.pack(expand=True, fill="both", padx=15, pady=10)

lbl_title = tk.Label(content_frame, text="⚡ АНАЛИЗАТОР РИТМИКИ ⚡", font=("Arial", 16, "bold"), bg=BG, fg=CYAN)
lbl_title.pack(pady=(0, 8))

top_bar = tk.Frame(content_frame, bg=BG)
top_bar.pack(fill="x", pady=(0, 10))

#поисковая строка
search_frame = tk.Frame(top_bar, bg=PANEL2, highlightbackground=PINK, highlightthickness=1)
search_frame.pack(side="right", padx=(10, 0))
search_var = tk.StringVar(value=PLACEHOLDER)
search_entry = tk.Entry(
    search_frame, textvariable=search_var, width=22, font=("Arial", 10),
    bg=PANEL2, fg=MUTED, insertbackground="white", bd=0
)
search_entry.pack(side="left", ipady=6, padx=(8, 0))
btn_search = tk.Button(
    search_frame, text="Найти", command=search_online, bg=PINK, fg="white",
    bd=0, padx=10, cursor="hand2", font=("Arial", 9, "bold"), activeforeground="white"
)
btn_search.pack(side="left", fill="y", padx=(6, 0))
add_hover(btn_search, PINK, "#ff6aa8")
search_entry.bind("<FocusIn>", on_search_focus_in)
search_entry.bind("<FocusOut>", on_search_focus_out)
search_entry.bind("<Return>", search_online)
search_entry.bind("<Escape>", hide_results)

#список результатов
lst_results = tk.Listbox(
    root, width=38, height=8, bg=PANEL2, fg=TEXT, font=("Arial", 9),
    selectbackground=PINK, bd=0, highlightthickness=1,
    highlightbackground=CYAN, activestyle="none"
)
lst_results.bind("<Double-Button-1>", download_selected)
lst_results.bind("<Return>", download_selected)

btn_open = tk.Button(
    top_bar, text="📁 Открыть и проанализировать аудио", command=open_and_analyze,
    font=("Arial", 10, "bold"), bg=CYAN, fg="#0a0a18", activebackground="#5cf0ff",
    activeforeground="#0a0a18", bd=0, padx=15, pady=8, cursor="hand2"
)
btn_open.pack(side="left", padx=10)
add_hover(btn_open, CYAN, "#5cf0ff")

lbl_status = tk.Label(top_bar, text="Файл не выбран", font=("Arial", 9, "italic"), bg=BG, fg=MUTED,
                      justify="left", wraplength=260)
lbl_status.pack(side="left", padx=10)

#визуализатор и пауза
spectrum_row = tk.Frame(content_frame, bg=BG)
spectrum_row.pack(side="bottom", fill="x", pady=(10, 0))

btn_pause = tk.Button(
    spectrum_row, text="⏸", command=toggle_pause, font=("Arial", 14, "bold"),
    bg=PURPLE, fg="white", activeforeground="white", bd=0, width=3, cursor="hand2"
)
btn_pause.pack(side="left", fill="y", padx=(0, 10))
add_hover(btn_pause, PURPLE, "#b57cf0")

spec_canvas = tk.Canvas(spectrum_row, width=SPEC_W, height=SPEC_H, bg=PANEL, highlightthickness=1,
                        highlightbackground=PURPLE)
spec_canvas.pack(side="left", fill="x", expand=True)
spec_bars = []
spec_caps = []
for _i in range(SPEC_BANDS):
    _color = lerp_color(CYAN, PINK, _i / (SPEC_BANDS - 1))
    spec_bars.append(spec_canvas.create_rectangle(0, SPEC_H, 0, SPEC_H, fill=_color, outline=""))
    spec_caps.append(spec_canvas.create_rectangle(0, SPEC_H, 0, SPEC_H, fill="#ffffff", outline=""))

#колонки
columns_frame = tk.Frame(content_frame, bg=BG)
columns_frame.pack(expand=True, fill="both")
left_column = tk.Frame(columns_frame, bg=BG)
left_column.pack(side="left", expand=True, fill="both", padx=(0, 10))

report_row = tk.Frame(left_column, bg=BG)
report_row.pack(fill="both", expand=True)

frame_report = tk.Frame(report_row, bg=PANEL, padx=15, pady=12, highlightbackground=CYAN, highlightthickness=1)
frame_report.pack(side="left", fill="both", expand=True)

bar_canvases = []
bar_fills = []

#заголовки критериев
def make_metric(text):
    lbl = tk.Label(frame_report, text=text, font=("Arial", 9), bg=PANEL, fg=TEXT,
                   justify="left", wraplength=BAR_W)
    lbl.pack(anchor="w", pady=(4, 2))
    cv = tk.Canvas(frame_report, width=BAR_W, height=16, bg=PANEL, highlightthickness=0)
    cv.create_line(8, 8, BAR_W - 8, 8, width=10, fill="#262650", capstyle="round")
    fill = cv.create_line(8, 8, 8, 8, width=10, fill=GREEN, capstyle="round", state="hidden")
    cv.pack(anchor="w", pady=(0, 6))
    bar_canvases.append(cv)
    bar_fills.append(fill)
    return lbl

lbl_metric1 = make_metric("1. Вариативность темпа: ---\n👉 ожидание файла...")
lbl_metric2 = make_metric("2. Структурный контраст: ---\n👉 ожидание файла...")
lbl_metric3 = make_metric("3. Отклонение тайминга: ---\n👉 ожидание файла...")

#финальный балл
gauge_frame = tk.Frame(report_row, bg=PANEL, padx=10, pady=8, highlightbackground=PINK, highlightthickness=1)
gauge_frame.pack(side="left", fill="y", padx=(10, 0))
gauge_canvas = tk.Canvas(gauge_frame, width=170, height=170, bg=PANEL, highlightthickness=0)
gauge_canvas.create_oval(15, 15, 155, 155, outline="#262650", width=12)
gauge_arc = gauge_canvas.create_arc(15, 15, 155, 155, start=90, extent=0, style="arc",
                                    outline=GREEN, width=12, state="hidden")
gauge_text = gauge_canvas.create_text(85, 78, text="--", font=("Arial", 30, "bold"), fill=MUTED)
gauge_canvas.create_text(85, 112, text="из 10", font=("Arial", 10), fill=MUTED)
gauge_canvas.pack()
lbl_total = tk.Label(gauge_frame, text="🌟 ОБЩИЙ БАЛЛ: --", font=("Arial", 10, "bold"), bg=PANEL, fg=GOLD,
                     wraplength=170, justify="center")
lbl_total.pack(pady=(6, 0))

right_column = tk.Frame(columns_frame, bg=BG, width=250)
right_column.pack(side="right", fill="both", padx=(10, 0))
right_column.pack_propagate(False)
frame_rating = tk.Frame(right_column, bg=PANEL, padx=15, pady=15, bd=1, relief="solid",
                        highlightbackground=GOLD, highlightthickness=1)
frame_rating.pack(fill="both", expand=True)
lbl_rating = tk.Label(
    frame_rating, text="🏆 РЕЙТИНГ ТРЕКОВ:\n" + "-" * 25 + "\nИстория пуста...\nПроанализируйте трек",
    font=("Courier New", 9, "bold"), bg=PANEL, fg=GOLD, justify="left", anchor="nw"
)
lbl_rating.pack(fill="both", expand=True)

root.after(120, animate_border)
root.after(33, ui_tick)
root.mainloop()