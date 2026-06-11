"""
AI Image Editor — управляет OpenCV через LM Studio (локальная LLM)
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import requests
import json
import threading


LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

SYSTEM_PROMPT = """Ты — помощник для редактирования изображений.
Пользователь пишет команду на русском языке, а ты должен вернуть ТОЛЬКО JSON с командой.
Никакого лишнего текста — только JSON!

Доступные команды:
- rotate: поворот изображения. Параметр angle: 90, 180, 270
- grayscale: сделать чёрно-белым. Нет параметров.
- blur: размытие. Параметр strength: целое нечётное число 3..51
- brightness: яркость. Параметр value: от -100 до 100
- resize: изменить размер. Параметры width и height (в пикселях)
- flip: отразить. Параметр direction: horizontal или vertical
- red_channel: оставить только красный канал. Нет параметров.
- green_channel: оставить только зелёный канал. Нет параметров.
- blue_channel: оставить только синий канал. Нет параметров.
- channel_select: выбрать каналы. Параметр channels: строка из букв R, G, B (например "RG" или "B")
- edge_detection: выделить края. Нет параметров.
- reset: сбросить изображение к исходному. Нет параметров.

Примеры ответов:
{"command": "rotate", "angle": 90}
{"command": "grayscale"}
{"command": "blur", "strength": 15}
{"command": "brightness", "value": 50}
{"command": "resize", "width": 640, "height": 480}
{"command": "flip", "direction": "horizontal"}
{"command": "red_channel"}
{"command": "green_channel"}
{"command": "blue_channel"}
{"command": "channel_select", "channels": "RG"}
{"command": "edge_detection"}
{"command": "reset"}

Если команда непонятна, верни: {"command": "unknown"}
"""

def ask_llm(user_text):
    """Отправляет текст в LM Studio и получает JSON-команду."""
    payload = {
        "model": "qwen2.5-coder-7b-instruct-spider-baseline",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text}
        ],
        "temperature": 0.0,   # 0 = детерминированный ответ, меньше выдумок
        "max_tokens": 100
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Убираем возможные ``` обёртки
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except requests.exceptions.ConnectionError:
        return {"command": "error", "message": "LM Studio не запущен! Запустите его на порту 1234."}
    except json.JSONDecodeError:
        return {"command": "error", "message": f"Модель вернула не JSON: {raw}"}
    except Exception as e:
        return {"command": "error", "message": str(e)}



def apply_command(image_bgr, cmd_json):
    """
    Принимает изображение (numpy array BGR) и JSON-команду.
    Возвращает (новое_изображение, сообщение_для_пользователя).
    """
    cmd = cmd_json.get("command", "unknown")

    if cmd == "rotate":
        angle = cmd_json.get("angle", 90)
        flags = {
            90:  cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        if angle not in flags:
            return image_bgr, f"Неверный угол: {angle}. Допустимы: 90, 180, 270."
        result = cv2.rotate(image_bgr, flags[angle])
        return result, f"✅ Повёрнуто на {angle}°"

    elif cmd == "grayscale":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return result, "✅ Чёрно-белое"

    elif cmd == "blur":
        strength = int(cmd_json.get("strength", 11))
        if strength % 2 == 0:
            strength += 1         
        strength = max(3, min(strength, 51))
        result = cv2.GaussianBlur(image_bgr, (strength, strength), 0)
        return result, f"✅ Размытие (сила={strength})"

    elif cmd == "brightness":
        value = int(cmd_json.get("value", 30))
        result = cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=value)
        return result, f"✅ Яркость изменена на {value:+d}"

    elif cmd == "resize":
        w = int(cmd_json.get("width",  640))
        h = int(cmd_json.get("height", 480))
        result = cv2.resize(image_bgr, (w, h))
        return result, f"✅ Размер изменён: {w}×{h}"

    elif cmd == "flip":
        direction = cmd_json.get("direction", "horizontal")
        flip_code = 1 if direction == "horizontal" else 0
        result = cv2.flip(image_bgr, flip_code)
        return result, f"✅ Отражено ({direction})"

    elif cmd == "red_channel":
        result = image_bgr.copy()
        result[:, :, 0] = 0   # синий = 0
        result[:, :, 1] = 0   # зелёный = 0
        # красный (индекс 2) остаётся
        return result, "✅ Только красный канал"

    elif cmd == "green_channel":
        result = image_bgr.copy()
        result[:, :, 0] = 0   # синий = 0
        result[:, :, 2] = 0   # красный = 0
        # зелёный (индекс 1) остаётся
        return result, "✅ Только зелёный канал"

    elif cmd == "blue_channel":
        result = image_bgr.copy()
        result[:, :, 1] = 0   # зелёный = 0
        result[:, :, 2] = 0   # красный = 0
        # синий (индекс 0) остаётся
        return result, "✅ Только синий канал"

    elif cmd == "channel_select":
        channels = cmd_json.get("channels", "RGB").upper()
        result = image_bgr.copy()
        # Обнуляем каналы, которые не выбраны
        if 'B' not in channels:
            result[:, :, 0] = 0  # синий
        if 'G' not in channels:
            result[:, :, 1] = 0  # зелёный
        if 'R' not in channels:
            result[:, :, 2] = 0  # красный
        return result, f"✅ Выбраны каналы: {channels}"

    elif cmd == "edge_detection":
        gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(gray, threshold1=50, threshold2=150)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return result, "✅ Выделены края (Canny)"

    elif cmd == "reset":
        return None, "✅ Сброс к оригиналу"   

    elif cmd == "error":
        return image_bgr, f"❌ Ошибка: {cmd_json.get('message', '')}"

    else:
        return image_bgr, "❓ Команда не распознана. Попробуйте иначе."



class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Editor  •  LM Studio + OpenCV")
        self.root.configure(bg="#1e1e2e")

        self.original_bgr = None   # исходное изображение
        self.current_bgr  = None   # текущее (после команд)

        self._build_ui()

    # ── UI ────────────────────────────────────
    def _build_ui(self):
        top = tk.Frame(self.root, bg="#1e1e2e", pady=10)
        top.pack(fill="x", padx=20)

        tk.Button(
            top, text="📂 Открыть изображение",
            command=self.load_image,
            bg="#7c3aed", fg="white", font=("Courier", 12, "bold"),
            relief="flat", padx=15, pady=6, cursor="hand2"
        ).pack(side="left")

        self.size_label = tk.Label(
            top, text="", bg="#1e1e2e", fg="#a1a1aa",
            font=("Courier", 10)
        )
        self.size_label.pack(side="left", padx=20)

        
        self.canvas = tk.Label(
            self.root, bg="#2a2a3e",
            text="← Откройте изображение",
            fg="#6b7280", font=("Courier", 14)
        )
        self.canvas.pack(padx=20, pady=10, fill="both", expand=True)

        
        bottom = tk.Frame(self.root, bg="#1e1e2e", pady=10)
        bottom.pack(fill="x", padx=20)

        self.entry = tk.Entry(
            bottom, font=("Courier", 13),
            bg="#2a2a3e", fg="white", insertbackground="white",
            relief="flat"
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.run_command())
        self.entry.insert(0, "Поверни на 90 градусов...")

        self.btn_run = tk.Button(
            bottom, text="▶ Выполнить",
            command=self.run_command,
            bg="#10b981", fg="white", font=("Courier", 12, "bold"),
            relief="flat", padx=15, pady=6, cursor="hand2"
        )
        self.btn_run.pack(side="left")

        self.status = tk.Label(
            self.root, text="Готов к работе.",
            bg="#111827", fg="#6ee7b7",
            font=("Courier", 10), anchor="w", padx=10
        )
        self.status.pack(fill="x", side="bottom")

    
        hints = (
            "Примеры команд:  «Сделай чёрно-белым»  •  «Размой»  •  «Выдели края»  •  "
            "«Увеличь яркость»  •  «Красный канал»  •  «Зелёный канал»  •  «Синий канал»  •  "
            "«Выбери красный и зелёный каналы»  •  «Сброс»"
        )
        tk.Label(
            self.root, text=hints,
            bg="#1e1e2e", fg="#4b5563",
            font=("Courier", 9), wraplength=800, justify="left"
        ).pack(fill="x", padx=20, pady=(0, 4))

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Ошибка", "Не удалось открыть файл.")
            return
        self.original_bgr = img.copy()
        self.current_bgr  = img.copy()
        self._show(self.current_bgr)
        self.set_status(f"Загружено: {path.split('/')[-1]}")

    def run_command(self):
        if self.current_bgr is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте изображение.")
            return
        text = self.entry.get().strip()
        if not text:
            return

        self.btn_run.config(state="disabled", text="⏳ Думаю...")
        self.set_status("Отправляю запрос в LM Studio...")

        threading.Thread(target=self._process, args=(text,), daemon=True).start()

    def _process(self, text):
        cmd_json = ask_llm(text)

        new_img, msg = apply_command(self.current_bgr, cmd_json)

        if new_img is None:
            new_img = self.original_bgr.copy()

        self.current_bgr = new_img

        self.root.after(0, lambda: self._after_process(msg))

    def _after_process(self, msg):
        self._show(self.current_bgr)
        self.set_status(msg)
        self.btn_run.config(state="normal", text="▶ Выполнить")

    def _show(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Подгоняем под размер окна (макс 780×520)
        pil.thumbnail((780, 520), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil)
        self.canvas.config(image=tk_img, text="")
        self.canvas.image = tk_img  # держим ссылку!

        h, w = bgr.shape[:2]
        self.size_label.config(text=f"{w} × {h} px")

    def set_status(self, msg):
        self.status.config(text=f"  {msg}")



if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("860x640")
    app = App(root)
    root.mainloop()
