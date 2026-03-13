import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import librosa
import numpy as np
import threading
import pygame
import subprocess
import os
import cairosvg
import io
import shutil
import sys
import glob
import random


def get_ffmpeg_path():
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except:
        pass
    path = shutil.which("ffmpeg")
    return path if path else "ffmpeg"


FFMPEG = get_ffmpeg_path()


class ITGMaker:
    def __init__(self, root):
        self.root = root
        root.title("ITGmania Generator - SYNC FIXED")
        root.geometry("1280x720")

        self.video_path = None
        self.preview_running = False
        self.beat_times = []
        self.tempo = 120
        self.spb = 0.5
        self.offset = 0.0
        self.pixels_per_beat = 200

        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir, ignore_errors=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.show_upload_screen()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def show_upload_screen(self):
        for w in self.root.winfo_children():
            w.destroy()
        frame = tk.Frame(self.root)
        frame.pack(expand=True)
        tk.Label(frame, text="ITGmania Chart Generator", font=("Arial", 24)).pack(
            pady=20
        )
        tk.Button(
            frame,
            text="Upload Video",
            font=("Arial", 16),
            width=20,
            command=self.select_video,
        ).pack(pady=10)

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv *.avi")])
        if not path:
            return
        self.video_path = path
        threading.Thread(target=self.analyze_audio, daemon=True).start()

    def analyze_audio(self):
        audio_file = os.path.join(self.data_dir, "temp_analysis.wav")
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-i",
                self.video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "1",
                audio_file,
            ],
            capture_output=True,
        )

        y, sr = librosa.load(audio_file, sr=44100)
        self.duration = y.shape[0] / sr
        _, y_percussive = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        self.tempo = float(np.atleast_1d(tempo)[0])
        self.spb = 60.0 / self.tempo
        self.offset = librosa.frames_to_time(beats[0], sr=sr) if len(beats) > 0 else 0.0

        rms = librosa.feature.rms(y=y)[0]
        threshold = np.max(rms) * 0.10

        final_grid_beats = []
        num_beats = int((self.duration - self.offset) / self.spb)
        for i in range(num_beats + 1):
            beat_time = self.offset + i * self.spb
            f_idx = librosa.time_to_frames(beat_time, sr=sr)
            win = librosa.time_to_frames(0.1, sr=sr)
            if (
                np.max(rms[max(0, f_idx - win) : min(len(rms), f_idx + win)])
                > threshold
            ):
                final_grid_beats.append(i * 1.0)

        self.beat_times = sorted(final_grid_beats)
        self.generate_chart_file()
        self.parse_sm(os.path.join(self.data_dir, "chart.sm"))
        self.root.after(0, self.show_preview_screen)

    def create_svg_arrows(self):
        arrows = {}
        colors = {"4th": "#ff2424", "8th": "#2466ff", "rec": "#1a1a1a"}
        paths = [
            "M10,50 L55,10 L55,35 L90,35 L90,65 L55,65 L55,90 Z",
            "M50,90 L10,45 L35,45 L35,10 L65,10 L65,45 L90,45 Z",
            "M50,10 L90,55 L65,55 L65,90 L35,90 L35,55 L10,55 Z",
            "M90,50 L45,10 L45,35 L10,35 L10,65 L45,65 L45,90 Z",
        ]
        for i in range(4):
            for t_name, t_hex in colors.items():
                svg = f'<svg width="100" height="100"><path d="{paths[i]}" fill="{t_hex}" stroke="white" stroke-width="4"/></svg>'
                if t_name == "rec":
                    svg = f'<svg width="100" height="100"><path d="{paths[i]}" fill="#111" stroke="#ccc" stroke-width="4" opacity="0.6"/></svg>'
                png = cairosvg.svg2png(bytestring=svg.encode())
                # --- FIX: ADDED io.BytesIO(png) TO PREVENT CRASH ---
                img = ImageTk.PhotoImage(
                    Image.open(io.BytesIO(png)).resize((70, 70), Image.LANCZOS)
                )
                arrows[f"{i}_{t_name}"] = img
        return arrows

    def parse_sm(self, sm_path):
        self.parsed_notes = []
        if not os.path.exists(sm_path):
            return
        with open(sm_path, "r", encoding="utf-8") as f:
            content = f.read()
        notes_part = content.split("#NOTES:")[-1].split(";")
        measures = notes_part[0].split(":")[-1].strip().split(",")
        for m_idx, measure in enumerate(measures):
            lines = [l.strip() for l in measure.strip().split("\n") if l.strip()]
            for l_idx, line in enumerate(lines):
                beat = (m_idx * 4.0) + (l_idx / float(len(lines)) * 4.0)
                for c_idx, char in enumerate(line[:4]):
                    if char == "1":
                        self.parsed_notes.append((beat, c_idx))

    def generate_chart_file(
        self, save_dir=None, music_file=None, video_file=None, image_file=None
    ):
        if not self.beat_times:
            return
        max_m = int(max(self.beat_times) // 4) + 1
        measures, last_leg = [], -1
        for m in range(max_m):
            lines = []
            for r in range(16):
                bi = (m * 4) + (r / 4.0)
                if any(abs(bi - b) < 0.005 for b in self.beat_times):
                    possible = (
                        [2, 3]
                        if last_leg == 0
                        else [0, 1] if last_leg == 1 else [0, 1, 2, 3]
                    )
                    lane = random.choice(possible)
                    last_leg = 0 if lane in [0, 1] else 1
                    note = ["0"] * 4
                    note[lane] = "1"
                    lines.append("".join(note))
                else:
                    lines.append("0000")
            measures.append("\n".join(lines))

        npm = len(self.beat_times) / (self.duration / 60)
        smart_diff = max(2, min(13, int(npm / 40)))
        title = os.path.splitext(os.path.basename(self.video_path))[0]
        itg_offset = -self.offset + 0.009
        content = [
            f"#TITLE:{title};",
            f"#OFFSET:{itg_offset:.3f};",
            f"#BPMS:0.000={self.tempo:.3f};",
        ]
        if music_file:
            content.append(f"#MUSIC:{music_file};")
        if image_file:
            content.append(f"#BANNER:{image_file};")
        if video_file:
            content.append(f"#BGCHANGES:0.000={video_file}=1.000=1=0=0=,,,;")
        content.extend(
            [
                f"#NOTES:\n     dance-single:\n     ITGMaker:\n     Hard:\n     {smart_diff}:\n     0.8,0.8,0.8,0.8,0.8:",
                ",\n".join(measures),
                ";",
            ]
        )
        sm_path = os.path.join(save_dir if save_dir else self.data_dir, "chart.sm")
        with open(sm_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    # --- FIX: MOVED EXPORT TO BACKGROUND THREAD & ADDED LOADING SCREEN ---
    def generate_chart(self):
        default_name = os.path.splitext(os.path.basename(self.video_path))[0]
        save_path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".sm",
            filetypes=[("StepMania File", "*.sm")],
        )
        if not save_path:
            return

        self.preview_running = False
        if hasattr(self, "cap"):
            self.cap.release()
        pygame.mixer.quit()

        for w in self.root.winfo_children():
            w.destroy()
        tk.Label(
            self.root,
            text="EXPORTING... PLEASE WAIT\nDO NOT CLOSE THE PROGRAM",
            font=("Arial", 18),
        ).place(relx=0.5, rely=0.5, anchor="center")

        threading.Thread(
            target=self.export_thread_logic, args=(save_path, default_name), daemon=True
        ).start()

    def export_thread_logic(self, save_path, default_name):
        song_folder = os.path.join(os.path.dirname(save_path), default_name)
        os.makedirs(song_dir := song_folder, exist_ok=True)
        for old in glob.glob(os.path.join(song_folder, "*.sm")) + glob.glob(
            os.path.join(song_folder, "*.ssc")
        ):
            os.remove(old)

        music_name = f"{default_name}.ogg"
        video_name = f"{default_name}{os.path.splitext(self.video_path)[1]}"
        image_name = f"{default_name}-bn.png"

        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-i",
                self.video_path,
                "-vn",
                "-acodec",
                "libvorbis",
                os.path.join(song_folder, music_name),
            ],
            capture_output=True,
        )
        shutil.copy2(self.video_path, os.path.join(song_folder, video_name))
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-ss",
                "00:00:05",
                "-i",
                self.video_path,
                "-frames:v",
                "1",
                "-update",
                "1",
                os.path.join(song_folder, image_name),
            ],
            capture_output=True,
        )

        self.generate_chart_file(song_folder, music_name, video_name, image_name)
        self.root.after(0, lambda: self.finish_export(song_folder))

    def finish_export(self, folder):
        messagebox.showinfo("Export Done", f"Pack saved to:\n{folder}")
        self.show_upload_screen()

    def show_preview_screen(self):
        for w in self.root.winfo_children():
            w.destroy()
        top = tk.Frame(self.root)
        top.pack()
        tk.Button(
            top,
            text="EXPORT FINAL",
            bg="green",
            fg="white",
            width=20,
            command=self.generate_chart,
        ).pack(pady=5)
        self.canvas = tk.Canvas(self.root, width=1280, height=720, bg="black")
        self.canvas.pack()
        self.arrows = self.create_svg_arrows()
        self.start_preview()

    def start_preview(self):
        self.preview_running = True
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24
        pygame.mixer.init()
        audio = os.path.join(self.data_dir, "preview.wav")
        if not os.path.exists(audio):
            subprocess.run(
                [
                    FFMPEG,
                    "-y",
                    "-i",
                    self.video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    audio,
                ]
            )
        pygame.mixer.music.load(audio)
        pygame.mixer.music.play()
        self.lane_x = [920, 1010, 1100, 1190]
        self.vid_obj = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.create_rectangle(840, 0, 1280, 720, fill="#0d0d0d", outline="")
        self.grid_lines = [
            self.canvas.create_line(840, -100, 1280, -100, fill="#555")
            for _ in range(16)
        ]
        for i, x in enumerate(self.lane_x):
            self.canvas.create_image(x, 100, image=self.arrows[f"{i}_rec"])
        self.note_pool = [self.canvas.create_image(-100, -100) for _ in range(100)]
        self.last_frame_idx = -1
        self.update_preview()

    def update_preview(self):
        if not self.preview_running:
            return
        pos = pygame.mixer.music.get_pos()
        if pos < 0:
            self.root.after(10, self.update_preview)
            return

        music_time = pos / 1000.0
        music_beat = (music_time - self.offset) / self.spb
        idx = int(music_time * self.fps)
        if idx > self.last_frame_idx:
            while self.last_frame_idx < idx:
                ret, frame = self.cap.read()
                self.last_frame_idx += 1
                if not ret:
                    break
            if ret:
                f = cv2.cvtColor(cv2.resize(frame, (840, 720)), cv2.COLOR_BGR2RGB)
                self.tk_img = ImageTk.PhotoImage(Image.fromarray(f))
                self.canvas.itemconfig(self.vid_obj, image=self.tk_img)

        for i in range(16):
            target_beat = (int(music_beat * 2) + i) * 0.5
            y = 100 + (target_beat - music_beat) * self.pixels_per_beat
            self.canvas.coords(self.grid_lines[i], 840, y, 1280, y)

        active = 0
        for beat, lane in self.parsed_notes:
            diff = beat - music_beat
            if -0.5 < diff < 4.0:
                y = 100 + (diff * self.pixels_per_beat)
                c_type = "4th" if (beat % 1.0 < 0.01) else "8th"
                obj = self.note_pool[active]
                self.canvas.coords(obj, self.lane_x[lane], y)
                self.canvas.itemconfig(
                    obj, image=self.arrows[f"{lane}_{c_type}"], state="normal"
                )
                active += 1
                if active >= 100:
                    break

        for i in range(active, 100):
            self.canvas.itemconfig(self.note_pool[i], state="hidden")
        self.root.after(15, self.update_preview)

    def close(self):
        self.preview_running = False
        if hasattr(self, "cap"):
            self.cap.release()
        pygame.mixer.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ITGMaker(root)
    root.mainloop()
