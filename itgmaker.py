import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import librosa
import numpy as np
import threading
import time
import pygame
import subprocess
import os
import cairosvg
import io
import shutil
import sys


# --- FFmpeg Robust Discovery ---
def get_ffmpeg_path():
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except ImportError:
        pass

    path = shutil.which("ffmpeg")
    if path:
        return path

    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
    if os.path.exists(local_path):
        return local_path

    conda_path = os.path.join(sys.prefix, "Library", "bin", "ffmpeg.exe")
    if os.path.exists(conda_path):
        return conda_path
    return None


FFMPEG = get_ffmpeg_path()


class ITGMaker:
    def __init__(self, root):
        self.root = root
        root.title("ITGmania Generator")
        root.geometry("1280x720")

        self.video_path = None
        self.preview_running = False
        self.beat_times = []
        self.tempo = 120
        self.scroll_speed = 250  # Slower for even better playability

        if FFMPEG is None:
            messagebox.showerror(
                "FFmpeg Missing", "FFmpeg not found! Run 'pip install imageio-ffmpeg'"
            )

        # Create an isolated temp directory, and wipe any previous session data
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

    def show_preview_screen(self):
        for w in self.root.winfo_children():
            w.destroy()
        top = tk.Frame(self.root)
        top.pack()
        tk.Button(
            top,
            text="Generate SM Chart",
            bg="green",
            fg="white",
            width=20,
            command=self.generate_chart,
        ).pack(pady=5)
        self.canvas = tk.Canvas(self.root, width=1280, height=720, bg="black")
        self.canvas.pack()
        self.arrows = self.create_svg_arrows()
        self.start_preview()

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
        if not os.path.exists(audio_file):
            return

        y, sr = librosa.load(audio_file, sr=44100)
        dur = y.shape[0] / sr
        _, y_percussive = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

        # 1. LOCK BPM
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        self.tempo = float(np.atleast_1d(tempo)[0])
        spb = 60.0 / self.tempo
        first_beat_time = (
            librosa.frames_to_time(beats[0], sr=sr) if len(beats) > 0 else 0.0
        )

        # 2. ARROWS ON FULL BEATS WITH NOISE-FLOOR AWARE SILENCE DETECTION
        # Intros have tape hiss or background noise that can trigger arrows if the threshold is too low.
        # Instead of 45db split or average volume, we will look at the ABSOLUTE PEAK of the song,
        # and treat anything less than 10% of the PEAK volume as silence.
        rms = librosa.feature.rms(y=y)[0]
        max_rms = np.max(rms)
        noise_floor_threshold = max_rms * 0.10  # 10% of peak volume

        final_grid_beats = []
        num_beats = int((dur - first_beat_time) / spb)
        for i in range(num_beats + 1):
            beat_time = first_beat_time + i * spb
            frame_idx = librosa.time_to_frames(beat_time, sr=sr)

            # Check a window of +/- 0.1 seconds around the beat line
            # so we don't skip an audible beat just because it didn't
            # land exactly perfectly on the exact millisecond of the grid line!
            window_frames = librosa.time_to_frames(0.1, sr=sr)
            start_f = max(0, frame_idx - window_frames)
            end_f = min(len(rms), frame_idx + window_frames)

            # If the max volume ANYWHERE near this beat is above the noisy floor
            if start_f < end_f and np.max(rms[start_f:end_f]) > noise_floor_threshold:
                final_grid_beats.append(i * 1.0)

        self.beat_times = sorted(final_grid_beats)
        self.offset = first_beat_time
        self.spb = spb

        # 4. WRITE THE SOURCE OF TRUTH
        self.generate_chart_file()

        # 5. READ FROM SOURCE OF TRUTH
        self.parse_sm(os.path.join(self.data_dir, "chart.sm"))

        if os.path.exists(audio_file):
            os.remove(audio_file)
        self.root.after(0, self.show_preview_screen)

    def create_svg_arrows(self):
        arrows = {}
        colors = {"4th": "#ff2424", "8th": "#2466ff", "rec": "#1a1a1a"}
        paths = [
            "M10,50 L55,10 L55,35 L90,35 L90,65 L55,65 L55,90 Z",  # L
            "M50,90 L10,45 L35,45 L35,10 L65,10 L65,45 L90,45 Z",  # D
            "M50,10 L90,55 L65,55 L65,90 L35,90 L35,55 L10,55 Z",  # U
            "M90,50 L45,10 L45,35 L10,35 L10,65 L45,65 L45,90 Z",  # R
        ]
        for i in range(4):
            for t_name, t_hex in colors.items():
                svg = f'<svg width="100" height="100"><path d="{paths[i]}" fill="{t_hex}" stroke="white" stroke-width="4"/></svg>'
                if t_name == "rec":
                    svg = f'<svg width="100" height="100"><path d="{paths[i]}" fill="#111" stroke="#ccc" stroke-width="4" opacity="0.6"/></svg>'
                png = cairosvg.svg2png(bytestring=svg.encode())
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

        # Extract metadata for sync
        for line in content.split(";"):
            if "#OFFSET:" in line:
                self.offset = -float(line.split(":")[-1])
            if "#BPMS:" in line:
                bpm_str = line.split("=")[-1]
                self.tempo = float(bpm_str)
                self.spb = 60.0 / self.tempo

        notes_part = content.split("#NOTES:")[-1].split(";")
        if not notes_part:
            return
        measures = notes_part[0].split(":")[-1].strip().split(",")
        for m_idx, measure in enumerate(measures):
            lines = [l.strip() for l in measure.strip().split("\n") if l.strip()]
            for l_idx, line in enumerate(lines):
                beat = (m_idx * 4.0) + (l_idx / float(len(lines)) * 4.0)
                for char_idx, char in enumerate(line[:4]):
                    if char == "1":
                        self.parsed_notes.append((beat, char_idx))

    def generate_chart_file(
        self, save_dir=None, audio_name=None, video_name=None, image_name=None
    ):
        if not self.beat_times:
            return
            
        # Dynamically calculate difficulty rating based on ACTUAL note density (Total notes / Total song length)
        total_notes = len(self.beat_times)
        song_length_seconds = max(self.beat_times) if total_notes > 0 else 1.0
        actual_nps = total_notes / song_length_seconds
        
        # In StepMania: ~2 NPS is generally easy (meter 2-4)
        # ~4-5 NPS is medium (meter 5-7)
        # ~6-8+ NPS is hard (meter 8+)
        meter = max(1, int(round(actual_nps * 1.6))) 
        
        if meter <= 2:
            diff_name = "Beginner"
        elif meter <= 4:
            diff_name = "Easy"
        elif meter <= 7:
            diff_name = "Medium"
        elif meter <= 9:
            diff_name = "Hard"
        else:
            diff_name = "Challenge"
            
        max_m = int(max(self.beat_times) // 4) + 1
        measures = []
        last_leg = -1  # 0 for left leg (0,1), 1 for right leg (2,3)
        last_lane = -1
        second_last_lane = -1
        import random

        for m in range(max_m):
            lines = []
            for r in range(16):
                bi = (m * 4) + (r / 4.0)
                hit = any(abs(bi - b) < 0.005 for b in self.beat_times)
                if hit:
                    # Switch legs perfectly every step: Left leg = lanes 0,1. Right leg = lanes 2,3
                    if last_leg == 0:
                        possible_lanes = [2, 3]  # Must use Right Leg
                        last_leg = 1
                    elif last_leg == 1:
                        possible_lanes = [0, 1]  # Must use Left Leg
                        last_leg = 0
                    else:  # First note, pick randomly
                        possible_lanes = [0, 1, 2, 3]
                        lane = random.choice(possible_lanes)
                        last_leg = 0 if lane in [0, 1] else 1

                    lane = random.choice(possible_lanes) if last_leg != -1 else lane
                    last_lane = lane

                    note = ["0", "0", "0", "0"]
                    note[lane] = "1"
                    lines.append("".join(note))
                else:
                    lines.append("0000")
            measures.append("\n".join(lines))

        title = os.path.basename(self.video_path)
        content = [f"#TITLE:{os.path.splitext(title)[0]};"]

        if audio_name:
            content.append(f"#MUSIC:{audio_name};")
        if image_name:
            content.append(f"#BANNER:{image_name};")
        if video_name:
            content.append(f"#BACKGROUND:{video_name};")

        content.extend(
            [
                f"#OFFSET:{-self.offset:.3f};",
                f"#BPMS:0.000={self.tempo:.3f};",
                f"\n#NOTES:\n     dance-single:\n     ITGMaker:\n     {diff_name}:\n     {meter}:\n     0.8,0.8,0.8,0.8,0.8:",
                ",\n".join(measures),
                ";",
            ]
        )

        sm_path = os.path.join(self.data_dir, "chart.sm")
        if save_dir:
            sm_path = os.path.join(save_dir, f"{os.path.splitext(title)[0]}.sm")

        with open(sm_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    def generate_chart(self):
        default_name = os.path.splitext(os.path.basename(self.video_path))[0]
        video_dir = os.path.dirname(self.video_path)
        
        # Use classic Save As dialog (gives you the Save button and filename field!)
        save_path = filedialog.asksaveasfilename(
            title="Save StepMania Song Pack",
            initialdir=video_dir,
            initialfile=default_name,
            defaultextension=".sm",
            filetypes=[("StepMania File", "*.sm")]
        )
        
        if not save_path:
            return
            
        save_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        
        song_folder = os.path.join(save_dir, base_name)

        if not os.path.exists(song_folder):
            os.makedirs(song_folder)
        else:
            # IMPORTANT: Delete ANY stale .sm and particularly .ssc files
            # If we don't do this, StepMania WILL cache and read the old 'Hard 8' file instead
            import glob
            for old_file in glob.glob(os.path.join(song_folder, "*.sm")):
                os.remove(old_file)
            for old_file in glob.glob(os.path.join(song_folder, "*.ssc")):
                os.remove(old_file)

        # 1. Extract raw Audio and compress to .ogg for Game Compatibility
        audio_name = f"{base_name}.ogg"
        audio_path = os.path.join(song_folder, audio_name)
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-i",
                self.video_path,
                "-vn",
                "-c:a",
                "libvorbis",
                "-q:a",
                "4",
                audio_path,
            ]
        )

        # 2. Copy the original Video as the background
        video_ext = os.path.splitext(self.video_path)[1]
        video_name = f"{base_name}-bg{video_ext}"
        video_dest = os.path.join(song_folder, video_name)
        shutil.copy2(self.video_path, video_dest)

        # 3. Extract a frame from the video to serve as the song banner/image
        image_name = f"{base_name}-bn.png"
        image_path = os.path.join(song_folder, image_name)
        # We try to grab a frame 5 seconds into the video.
        # Added -update 1 so ffmpeg knows it's a single static image and not an image sequence
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
                "-q:v",
                "2",
                "-update",
                "1",
                image_path,
            ]
        )

        # 4. Generate the actual full .sm file into that folder pointing to the new files
        self.generate_chart_file(
            save_dir=song_folder,
            audio_name=audio_name,
            video_name=video_name,
            image_name=image_name,
        )

        messagebox.showinfo(
            "Export Successful",
            f"Full Song Pack exported successfully to:\n{song_folder}\n\nYou can now copy this folder directly into your ITGmania/Songs folder.",
        )

    def start_preview(self):
        self.preview_running = True
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24
        pygame.mixer.init()
        audio = os.path.join(self.data_dir, "preview_audio.wav")
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

        # Guide Lines: Make them consistent thickness and color to avoid confusion
        self.grid_lines = []
        for i in range(16):
            self.grid_lines.append(
                self.canvas.create_line(840, -100, 1280, -100, fill="#555", width=1)
            )

        for i, x in enumerate(self.lane_x):
            self.canvas.create_image(x, 100, image=self.arrows[f"{i}_rec"])

        self.note_pool = [self.canvas.create_image(-100, -100) for _ in range(80)]
        self.last_frame_idx, self.pixels_per_beat = -1, 200
        self.update_preview()

    def update_preview(self):
        if not self.preview_running:
            return
        # Locking to audio clock for millisecond precision
        pos = pygame.mixer.music.get_pos()
        if pos < 0:
            self.root.after(10, self.update_preview)
            return

        music_time = pos / 1000.0
        music_beat = (
            music_time - self.offset
        ) / self.spb  # Correctly respecting the audio offset!

        # 1. Vid Sync
        idx = int(music_time * self.fps)
        if idx > self.last_frame_idx:
            while self.last_frame_idx < idx:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.last_frame_idx += 1
            if ret:
                f = cv2.cvtColor(cv2.resize(frame, (840, 720)), cv2.COLOR_BGR2RGB)
                self.tk_img = ImageTk.PhotoImage(Image.fromarray(f))
                self.canvas.itemconfig(self.vid_obj, image=self.tk_img)

        # 2. PIXEL-LOCKED GRID (Locked to receptors at Y=100)
        curr_half_beat = int(music_beat * 2)
        for i in range(16):
            target_beat = (curr_half_beat + i) * 0.5
            y = 100 + (target_beat - music_beat) * self.pixels_per_beat
            self.canvas.coords(self.grid_lines[i], 840, y, 1280, y)

        # 3. PIXEL-LOCKED ARROWS (Source: chart.sm)
        active = 0
        for beat, lane in self.parsed_notes:
            diff = beat - music_beat
            if -0.5 < diff < 4.0:
                y = 100 + (diff * self.pixels_per_beat)
                c_type = "4th" if (beat % 1.0 < 0.01) else "8th"

                if active < len(self.note_pool):
                    obj = self.note_pool[active]
                    self.canvas.coords(obj, self.lane_x[lane], y)
                    self.canvas.itemconfig(
                        obj, image=self.arrows[f"{lane}_{c_type}"], state="normal"
                    )
                    active += 1

        for i in range(active, len(self.note_pool)):
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
