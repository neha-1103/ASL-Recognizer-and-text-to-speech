import os
import sys
import operator
import time
from string import ascii_uppercase

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# Keras import depending on installation; try common locations
try:
    # TF 2.x
    from keras.models import model_from_json
except Exception:
    try:
        from tensorflow.keras.models import model_from_json
    except Exception:
        raise ImportError("Could not import model_from_json from keras or tensorflow.keras")

# Optional hunspell
try:
    import hunspell
    HUNSPELL_AVAILABLE = True
except Exception:
    HUNSPELL_AVAILABLE = False
    import difflib

# ----------------------------
# Configuration / constants
# ----------------------------
MODEL_DIR = "Model"
PIXEL_SIZE = 128          # taken from your file (resized to 128x128)
CAMERA_INDEX = 0          # default webcam
FRAME_DELAY_MS = 30       # GUI update interval
CONFIRM_FRAMES = 60       # frames required to accept a character
SIMILARITY_THRESHOLD = 20 # tolerance when comparing counters

# ----------------------------
# Helper functions
# ----------------------------
def safe_join(*parts):
    return os.path.join(*parts)

def load_json_model(json_path, weights_path):
    """Load a Keras model saved as JSON + H5. Returns None on failure."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    with open(json_path, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

def preprocess_for_model(binary_image, size=PIXEL_SIZE):
    """
    Preprocess a binary ROI for model:
    - expects single-channel binary (0/255) or grayscale array
    - resize to (size,size)
    - invert if necessary, normalize to 0..1
    - return shape (size,size,1)
    """
    # convert to uint8 if not
    img = binary_image.copy()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    # if 3 channels, convert to gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize
    img = cv2.resize(img, (size, size))
    # ensure binary-like: threshold
    _, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    # normalize
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (size,size,1)
    return img

# ----------------------------
# Main Application
# ----------------------------
class ASLApplication:
    def __init__(self, model_dir=MODEL_DIR):
        self.directory = model_dir

        # Hunspell or fallback
        if HUNSPELL_AVAILABLE:
            try:
                # Common Unix path - allow exceptions to fallback
                self.hs = hunspell.HunSpell('/usr/share/hunspell/en_US.dic',
                                            '/usr/share/hunspell/en_US.aff')
            except Exception:
                # try default constructor (some systems require different paths)
                try:
                    self.hs = hunspell.HunSpell()
                except Exception:
                    self.hs = None
        else:
            self.hs = None

        # Video capture
        self.vs = cv2.VideoCapture(CAMERA_INDEX)
        if not self.vs.isOpened():
            raise RuntimeError("Could not open webcam. Check CAMERA_INDEX or camera permissions.")

        self.current_image = None
        self.current_image2 = None

        # Load models
        try:
            # main A-Z model (binary + blank as first output)
            self.loaded_model = load_json_model(
                safe_join(self.directory, "model-bw.json"),
                safe_join(self.directory, "model-bw.h5")
            )
            # hierarchical models
            self.loaded_model_dru = load_json_model(
                safe_join(self.directory, "model-bw_dru.json"),
                safe_join(self.directory, "model-bw_dru.h5")
            )
            self.loaded_model_tkdi = load_json_model(
                safe_join(self.directory, "model-bw_tkdi.json"),
                safe_join(self.directory, "model-bw_tkdi.h5")
            )
            self.loaded_model_smn = load_json_model(
                safe_join(self.directory, "model-bw_smn.json"),
                safe_join(self.directory, "model-bw_smn.h5")
            )
        except Exception as e:
            # re-raise with helpful message
            raise RuntimeError(f"Error loading models: {e}")

        # counters for temporal smoothing
        self.ct = {'blank': 0}
        for ch in ascii_uppercase:
            self.ct[ch] = 0
        self.blank_flag = 0

        # UI state
        self.str_sentence = ""
        self.word = ""
        self.current_symbol = "Empty"

        # Build UI
        self._build_ui()

        # start loop
        self.video_loop()

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("750x750")

        self.panel = tk.Label(self.root)
        self.panel.place(x=135, y=10, width=640, height=640)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=460, y=95, width=310, height=310)

        self.title_label = tk.Label(self.root, text="Sign Language to Text", font=("Courier", 40, "bold"))
        self.title_label.place(x=31, y=17)

        self.panel3 = tk.Label(self.root, text=self.current_symbol, font=("Courier", 50))
        self.panel3.place(x=500, y=640)

        tk.Label(self.root, text="Character :", font=("Courier", 40, "bold")).place(x=10, y=640)
        tk.Label(self.root, text="Word :", font=("Courier", 40, "bold")).place(x=10, y=700)
        tk.Label(self.root, text="Sentence :", font=("Courier", 40, "bold")).place(x=10, y=760)

        self.panel4 = tk.Label(self.root, text=self.word, font=("Courier", 40))
        self.panel4.place(x=220, y=700)

        self.panel5 = tk.Label(self.root, text=self.str_sentence, font=("Courier", 40))
        self.panel5.place(x=350, y=760)

        self.T4 = tk.Label(self.root, text="Suggestions", fg="red", font=("Courier", 40, "bold"))
        self.T4.place(x=250, y=820)

        # suggestion buttons
        self.bt_about = tk.Button(self.root, text="About", command=self.action_call, font=("Courier", 14))
        self.bt_about.place(x=825, y=0)

        self.bt_sugg1 = tk.Button(self.root, text="", command=self.action1, font=("Courier", 12), width=18)
        self.bt_sugg1.place(x=26, y=890)
        self.bt_sugg2 = tk.Button(self.root, text="", command=self.action2, font=("Courier", 12), width=18)
        self.bt_sugg2.place(x=325, y=890)
        self.bt_sugg3 = tk.Button(self.root, text="", command=self.action3, font=("Courier", 12), width=18)
        self.bt_sugg3.place(x=625, y=890)
        self.bt_sugg4 = tk.Button(self.root, text="", command=self.action4, font=("Courier", 12), width=18)
        self.bt_sugg4.place(x=125, y=950)
        self.bt_sugg5 = tk.Button(self.root, text="", command=self.action5, font=("Courier", 12), width=18)
        self.bt_sugg5.place(x=425, y=950)
    
    # def _build_ui(self):
    #     self.root.title("Sign Language to Text Converter")
    #     self.root.geometry("1100x800")  # Increased height
    #     self.root.resizable(True, True)
    #     self.root.configure(bg="white")

    # # Title
    #     title_label = tk.Label(self.root, text="Sign Language to Text", font=("Courier", 28, "bold"), bg="white")
    #     title_label.grid(row=0, column=0, columnspan=2, pady=20)

    #     # Frame for camera and processed image
    #     frame = tk.Frame(self.root, bg="white")
    #     frame.grid(row=1, column=0, columnspan=2, pady=10)

    #     self.camera_label = tk.Label(frame, bg="white")
    #     self.camera_label.grid(row=0, column=0, padx=10)

    #     self.processed_label = tk.Label(frame, bg="white", relief="solid", bd=2)
    #     self.processed_label.grid(row=0, column=1, padx=10)

    # # Output section
    #     output_frame = tk.Frame(self.root, bg="white")
    #     output_frame.grid(row=2, column=0, columnspan=2, pady=20)

    # # Character
    #     tk.Label(output_frame, text="Character :", font=("Courier", 20, "bold"), bg="white").grid(row=0, column=0, sticky="w", pady=5)
    #     self.char_label = tk.Label(output_frame, text="", font=("Courier", 20, "bold"), bg="white")
    #     self.char_label.grid(row=0, column=1, sticky="w", pady=5)

    # # Word
    #     tk.Label(output_frame, text="Word :", font=("Courier", 20, "bold"), bg="white").grid(row=1, column=0, sticky="w", pady=5)
    #     self.word_label = tk.Label(output_frame, text="", font=("Courier", 20, "bold"), bg="white")
    #     self.word_label.grid(row=1, column=1, sticky="w", pady=5)

    #     # Sentence
    #     tk.Label(output_frame, text="Sentence :", font=("Courier", 20, "bold"), bg="white").grid(row=2, column=0, sticky="nw", pady=5)
    #     self.sentence_label = tk.Label(
    #     output_frame,
    #     text="",
    #     font=("Courier", 20, "bold"),
    #     bg="white",
    #     wraplength=950,
    #     justify="left",
    #     anchor="w"
    #     )
    #     self.sentence_label.grid(row=2, column=1, sticky="w", pady=5)
    
    # # Allow dynamic resizing
    #     self.root.grid_rowconfigure(3, weight=1)
    #     self.root.grid_columnconfigure(0, weight=1)
    #     self.root.grid_columnconfigure(1, weight=1)


    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            # flip for mirror
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ROI coordinates (same idea as original): right half square
            x1 = int(0.5 * w)
            y1 = 10
            x2 = w - 10
            y2 = y1 + int(0.5 * w)  # square region
            # ensure bounds
            y2 = min(y2, h - 10)

            # draw rectangle on original preview
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

            # show preview on left panel
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # extract ROI and preprocess similar to original
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                # in case ROI invalid
                self.root.after(FRAME_DELAY_MS, self.video_loop)
                return

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # pass the binary image to predict
            self.predict(res)

            # show processed ROI on panel2
            self.current_image2 = Image.fromarray(res)
            imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk2
            self.panel2.config(image=imgtk2)

            # update UI text
            self.panel3.config(text=self.current_symbol, font=("Courier", 50))
            self.panel4.config(text=self.word, font=("Courier", 40))
            self.panel5.config(text=self.str_sentence, font=("Courier", 40))

            # suggestions
            suggestions = self.get_suggestions(self.word)
            # fill buttons up to 5
            buttons = [self.bt_sugg1, self.bt_sugg2, self.bt_sugg3, self.bt_sugg4, self.bt_sugg5]
            for i, btn in enumerate(buttons):
                if i < len(suggestions):
                    btn.config(text=suggestions[i])
                else:
                    btn.config(text="")

        # schedule next call
        self.root.after(FRAME_DELAY_MS, self.video_loop)

    def get_suggestions(self, word):
        if not word:
            return []
        if self.hs:
            try:
                preds = self.hs.suggest(word)
                return preds[:5]
            except Exception:
                pass
        # fallback: use difflib on a small built-in english wordlist if hunspell not available
        # small default wordlist - could be extended
        sample_words = ["hello", "help", "how", "house", "have", "happy", "hold", "home", "helping", "halo"]
        matches = difflib.get_close_matches(word, sample_words, n=5, cutoff=0.1)
        return matches

    def predict(self, test_image):
        """
        test_image: binary (0/255) numpy array of ROI
        The model outputs assumed:
          - main model: first output is 'blank' then probabilities for A..Z in order
          - dru model: outputs probs for D,R,U (3)
          - tkdi model: outputs probs for D,T,K,I (4)
          - smn model: outputs probs for S,M,N (3)
        """
        # Preprocess
        x = preprocess_for_model(test_image, PIXEL_SIZE)
        x_batch = np.expand_dims(x, axis=0)  # shape (1, PIXEL_SIZE, PIXEL_SIZE, 1)

        # raw predictions
        try:
            result = self.loaded_model.predict(x_batch, verbose=0)
            result_dru = self.loaded_model_dru.predict(x_batch, verbose=0)
            result_tkdi = self.loaded_model_tkdi.predict(x_batch, verbose=0)
            result_smn = self.loaded_model_smn.predict(x_batch, verbose=0)
        except Exception as e:
            print("Model predict error:", e)
            return

        # build dictionary for main outputs
        prediction = {}
        prediction['blank'] = float(result[0][0])
        idx = 1
        for ch in ascii_uppercase:
            prediction[ch] = float(result[0][idx])
            idx += 1

        # sort descending
        sorted_main = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        chosen = sorted_main[0][0]

        # Hierarchical disambiguation
        # If top is among D,R,U => use dru model
        if chosen in ('D', 'R', 'U'):
            tmp = {'D': float(result_dru[0][0]), 'R': float(result_dru[0][1]), 'U': float(result_dru[0][2])}
            chosen = sorted(tmp.items(), key=operator.itemgetter(1), reverse=True)[0][0]

        # If top among D,I,K,T => use tkdi model
        if chosen in ('D', 'I', 'K', 'T'):
            tmp = {'D': float(result_tkdi[0][0]), 'I': float(result_tkdi[0][1]),
                   'K': float(result_tkdi[0][2]), 'T': float(result_tkdi[0][3])}
            chosen = sorted(tmp.items(), key=operator.itemgetter(1), reverse=True)[0][0]

        # If top among M,N,S => use smn
        if chosen in ('M', 'N', 'S'):
            tmp = {'M': float(result_smn[0][0]), 'N': float(result_smn[0][1]), 'S': float(result_smn[0][2])}
            top_smn = sorted(tmp.items(), key=operator.itemgetter(1), reverse=True)[0][0]
            # original logic tried to set 'S' only in some cases - here adopt simpler rule
            chosen = top_smn

        self.current_symbol = chosen

        # temporal smoothing logic (kept close to original)
        if self.current_symbol == 'blank':
            for ch in ascii_uppercase:
                self.ct[ch] = 0

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > CONFIRM_FRAMES:
            # check differences vs others
            for ch in ascii_uppercase:
                if ch == self.current_symbol:
                    continue
                diff = abs(self.ct[self.current_symbol] - self.ct[ch])
                if diff <= SIMILARITY_THRESHOLD:
                    # ambiguous, reset and wait
                    self.ct['blank'] = 0
                    for c in ascii_uppercase:
                        self.ct[c] = 0
                    return
            # accept symbol
            self.ct['blank'] = 0
            for c in ascii_uppercase:
                self.ct[c] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str_sentence) > 0:
                        self.str_sentence += " "
                    self.str_sentence += self.word
                    self.word = ""
            else:
                # add char to word
                if len(self.str_sentence) > 1000:
                    self.str_sentence = ""  # keep sentence bounded
                self.blank_flag = 0
                self.word += self.current_symbol

    # suggestion actions
    def action1(self):
        preds = self.get_suggestions(self.word)
        if len(preds) > 0:
            self.word = ""
            self.str_sentence += " " + preds[0] if self.str_sentence else preds[0]

    def action2(self):
        preds = self.get_suggestions(self.word)
        if len(preds) > 1:
            self.word = ""
            self.str_sentence += " " + preds[1] if self.str_sentence else preds[1]

    def action3(self):
        preds = self.get_suggestions(self.word)
        if len(preds) > 2:
            self.word = ""
            self.str_sentence += " " + preds[2] if self.str_sentence else preds[2]

    def action4(self):
        preds = self.get_suggestions(self.word)
        if len(preds) > 3:
            self.word = ""
            self.str_sentence += " " + preds[3] if self.str_sentence else preds[3]

    def action5(self):
        preds = self.get_suggestions(self.word)
        if len(preds) > 4:
            self.word = ""
            self.str_sentence += " " + preds[4] if self.str_sentence else preds[4]

    def destructor(self):
        print("Closing Application...")
        try:
            self.vs.release()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        cv2.destroyAllWindows()
        # exit program
        try:
            sys.exit(0)
        except SystemExit:
            pass

    def action_call(self):
        # Simple About dialog
        top = tk.Toplevel(self.root)
        top.title("About")
        top.geometry("600x300")
        tk.Label(top, text="Efforts By - Converted App", fg="red", font=("Courier", 20, "bold")).pack(pady=10)
        tk.Label(top, text="Converted to cleaned realtime script", font=("Courier", 14)).pack(pady=5)
        tk.Button(top, text="Close", command=top.destroy).pack(pady=20)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    print("Starting Application...")
    try:
        app = ASLApplication(model_dir=MODEL_DIR)
        app.root.mainloop()
    except Exception as e:
        print("Fatal error:", e)
        raise
