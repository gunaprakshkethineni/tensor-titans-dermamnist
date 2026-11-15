# Tensor Titans: skin lesion app

Built on the course `app_template.py` (Gradio). The template's `prepare` (resize, starter MEAN / STD) and
`classify()` with the top-3 classes are kept. One step is added before them: the photo is cut to a centred
square and shrunk to 28 x 28, exactly how DermaMNIST was made from the original photos (the template would
squash a 600 x 450 photo). This made our model clearly better on full-size photos.

What we added for our model:
- **What the model sees:** the 28 x 28 image the network really gets, shown large.
- **How the 5 networks vote:** model.pt is an ensemble of 5 CNNs. The app runs each one on its own and
  shows whether they agree (all agree / fairly sure / unsure).
- **About the top class:** a plain-language explanation, how often our model finds this class on the
  test set (e.g. 67 of 100 melanomas), and a warning for classes that can be (pre-)cancerous.
- **Example photos in 3 qualities**, all from the validation split (never training or test images):
  sharp originals (600 x 450, HAM10000), slightly blurred (64 x 64) and very blurred (28 x 28, DermaMNIST).
  The true class of each example is listed above its row.

## Run it

**Google Colab (easiest)**
1. Open https://colab.research.google.com, create a new notebook.
2. Upload `tensor_titans_app.zip` with the folder icon on the left.
3. Run this in a cell:
   ```
   !unzip -q -o tensor_titans_app.zip && pip install -q gradio && cd tensor_titans_app && python app.py
   ```
4. After about 30 s it prints a public link like `https://xxxx.gradio.live`. Open it (also works on a phone).
   The link works while the cell is running (up to about 1 week). Stop it with the stop button of the cell.

**Hugging Face Spaces (a fixed link)**
1. Log in at https://huggingface.co, then New, Space: any name, SDK **Gradio**, hardware **CPU basic (free)**.
2. Files, Add file, Upload files: `app.py`, `model.pt`, `requirements.txt` and the `examples` folder.
3. Wait a few minutes until the Space shows "Running".

**Your laptop**
```
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:7860

## Demo script (about 3 minutes)

1. **Very blurred melanoma** (first image of the last row): the photo from our poster.
   Melanoma 60 %, benign keratosis 30 %, mole 6 %. All 5 networks agree.
   Point at "what the model sees": the network only gets 28 x 28 pixels.
2. **Sharp melanoma** (first image of the top row): the original high-resolution photo.
   The app shrinks it to 28 x 28 first. Melanoma 46 %, but only 3 of 5 networks agree: the app says so.
3. **Same melanoma, slightly blurred** (first of the middle row): almost the same answer (48 %),
   because the model always works on 28 x 28.
4. **Sharp benign keratosis** (last of the top row): the model says melanoma (42 %), a mistake.
   This is the trade-off on our poster: we find many more melanomas, but we also raise more false alarms.
5. Close with the disclaimer: a student project, not a medical device.

Photos: HAM10000 (Tschandl, Rosendahl & Kittler, Sci. Data 2018), CC BY-NC 4.0.
