"""Tensor Titans - skin lesion app (DermaMNIST challenge), built on the course app_template.py.

Upload a skin-lesion image -> top-3 classes with probabilities (as in the template), plus:
  * "what the model sees": the 28 x 28 image the network really gets,
  * how the 5 networks inside our ensemble vote (do they agree?),
  * a short explanation of the top class and how often our model finds this class on the test set,
  * example photos in 3 qualities: sharp originals, slightly blurred (64 x 64), very blurred (28 x 28).

Colab:                 !pip install -q gradio   then  !python app.py   (model.pt and examples/ next to it)
                       -> share=True prints a public link you can open on your phone
Hugging Face Spaces:   new Space (SDK: Gradio), upload app.py, model.pt, requirements.txt and the examples/ folder
Local:                 pip install -r requirements.txt   then  python app.py
Do not change MEAN / STD: they must be the numbers of the starter notebook.
"""
import glob
import os

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

MEAN = [0.7631, 0.5381, 0.5614]
STD = [0.1366, 0.1543, 0.1692]
CLASSES = ["actinic keratoses", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
           "melanoma", "melanocytic nevi", "vascular lesions"]

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pt")
if not os.path.exists(MODEL_PATH):                         # in the repository, model.pt sits one folder up
    MODEL_PATH = os.path.join(HERE, "..", "model.pt")
model = torch.jit.load(MODEL_PATH, map_location="cpu")    # our exported ensemble
model.eval()
networks = list(model.models.children())       # the 5 CNNs inside model.pt (each votes on its own)

prepare = v2.Compose([                           # unchanged from app_template.py
    v2.ToImage(),                                  # PIL image -> tensor (3, H, W)
    v2.Resize((28, 28), antialias=True),           # the size the model was trained on
    v2.ToDtype(torch.float32, scale=True),         # 0...255 -> 0...1
    v2.Normalize(MEAN, STD),                       # exactly like in the starter notebook
])



def to_model_size(image):
    """Like DermaMNIST was made: cut the photo to a centred square, then shrink it to 28 x 28.

    The template resizes directly, which squashes a 600 x 450 photo. DermaMNIST cropped to a square first,
    so we do the same - uploaded photos then look like the training images. A 28 x 28 image stays unchanged.
    """
    image = image.convert("RGB")
    w, h = image.size
    s = min(w, h)
    image = image.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return image.resize((28, 28), resample=4)                # 4 = BOX: average of each pixel block


# plain-language info for the result card
INFO = {
    "actinic keratoses": ("Rough, scaly patches from sun damage (incl. Bowen's disease). "
                          "Pre-cancerous: can turn into skin cancer.", True),
    "basal cell carcinoma": ("The most common skin cancer. Grows slowly and rarely spreads, "
                             "but needs treatment.", True),
    "benign keratosis": ("Harmless growths such as seborrheic keratosis or age spots.", False),
    "dermatofibroma": ("A harmless, firm little bump in the skin.", False),
    "melanoma": ("The most dangerous skin cancer, from pigment cells. Early detection matters a lot.", True),
    "melanocytic nevi": ("A common mole. Harmless in almost all cases.", False),
    "vascular lesions": ("Lesions of small blood vessels, e.g. cherry angiomas. Usually harmless.", False),
}
# how many of 100 cases of each class our model.pt found on the official test set (sensitivity)
TEST_HITS = {"actinic keratoses": 55, "basal cell carcinoma": 57, "benign keratosis": 47, "dermatofibroma": 52,
             "melanoma": 67, "melanocytic nevi": 68, "vascular lesions": 93}


def classify(image):
    """PIL image -> {class name: probability} for the 3 most likely classes (same as the template)."""
    x = prepare(to_model_size(image)).unsqueeze(0)          # (1, 3, 28, 28)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    top = probs.topk(3)
    return {CLASSES[i]: float(p) for p, i in zip(top.values, top.indices)}


def eight_views(x):
    """The 8 flips / 90 degree rotations that model.pt also averages over."""
    views = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)]
    return views + [v.transpose(-1, -2) for v in views]


def network_votes(image):
    """Each of the 5 networks on its own: (predicted class, its probability)."""
    x = prepare(to_model_size(image)).unsqueeze(0)
    votes = []
    with torch.no_grad():
        for net in networks:
            logits = torch.stack([net(v) for v in eight_views(x)]).mean(0)
            p = F.softmax(logits, dim=1)[0]
            votes.append((CLASSES[int(p.argmax())], float(p.max())))
    return votes


def what_the_model_sees(image):
    """The 28 x 28 input, shown large and pixelated."""
    return to_model_size(image).resize((224, 224), resample=0)   # 0 = nearest neighbour: keep the pixels


def result_card(top3, votes):
    """HTML card. Every element gets its own colour, so it is readable in light and dark mode."""
    name, prob = next(iter(top3.items()))
    text, serious = INFO[name]
    agree = sum(v[0] == name for v in votes)
    ink, muted = "#1f1f1f", "#55534e"
    chips = "".join(
        f'<div style="flex:1;min-width:92px;padding:8px 6px;border-radius:8px;text-align:center;'
        f'background:{"#2a78d6" if c == name else "#e9e8e4"}">'
        f'<div style="font-size:12px;color:{"#dce9fb" if c == name else muted}">CNN {k + 1}</div>'
        f'<div style="font-weight:700;font-size:13px;color:{"#ffffff" if c == name else ink}">{c}</div>'
        f'<div style="font-size:13px;color:{"#ffffff" if c == name else ink}">{p:.0%}</div></div>'
        for k, (c, p) in enumerate(votes))
    if agree == 5:
        verdict = "All 5 networks agree."
    elif agree >= 3:
        verdict = f"{agree} of 5 networks agree. Fairly sure."
    else:
        verdict = f"Only {agree} of 5 networks agree. The model is unsure: treat this result with care."
    warn = ""
    if serious:
        warn = (f'<div style="margin-top:10px;padding:10px 12px;border-radius:8px;background:#fff1ea;'
                f'border-left:5px solid #eb6834;color:{ink}">This class can be (pre-)cancerous. '
                f'With a real skin spot: please see a dermatologist.</div>')
    low = ""
    if prob < 0.5:
        low = (f'<div style="margin-top:6px;color:#8a5a00">Top probability is only {prob:.0%}: '
               f'look at the other two classes as well.</div>')
    return f"""
<div style="font-family:Inter,system-ui,sans-serif;color:{ink};background:#fcfcfb;border:1px solid #e4e3df;
            border-radius:12px;padding:14px 16px">
  <div style="font-size:13px;color:{muted};margin-bottom:6px">HOW THE 5 NETWORKS INSIDE OUR MODEL VOTE</div>
  <div style="display:flex;gap:6px;flex-wrap:wrap">{chips}</div>
  <div style="margin:8px 0 12px;font-weight:600;color:{ink}">{verdict}</div>
  <div style="font-size:13px;color:{muted}">ABOUT THE TOP CLASS</div>
  <div style="font-size:20px;font-weight:700;margin:2px 0;color:{ink}">{name}</div>
  <div style="color:{ink}">{text}</div>
  <div style="margin-top:6px;color:{muted}">On the test set our model finds
       <span style="font-weight:700;color:{ink}">{TEST_HITS[name]} of 100</span> {name} cases.</div>
  {low}{warn}
</div>"""


def analyse(image):
    if image is None:
        return None, None, ""
    top3 = classify(image)
    return what_the_model_sees(image), top3, result_card(top3, network_votes(image))




def examples_in(folder):
    """Example files and their true class (taken from the file name)."""
    files = sorted(glob.glob(os.path.join(HERE, "examples", folder, "*")))
    names = []
    for f in files:
        base = os.path.basename(f).lower()
        names.append(next((c for c in CLASSES if c.replace(" ", "_") in base), ""))
    return files, names


SHORT = {"actinic keratoses": "actinic ker.", "basal cell carcinoma": "basal cell ca.",
         "benign keratosis": "benign ker.", "melanocytic nevi": "mole", "vascular lesions": "vascular"}

EXAMPLE_ROWS = [
    ("1_sharp", "SHARP photos (original 600 x 450)"),
    ("2_slightly_blurred", "SLIGHTLY BLURRED (64 x 64)"),
    ("3_very_blurred", "VERY BLURRED (28 x 28 = DermaMNIST quality)"),
]

with gr.Blocks(title="Tensor Titans - skin lesion classifier") as demo:
    gr.Markdown(
        "# Skin lesion classifier (DermaMNIST challenge)\n"
        "**Tensor Titans:** Guna Prakash Kethineni, Varun Kumar Sama, Panyala Shesha Shayana Reddy  \n"
        "Upload a photo of a skin lesion. The model sorts it into one of 7 classes. "
        "**Student project, not a medical device!**")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="skin lesion image", height=300)
            button = gr.Button("Analyse", variant="primary")
            for folder, title in EXAMPLE_ROWS:
                files, names = examples_in(folder)
                if not files:
                    continue
                short = [SHORT.get(n, n) for n in names]
                gr.Examples(examples=files, inputs=image, label=f"{title}. True class, left to right: " + ", ".join(short))
            gr.Markdown("<small>All examples are validation images (never training or test images). "
                        "Photos: HAM10000, Tschandl et al., Sci. Data 2018, CC BY-NC 4.0.</small>")
        with gr.Column(scale=1):
            seen = gr.Image(label="what the model sees (28 x 28 pixels)", height=180, interactive=False)
            label = gr.Label(num_top_classes=3, label="prediction (top 3)")
        with gr.Column(scale=1):
            card = gr.HTML()
    button.click(analyse, inputs=image, outputs=[seen, label, card])
    image.change(analyse, inputs=image, outputs=[seen, label, card])

if __name__ == "__main__":
    demo.launch(share=True)   # share=True: public link (needed in Colab); on Hugging Face Spaces it is ignored
