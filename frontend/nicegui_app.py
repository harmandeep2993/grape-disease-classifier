from nicegui import ui
import requests
import base64
import io
import os
import time
from dotenv import load_dotenv

load_dotenv()

API_URL  = os.getenv("API_URL", "http://127.0.0.1:8000")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_BG = {
    "Black rot"                          : "#FCEBEB",
    "Esca (Black Measles)"               : "#FAEEDA",
    "Leaf blight (Isariopsis Leaf Spot)" : "#E6F1FB",
    "healthy"                            : "#EAF3DE",
}

CLASS_TEXT = {
    "Black rot"                          : "#A32D2D",
    "Esca (Black Measles)"               : "#854F0B",
    "Leaf blight (Isariopsis Leaf Spot)" : "#185FA5",
    "healthy"                            : "#3B6D11",
}

CLASS_DOT = {
    "Black rot"   : "#A32D2D",
    "Esca"        : "#854F0B",
    "Leaf blight" : "#185FA5",
    "Healthy"     : "#3B6D11",
}

SAMPLE_IMAGES = {
    "Black rot"   : os.path.join(BASE_DIR, "test_images", "Black_rot.jpg"),
    "Esca"        : os.path.join(BASE_DIR, "test_images", "Esca.jpg"),
    "Leaf blight" : os.path.join(BASE_DIR, "test_images", "Leaf_blight.jpg"),
    "Healthy"     : os.path.join(BASE_DIR, "test_images", "Healthy.jpg"),
}

uploaded = {"bytes": None, "name": None}


def call_api(image_bytes, filename):
    for attempt in range(3):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (filename, image_bytes, "image/jpeg")},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return {"error": str(e)}


ui.add_head_html("""
<style>
    body {
        background: #f0f0f0;
        margin: 0;
        font-family: sans-serif;
        height: 100vh;
        overflow: hidden;
    }
    .nicegui-content {
        height: 100vh;
        overflow: hidden;
    }
</style>
""")

with ui.column().style(
    "width:100%; height:95vh; padding:10px; box-sizing:border-box; gap:6px;"
):
    # Title — no card
    ui.label("🍇 Grape disease classifier").style(
        "font-size:20px; font-weight:500; text-align:center; width:100%;"
    )
    ui.label(
        "Transfer learning on PlantVillage dataset · DenseNet121 · 99.34% accuracy · 4 disease classes"
    ).style(
        "font-size:11px; color:#aaa; text-align:center; width:100%; margin-top:-4px;"
    )

    prob_labels = {}

    # Three columns
    with ui.row().style(
        "width:100%; gap:10px; align-items:stretch; "
        "height:calc(100vh - 80px); min-height:0;"
    ):

        # Col 1 — Upload + classes + model info
        with ui.card().style(
            "width:210px; flex-shrink:0; padding:12px; display:flex; "
            "flex-direction:column; gap:6px; overflow-y:auto; "
            "box-sizing:border-box; height:100%;"
        ):
            ui.label("Upload").style("font-size:12px; font-weight:500;")

            async def handle_upload(e):
                uploaded["bytes"] = await e.file.read()
                uploaded["name"]  = e.file.name
                encoded = base64.b64encode(uploaded["bytes"]).decode()
                center_image.set_source(f"data:image/jpeg;base64,{encoded}")
                center_image.style(
                    "width:100%; border-radius:8px; display:block; "
                    "object-fit:contain; max-height:calc(100vh - 220px);"
                )
                right_image.style("display:none;")
                tab_row.style("display:flex; gap:4px;")
                predict_btn.visible = True
                remove_btn.visible  = True
                pred_card.style("display:none;")
                prob_section.style("display:none;")
                spot_section.style("display:none;")

            ui.upload(
                on_upload=handle_upload,
                label="Click to upload",
                max_files=1,
                max_file_size=200_000_000,
                auto_upload=True
            ).style("width:100%;")

            ui.label("Or try a sample:").style("font-size:10px; color:#888;")
            with ui.grid(columns=2).style("gap:4px; width:100%;"):
                for label, path in SAMPLE_IMAGES.items():
                    def make_handler(p=path):
                        def handler():
                            with open(p, "rb") as f:
                                uploaded["bytes"] = f.read()
                                uploaded["name"]  = os.path.basename(p)
                            encoded = base64.b64encode(uploaded["bytes"]).decode()
                            center_image.set_source(f"data:image/jpeg;base64,{encoded}")
                            center_image.style(
                                "width:100%; border-radius:8px; display:block; "
                                "object-fit:contain; max-height:calc(100vh - 220px);"
                            )
                            right_image.style("display:none;")
                            tab_row.style("display:flex; gap:4px;")
                            predict_btn.visible = True
                            remove_btn.visible  = True
                            pred_card.style("display:none;")
                            prob_section.style("display:none;")
                            spot_section.style("display:none;")
                        return handler
                    ui.button(label, on_click=make_handler()).style("font-size:10px; padding:4px;")

            ui.separator()

            with ui.row().style("gap:6px; width:100%;"):
                predict_btn = ui.button("Predict", icon="search").style(
                    "flex:1; background:#27500A; color:#C0DD97; font-size:11px;"
                )
                remove_btn = ui.button("", icon="close").style(
                    "background:#FCEBEB; color:#A32D2D; font-size:11px;"
                )

            predict_btn.visible = False
            remove_btn.visible  = False

            ui.separator()
            ui.label("Model info").style(
                "font-size:10px; color:#888; text-transform:uppercase; letter-spacing:0.05em;"
            )
            for key, val in [
                ("Architecture", "DenseNet121"),
                ("Parameters",   "7M"),
                ("Pretrained",   "ImageNet"),
                ("Dataset",      "PlantVillage"),
                ("Accuracy",     "99.34%"),
            ]:
                with ui.row().style("justify-content:space-between; width:100%;"):
                    ui.label(key).style("font-size:11px; color:#888;")
                    ui.label(val).style(
                        f"font-size:11px; font-weight:500; "
                        f"color:{'#27500A' if key == 'Accuracy' else '#333'};"
                    )

        # Col 2 — Split image view
        with ui.card().style(
            "flex:1; padding:12px; display:flex; flex-direction:column; "
            "gap:6px; min-width:0; height:100%; box-sizing:border-box;"
        ):
            with ui.row().style(
                "width:100%; align-items:center; justify-content:space-between;"
            ):
                ui.label("Image").style("font-size:12px; font-weight:500;")
                tab_row = ui.row().style("display:none; gap:4px;")
                with tab_row:
                    def show_annotated():
                        if "annotated_b64" in uploaded:
                            right_image.set_source(
                                f"data:image/png;base64,{uploaded['annotated_b64']}"
                            )
                            right_image.style(
                                "width:100%; border-radius:8px; display:block; "
                                "object-fit:contain; max-height:calc(100vh - 220px);"
                            )

                    def show_gradcam():
                        if "gradcam_b64" in uploaded:
                            right_image.set_source(
                                f"data:image/png;base64,{uploaded['gradcam_b64']}"
                            )
                            right_image.style(
                                "width:100%; border-radius:8px; display:block; "
                                "object-fit:contain; max-height:calc(100vh - 220px);"
                            )

                    ui.button("Disease spots", on_click=show_annotated).style(
                        "font-size:10px; padding:3px 8px;"
                    )
                    ui.button("Grad-CAM", on_click=show_gradcam).style(
                        "font-size:10px; padding:3px 8px;"
                    )

            # Split left and right
            with ui.row().style("width:100%; flex:1; gap:8px; min-height:0;"):

                # Left — original
                with ui.column().style(
                    "flex:1; min-width:0; align-items:center; gap:4px;"
                ):
                    ui.label("Original").style("font-size:10px; color:#888;")
                    center_image = ui.image().style(
                        "width:100%; border-radius:8px; display:none; "
                        "object-fit:contain; max-height:calc(100vh - 220px);"
                    )

                ui.element("div").style(
                    "width:1px; background:#eee; align-self:stretch;"
                )

                # Right — spots or gradcam
                with ui.column().style(
                    "flex:1; min-width:0; align-items:center; gap:4px;"
                ):
                    ui.label("Disease spots / Grad-CAM").style(
                        "font-size:10px; color:#888;"
                    )
                    right_image = ui.image().style(
                        "width:100%; border-radius:8px; display:none; "
                        "object-fit:contain; max-height:calc(100vh - 220px);"
                    )

        # Col 3 — Result
        with ui.card().style(
            "width:210px; flex-shrink:0; padding:12px; display:flex; "
            "flex-direction:column; gap:6px; overflow-y:auto; "
            "box-sizing:border-box; height:100%;"
        ):
            ui.label("Result").style("font-size:12px; font-weight:500;")

            pred_card = ui.element("div").style(
                "display:none; border-radius:8px; padding:10px; text-align:center;"
            )
            with pred_card:
                pred_tag  = ui.label("Prediction").style(
                    "font-size:10px; text-transform:uppercase; "
                    "letter-spacing:0.05em; display:block;"
                )
                pred_name = ui.label("").style(
                    "font-size:14px; font-weight:500; display:block; margin-top:3px;"
                )
                pred_conf = ui.label("").style(
                    "font-size:11px; display:block; margin-top:2px;"
                )

            prob_section = ui.element("div").style("display:none; width:100%;")
            with prob_section:
                ui.label("Probability per class").style(
                    "font-size:10px; color:#888; margin-bottom:4px; display:block;"
                )
                for cls in CLASS_TEXT:
                    short = cls.split("(")[0].strip() if "(" in cls else cls
                    with ui.element("div").style(
                        "display:flex; justify-content:space-between; padding:4px 7px; "
                        "border-radius:6px; margin-bottom:3px; background:#f5f5f5;"
                    ) as row:
                        ui.label(short).style("font-size:11px; color:#666;")
                        prob_labels[cls] = {
                            "pct": ui.label("").style(
                                "font-size:11px; font-weight:500; color:#666;"
                            ),
                            "row": row
                        }

            spot_section = ui.element("div").style("display:none; width:100%;")
            with spot_section:
                ui.separator()
                ui.label("Spot detection").style(
                    "font-size:10px; color:#888; text-transform:uppercase; "
                    "letter-spacing:0.05em; display:block; margin-bottom:4px;"
                )
                spot_val = ui.label("0").style(
                    "font-size:20px; font-weight:500; text-align:center; display:block;"
                )
                ui.label("disease spots found").style(
                    "font-size:10px; color:#888; text-align:center; display:block;"
                )

    def on_predict():
        if not uploaded["bytes"]:
            return

        result = call_api(uploaded["bytes"], uploaded["name"])
        if "error" in result:
            ui.notify(f"Error: {result['error']}", type="negative")
            return

        pred  = result["prediction"]
        conf  = result["confidence"]
        bg    = CLASS_BG.get(pred, "#f5f5f5")
        color = CLASS_TEXT.get(pred, "#333")

        uploaded["annotated_b64"] = result.get("annotated", "")
        uploaded["gradcam_b64"]   = result.get("gradcam",   "")

        # Auto show disease spots on right after prediction
        if uploaded["annotated_b64"]:
            right_image.set_source(
                f"data:image/png;base64,{uploaded['annotated_b64']}"
            )
            right_image.style(
                "width:100%; border-radius:8px; display:block; "
                "object-fit:contain; max-height:calc(100vh - 220px);"
            )

        pred_card.style(
            f"display:block; background:{bg}; border-radius:8px; "
            f"padding:10px; text-align:center; border:0.5px solid {color};"
        )
        pred_tag.style(
            f"font-size:10px; text-transform:uppercase; letter-spacing:0.05em; "
            f"display:block; color:{color};"
        )
        pred_name.set_text(pred)
        pred_name.style(
            f"font-size:14px; font-weight:500; display:block; margin-top:3px; color:{color};"
        )
        pred_conf.set_text(f"Confidence: {conf}%")
        pred_conf.style(f"font-size:11px; display:block; margin-top:2px; color:{color};")

        prob_section.style("display:block; width:100%;")
        for cls, prob in result["probabilities"].items():
            if cls in prob_labels:
                prob_labels[cls]["pct"].set_text(f"{prob}%")
                if cls == pred:
                    prob_labels[cls]["row"].style(
                        f"display:flex; justify-content:space-between; padding:4px 7px; "
                        f"border-radius:6px; margin-bottom:3px; background:{bg}; "
                        f"border:0.5px solid {color};"
                    )
                    prob_labels[cls]["pct"].style(
                        f"font-size:11px; font-weight:500; color:{color};"
                    )
                else:
                    prob_labels[cls]["row"].style(
                        "display:flex; justify-content:space-between; padding:4px 7px; "
                        "border-radius:6px; margin-bottom:3px; background:#f5f5f5;"
                    )
                    prob_labels[cls]["pct"].style(
                        "font-size:11px; font-weight:500; color:#666;"
                    )

        spot_count = result.get("spot_count", 0)
        spot_section.style("display:block; width:100%;")
        spot_val.set_text(str(spot_count))

    def on_remove():
        uploaded["bytes"] = None
        uploaded["name"]  = None
        uploaded.pop("annotated_b64", None)
        uploaded.pop("gradcam_b64",   None)
        center_image.style("display:none;")
        right_image.style("display:none;")
        tab_row.style("display:none;")
        pred_card.style("display:none;")
        prob_section.style("display:none;")
        spot_section.style("display:none;")
        for cls in prob_labels:
            prob_labels[cls]["pct"].set_text("")
            prob_labels[cls]["row"].style(
                "display:flex; justify-content:space-between; padding:4px 7px; "
                "border-radius:6px; margin-bottom:3px; background:#f5f5f5;"
            )
        predict_btn.visible = False
        remove_btn.visible  = False

    predict_btn.on_click(on_predict)
    remove_btn.on_click(on_remove)


# ui.run(title="Grape Disease Classifier", port=8080, host="0.0.0.0")
ui.run(title="Grape Disease Classifier", port=7860, host="0.0.0.0")