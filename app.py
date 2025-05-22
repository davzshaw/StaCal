from flask import Flask, render_template, request, jsonify, send_file, session
import os
import zipfile
import io
from backend import (
    generateRandomData,
    extractFirstNumericColumn,
    calculateStatistics,
    generatePlots,
)

app = Flask(__name__)
app.secret_key = "stacal"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["PLOTS_FOLDER"] = "static/plots"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PLOTS_FOLDER"], exist_ok=True)

# === Global State ===
current_data = []
is_sample = True
last_plots = ("", "")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/visualize")
def visualize():
    hist = session.get("hist_path")
    box = session.get("box_path")

    if not hist or not box:
        return "Plots not available. Please calculate statistics first.", 404

    return render_template("visualize.html", hist_path=hist, box_path=box)


@app.route("/random-data", methods=["POST"])
def random_data():
    global current_data
    try:
        req = request.get_json()
        n = int(req.get("n", 50))
        distro = req.get("distribution", "normal")

        params = req.get("params", {})
        current_data = generateRandomData(n, distro, **params)


        return jsonify({
            "message": f"{n} random values generated using {distro} distribution.",
            "success": True
        })
    except Exception as e:
        print("Error in /random-data:", e)
        return jsonify({
            "message": "Error generating random data. Fallback data will be used.",
            "success": False
        })


@app.route("/upload", methods=["POST"])
def upload():
    global current_data
    file = request.files.get("file")

    if not file:
        return jsonify({"message": "No file uploaded.", "success": False})

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        current_data = extractFirstNumericColumn(file_path)
        return jsonify({"message": "File uploaded and data extracted.", "success": True})
    except Exception as e:
        print("Error in /upload:", e)
        return jsonify({"message": "Failed to process the uploaded file.", "success": False})


@app.route("/toggle-sample", methods=["POST"])
def toggle_sample():
    global is_sample
    is_sample = bool(request.json.get("sample", True))
    return jsonify({"message": f"Sample mode set to {is_sample}.", "success": True})


@app.route("/calculate", methods=["GET"])
def calculate():
    global last_plots

    if not current_data:
        return jsonify({
            "message": "No data available for analysis.",
            "success": False
        })

    try:
        stats = calculateStatistics(current_data, is_sample)
        hist_path, box_path = generatePlots(current_data)

        if not hist_path or not box_path:
            raise Exception("Plot generation failed.")

        session["hist_path"] = os.path.basename(hist_path)
        session["box_path"] = os.path.basename(box_path)
        last_plots = (hist_path, box_path)

        return jsonify({
            "results": stats,
            "message": "Statistics calculated and plots generated.",
            "success": True
        })
    except Exception as e:
        print("Error in /calculate:", e)
        return jsonify({
            "message": "Failed to calculate statistics or generate plots.",
            "success": False
        })


@app.route("/download/<plot_type>", methods=["GET"])
def download_plot(plot_type):
    path = last_plots[0] if plot_type == "hist" else last_plots[1]
    if not path or not os.path.exists(path):
        return jsonify({"message": "Requested plot not found.", "success": False})
    return send_file(path, as_attachment=True)


@app.route("/download-plots", methods=["GET"])
def download_plots():
    hist = session.get("hist_path")
    box = session.get("box_path")

    if not hist or not box:
        return "Plots not found. Please calculate first.", 404

    try:
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, "w") as zf:
            zf.write(os.path.join(app.config["PLOTS_FOLDER"], hist), arcname=hist)
            zf.write(os.path.join(app.config["PLOTS_FOLDER"], box), arcname=box)

        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype="application/zip",
            as_attachment=True,
            download_name="plots.zip"
        )
    except Exception as e:
        print("Error in /download-plots:", e)
        return "Failed to prepare ZIP file.", 500


if __name__ == "__main__":
    app.run(debug=True, port=80)
