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
currentData = []
isSample = True
lastPlots = ("", "")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/random-data", methods=["POST"])
def randomData():
    try:
        n = int(request.json.get("n", 50))
        distro = request.json.get("distribution", "normal")
        global currentData
        currentData = generateRandomData(n, distro)
        return jsonify({
            "message": f"{n} random values generated using {distro} distribution.",
            "success": True,
        })
    except Exception as e:
        print(e)
        return jsonify({
            "message": "Error generating random data. Using fallback.",
            "success": False,
        })

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"message": "No file uploaded.", "success": False})
    
    filePath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filePath)

    global currentData
    currentData = extractFirstNumericColumn(filePath)

    return jsonify({"message": "Data uploaded and processed.", "success": True})

@app.route("/toggle-sample", methods=["POST"])
def toggleSample():
    global isSample
    isSample = bool(request.json.get("sample", True))
    return jsonify({
        "message": f"Sample mode set to {isSample}.",
        "success": True
    })

@app.route("/calculate", methods=["GET"])
def calculate():
    global lastPlots

    if not currentData:
        return jsonify({
            "message": "No data available. Using fallback.",
            "success": False
        })

    stats = calculateStatistics(currentData, isSample)
    histPath, boxPath = generatePlots(currentData)
    lastPlots = (histPath, boxPath)

    session["hist_path"] = os.path.basename(histPath)
    session["box_path"] = os.path.basename(boxPath)

    return jsonify({
        "results": stats,
        "message": "Calculation complete.",
        "success": True
    })

@app.route("/download/<plotType>", methods=["GET"])
def downloadPlot(plotType):
    path = lastPlots[0] if plotType == "hist" else lastPlots[1]
    if not path or not os.path.exists(path):
        return jsonify({"message": "Plot not available.", "success": False})
    return send_file(path, as_attachment=True)

@app.route("/download-plots")
def downloadPlots():
    hist = session.get("hist_path")
    box = session.get("box_path")

    if not hist or not box:
        return "Plots not found", 404

    memoryFile = io.BytesIO()
    with zipfile.ZipFile(memoryFile, "w") as zf:
        zf.write(os.path.join("static/plots", hist), arcname=hist)
        zf.write(os.path.join("static/plots", box), arcname=box)

    memoryFile.seek(0)
    return send_file(
        memoryFile,
        mimetype="application/zip",
        as_attachment=True,
        download_name="plots.zip",
    )

if __name__ == "__main__":
    app.run(debug=True, port=80)
