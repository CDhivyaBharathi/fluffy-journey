from flask import Flask , jsonify , request
from classifier import ImgPred

app = Flask(__name__)
@app.route("/predict-alphabet" , methods = ["POST"])

def Pred_Data():
    image = request.files.get("alphabet")
    prediction = ImgPred(image)
    return jsonify({
        "Prediction":prediction
    }) , 200

if __name__ == "__main__" :
    app.run(debug = True)




