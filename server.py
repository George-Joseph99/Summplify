from flask import Flask, jsonify, request
from flask_cors import CORS
from Models import *

app = Flask(__name__)
CORS(app)

# @app.route("/members")
# def members():
#     return {"members" : ["Member1", "Member2", "Member3"]}


@app.route('/main', methods=['POST'])
def receive_text():
    data = request.get_json()
    # print(data)
    text = data['text']
    summary_or_simplify = data['summarizeOrSimplify']
    if summary_or_simplify == 1:
        extractive_or_abstractive = data['extractiveOrAbstractive']
        if extractive_or_abstractive == '1':
            output_text = Summarizer.extractive_summary(text)
            print(output_text)
        elif extractive_or_abstractive == '0':
            output_text = Summarizer.abstrctive_summary(text)
            print(output_text)
    elif summary_or_simplify == 0:
        output_text = Simplifier.simplify(text)
        print(output_text)
    return output_text

if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask

# app = Flask(__name__)

# @app.route("/members")
# def members():
#     return {"members": ["Member1", "Member2", "Member3"]}

# if __name__ == "__main__":
#     app.run(debug=True)
