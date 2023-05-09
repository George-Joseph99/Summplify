from flask import Flask, jsonify, request
from flask_cors import CORS
from Models import *

app = Flask(__name__)
CORS(app)

# @app.route("/members")
# def members():
#     return {"members" : ["Member1", "Member2", "Member3"]}

@app.route('/summarize')
def run_summarize():
    print('d5l henaaa')
    result = Summarizer.run_cr7()
    return result

@app.route('/main', methods=['POST'])
def receive_text():
    data = request.get_json()
    text = data['text']
    summary_or_simplify = data['summarizeOrSimplify']
    if summary_or_simplify == 1:
        output_text = Summarizer.summarize(text)
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
