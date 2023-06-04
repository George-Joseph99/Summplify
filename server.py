from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
from Models import *

app = Flask(__name__)
CORS(app)

@app.route('/main', methods=['POST'])
def receive_text():
    data = request.get_json()
    # print(data)
    text = data['text']
    summary_or_simplify = data['summarizeOrSimplify']
    if summary_or_simplify == 1:
        extractive_or_abstractive = data['extractiveOrAbstractive']
        if extractive_or_abstractive == '1':
            compressed_length = data['compressedLength']
            output_text = Summarizer.extractive_summary(text, compressed_length)
            print(output_text)
        elif extractive_or_abstractive == '0':
            output_text = Summarizer.abstrctive_summary(text)
            print(output_text)
    elif summary_or_simplify == 0:
        output_text, details = Simplifier.simplify(text)
        print(output_text)
        print(details)
        simplified_data = []
        simplified_data.append(output_text)
        for detail in details:
            simplified_data.append(detail)
        return jsonify(simplified_data)
    return output_text

if __name__ == "__main__":
    app.run(port=8000, debug=True)
