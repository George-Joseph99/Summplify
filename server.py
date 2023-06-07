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
    summary_or_simplify = data['summarizeOrSimplify'] # 2 for translate, 1 for summary, 0 for simplify
    if summary_or_simplify == 1:
        extractive_or_abstractive = data['extractiveOrAbstractive']
        if extractive_or_abstractive == '1':
            compressed_length = data['compressedLength']
            output_text = Summarizer.extractive_summary(text, compressed_length)
            print(output_text)
        elif extractive_or_abstractive == '0':
            output_text = Summarizer.abstractive_summary(text)
            print(output_text)
    elif summary_or_simplify == 0:
        show_definitions = data['showDefinitions']
        print(show_definitions)
        output_text, details = Simplifier.simplify(text,show_definitions)
        simplified_data = []
        simplified_data.append(output_text)
        for detail in details:
            simplified_data.append(detail)
        return jsonify(simplified_data)
    elif summary_or_simplify == 2:
        output_text = Translator.translate(text)
    return output_text
if __name__ == "__main__":
    app.run(port=8000, debug=True)
