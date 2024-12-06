from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Chemin vers les modèles locaux
model_fr_en_path = './model_fr_en'
model_en_fr_path = './model_en_fr'

# Chargement des tokenizers et des modèles
tokenizer_fr_en = MarianTokenizer.from_pretrained(model_fr_en_path)
model_fr_en = MarianMTModel.from_pretrained(model_fr_en_path)

tokenizer_en_fr = MarianTokenizer.from_pretrained(model_en_fr_path)
model_en_fr = MarianMTModel.from_pretrained(model_en_fr_path)

@app.route('/correction', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text')
    
    # Traduction FR -> EN -> FR
    texte_en = translate(text, model=model_fr_en, tokenizer=tokenizer_fr_en)
    texte_corrige = translate(texte_en, model=model_en_fr, tokenizer=tokenizer_en_fr)

    return jsonify({"correction": texte_corrige})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
