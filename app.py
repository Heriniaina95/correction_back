from flask import Flask, request, jsonify
from flask_cors import CORS  # Importez CORS
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
CORS(app)  # Active CORS pour toutes les origines

# Chemin vers les modèles locaux
model_fr_en_path = './model_fr_en'
model_en_fr_path = './model_en_fr'

# Chargement des tokenizers et des modèles
tokenizer_fr_en = MarianTokenizer.from_pretrained(model_fr_en_path)
model_fr_en = MarianMTModel.from_pretrained(model_fr_en_path)

tokenizer_en_fr = MarianTokenizer.from_pretrained(model_en_fr_path)
model_en_fr = MarianMTModel.from_pretrained(model_en_fr_path)

def perform_translation(text, model, tokenizer):
    """
    Effectue une traduction avec le modèle et le tokenizer fournis.
    Retourne le texte traduit.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        raise ValueError(f"Erreur de traduction : {str(e)}")

@app.route('/correction', methods=['POST'])
def translate():
    """Point de terminaison pour la correction du texte en français."""
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "Aucun texte fourni"}), 400

    if len(text) > 512:
        return jsonify({"error": "Le texte est trop long. Limitez-vous à 512 caractères."}), 400

    try:
        # Étape 1: Traduction FR -> EN
        texte_en = perform_translation(text, model=model_fr_en, tokenizer=tokenizer_fr_en)
        print(f"Texte traduit en anglais : {texte_en}")

        # Étape 2: Traduction EN -> FR (correction)
        texte_corrige = perform_translation(texte_en, model=model_en_fr, tokenizer=tokenizer_en_fr)
        print(f"Texte corrigé en français : {texte_corrige}")

        return jsonify({"correction": texte_corrige})
    except ValueError as e:
        # Erreurs spécifiques liées à la traduction
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        # Autres erreurs inattendues
        return jsonify({"error": f"Erreur inattendue : {str(e)}"}), 500

if __name__ == "__main__":
    # Lancer le serveur Flask
    app.run(host="0.0.0.0", port=5000)
