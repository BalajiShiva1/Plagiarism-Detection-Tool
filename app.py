from flask import Flask, render_template, request, jsonify
import difflib
import matplotlib.pyplot as plt
import networkx as nx
import re
import ast
import plotly.graph_objects as go
import itertools
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing as mp
import functools
import os

app = Flask(__name__)

def preprocess_code(code, language):
    lexer = get_lexer_by_name(language)
    tokens = lex(code, lexer)
    # Remove comments and standardize variable names
    preprocessed_code = []
    for token_type, token_value in tokens:
        if token_type not in {Token.Comment.Single, Token.Comment.Multiline}:
            preprocessed_code.append(token_value)
    return ' '.join(preprocessed_code)

def extract_control_structures(code):
    control_structures = {
        'for': len(re.findall(r'\bfor\s+\w+.*:', code)),
        'while': len(re.findall(r'\bwhile\s+.*:', code)),
        'if': len(re.findall(r'\bif\s+.*:', code)),
        'switch': len(re.findall(r'\bswitch\s+.*:', code)),
        'do-while': len(re.findall(r'\bdo\s+.*while\s*\(.*\):', code))
    }
    return control_structures

def compare_logical_structures(code1, code2):
    control1 = extract_control_structures(code1)
    control2 = extract_control_structures(code2)
    similarity = fuzz.ratio(list(control1.values()), list(control2.values()))
    return similarity / 100

def compare_codes(code1, code2, language):
    # Preprocess code
    code1 = preprocess_code(code1, language)
    code2 = preprocess_code(code2, language)
    
    # Token-based similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([code1, code2])
    token_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    # Logical structure similarity
    logical_similarity = compare_logical_structures(code1, code2)

    # Combine similarities
    final_similarity = (token_similarity + logical_similarity) / 2
    return final_similarity

def compare_multiple_codes(codes, language):
    similarity_matrix = {}
    for i, code1 in enumerate(codes):
        for j, code2 in enumerate(codes):
            if i != j:
                pair = (f'Code {i+1}', f'Code {j+1}')
                similarity_percentage = compare_codes(code1, code2, language) * 100
                similarity_matrix[pair] = similarity_percentage
    return similarity_matrix

def plot_similarity(similarity_matrix, threshold):
    G = nx.Graph()
    for (code1, code2), similarity in similarity_matrix.items():
        if similarity >= threshold:
            G.add_edge(code1, code2, weight=similarity)
    
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, width=weights, edge_color=weights, edge_cmap=plt.cm.Blues, node_size=3000, node_color='lightblue')
    plt.title(f'Similarity Network (Threshold: {threshold}%)')

    # Ensure the 'static' directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    # Save the plot
    filepath = os.path.join('static', 'similarity_network.png')
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_heatmap(similarity_matrix):
    keys = list(similarity_matrix.keys())
    values = list(similarity_matrix.values())
    matrix = [[0] * len(keys) for _ in range(len(keys))]
    for (i, code1), (j, code2) in itertools.product(enumerate(keys), repeat=2):
        if (code1, code2) in similarity_matrix:
            matrix[i][j] = similarity_matrix[(code1, code2)]
    fig = go.Figure(data=go.Heatmap(z=matrix, x=keys, y=keys, colorscale='Viridis'))
    fig.update_layout(title='Similarity Heatmap')

    # Ensure the 'static' directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    # Save the heatmap
    filepath = os.path.join('static', 'similarity_heatmap.png')
    fig.write_image(filepath)
    return filepath

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        language = request.form.get('language', 'python')
        threshold = float(request.form.get('threshold', 70))  # Default threshold 70%
        
        # Ensure the 'static' directory exists
        if not os.path.exists('static'):
            os.makedirs('static')
        
        codes = [file.read().decode('utf-8') for file in files]
        similarity_matrix = compare_multiple_codes(codes, language)
        
        try:
            network_path = plot_similarity(similarity_matrix, threshold)  # Generate similarity network
            heatmap_path = plot_heatmap(similarity_matrix)  # Generate heatmap
        except Exception as e:
            return f"An error occurred while generating the similarity plots: {e}"
        
        # Render results page with similarity data
        return render_template('results.html', 
                               similarity_matrix=similarity_matrix,
                               network_image=network_path,
                               heatmap_image=heatmap_path)

if __name__ == '__main__':
    app.run(debug=True)
