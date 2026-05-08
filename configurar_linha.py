import cv2
import json
import threading
import time
from flask import Flask, render_template_string, request, jsonify

URL_CAMERA = "https://video.dot.state.mn.us/public/C9181.stream/chunklist_w1884308494.m3u8"
LARGURA = 640
ALTURA = 360

# Capturar um frame
cap = cv2.VideoCapture(URL_CAMERA)
if not cap.isOpened():
    print("❌ Não foi possível abrir a câmera")
    exit(1)
ret, frame = cap.read()
cap.release()
if not ret:
    print("❌ Não foi possível ler um frame")
    exit(1)

frame = cv2.resize(frame, (LARGURA, ALTURA))

# Codificar frame para JPEG
_, jpeg = cv2.imencode('.jpg', frame)
frame_b64 = jpeg.tobytes().hex()  # ou usar base64 diretamente

# HTML com imagem e JavaScript para capturar cliques
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Configurar Linha de Contagem</title>
    <style>
        body { background: #222; color: #eee; font-family: monospace; text-align: center; }
        canvas { border: 2px solid yellow; cursor: crosshair; max-width: 100%; }
        .info { margin-top: 20px; }
        button { padding: 10px 20px; font-size: 16px; margin: 10px; cursor: pointer; }
        .point { display: inline-block; margin: 0 20px; }
    </style>
</head>
<body>
    <h2>⚙️ Clique em dois pontos na imagem para definir a linha</h2>
    <canvas id="canvas" width="{{ width }}" height="{{ height }}"></canvas>
    <div class="info">
        <div class="point"><span style="color:yellow">🔵 Ponto 1:</span> <span id="p1">(nenhum)</span></div>
        <div class="point"><span style="color:yellow">🔴 Ponto 2:</span> <span id="p2">(nenhum)</span></div>
        <div>
            <button id="salvar" disabled>💾 Salvar linha</button>
            <button id="limpar">🗑️ Limpar pontos</button>
        </div>
        <div id="msg"></div>
    </div>
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const width = {{ width }};
        const height = {{ height }};
        let points = [];
        let img = new Image();
        img.onload = () => {
            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(img, 0, 0, width, height);
        };
        img.src = "data:image/jpeg;base64,{{ img_b64 }}";

        function drawLine() {
            if (points.length === 2) {
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                ctx.lineTo(points[1].x, points[1].y);
                ctx.strokeStyle = '#ff0';
                ctx.lineWidth = 3;
                ctx.stroke();
                ctx.fillStyle = '#0f0';
                points.forEach((p, i) => {
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 6, 0, 2*Math.PI);
                    ctx.fillStyle = i === 0 ? '#0ff' : '#f00';
                    ctx.fill();
                });
            }
        }

        canvas.addEventListener('click', (e) => {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            let x = (e.clientX - rect.left) * scaleX;
            let y = (e.clientY - rect.top) * scaleY;
            x = Math.min(Math.max(0, x), width);
            y = Math.min(Math.max(0, y), height);
            if (points.length < 2) {
                points.push({x: Math.round(x), y: Math.round(y)});
                updateDisplay();
                drawLine();
            }
            if (points.length === 2) {
                document.getElementById('salvar').disabled = false;
            }
        });

        function updateDisplay() {
            document.getElementById('p1').innerHTML = points[0] ? `(${points[0].x}, ${points[0].y})` : '(nenhum)';
            document.getElementById('p2').innerHTML = points[1] ? `(${points[1].x}, ${points[1].y})` : '(nenhum)';
            ctx.drawImage(img, 0, 0, width, height);
            drawLine();
        }

        document.getElementById('salvar').onclick = () => {
            if (points.length !== 2) return;
            fetch('/salvar', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x1: points[0].x, y1: points[0].y, x2: points[1].x, y2: points[1].y})
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById('msg').innerHTML = '<span style="color:lime">✅ Linha salva com sucesso! Você já pode fechar esta página e reiniciar o servidor principal.</span>';
                document.getElementById('salvar').disabled = true;
            })
            .catch(err => alert('Erro ao salvar: ' + err));
        };
        document.getElementById('limpar').onclick = () => {
            points = [];
            updateDisplay();
            document.getElementById('salvar').disabled = true;
            document.getElementById('msg').innerHTML = '';
        };
    </script>
</body>
</html>
"""

app = Flask(__name__)

@app.route('/')
def index():
    import base64
    img_b64 = base64.b64encode(frame.tobytes()).decode()
    return render_template_string(html, width=LARGURA, height=ALTURA, img_b64=img_b64)

@app.route('/salvar', methods=['POST'])
def salvar():
    data = request.get_json()
    linha = {
        "x1": data['x1'],
        "y1": data['y1'],
        "x2": data['x2'],
        "y2": data['y2']
    }
    with open('linha_config.json', 'w') as f:
        json.dump(linha, f, indent=4)
    print(f"✅ Linha salva: {linha}")
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print("📡 Configurador de linha – acesse via Cloudflare Tunnel")
    print("1. Em outro terminal, inicie o túnel: ./cloudflared tunnel --url http://8081")
    print("2. Abra a URL gerada no navegador")
    print("3. Clique em dois pontos na imagem para definir a linha")
    print("4. Clique em 'Salvar linha' e depois reinicie o servidor principal")
    app.run(host='0.0.0.0', port=8081, debug=False)