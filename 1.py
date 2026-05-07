import cv2
import json
import os
import threading
import queue
from collections import defaultdict, deque
from ultralytics import YOLO
import time
import subprocess
import shutil
import signal
import torch
import numpy as np

# ══════════════════════════════════════════════════════════════
#  OTIMIZAÇÕES PARA GPU
# ══════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.set_device(0)
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")
else:
    print("⚠️ CUDA não disponível, usando CPU")

# ══════════════════════════════════════════════════════════════
#  REST API
# ══════════════════════════════════════════════════════════════
try:
    from flask import Flask, jsonify
    from flask_cors import CORS
    _FLASK_OK = True
except ImportError:
    _FLASK_OK = False
    print("WARNING: flask/flask-cors not installed - API disabled")

# Phase timing (seconds)
_API_BET      = 30
_API_COUNT    = 60
_API_RESULT   = 5
_API_PAUSE    = 5
_API_TOTAL    = _API_BET + _API_COUNT + _API_RESULT + _API_PAUSE

# Web stream optimization
WEB_STREAM_FPS = 12
WEB_JPEG_QUALITY = 75
ultimo_jpeg_web = None
jpeg_lock = threading.Lock()

# HLS/static mode
STATIC_DIR = os.path.join(os.path.dirname(__file__), "web")
HLS_DIR = os.path.join(STATIC_DIR, "hls")
HLS_FPS = 10
HLS_WIDTH = 854
HLS_HEIGHT = 480
HLS_BITRATE = "1200k"
STATUS_JSON_PATH = os.path.join(STATIC_DIR, "status.json")
ffmpeg_proc = None
ultimo_hls_write = 0.0
ultimo_status_write = 0.0

STATUS_DELAY_SECONDS = 2.0
STATUS_WRITE_INTERVAL = 0.15
status_history = deque(maxlen=240)

# Markets config
_MARKETS = [
    {
        "id": "car-flow-001", "title": "Car Flow — I-77 South",
        "icon": "🚗", "category": "yolo", "sort_order": 1,
        "phase": "bet", "phase_ends_at": None,
        "sides": {
            "over":  {"label": "OVER",  "icon": "📈", "image_url": ""},
            "under": {"label": "UNDER", "icon": "📉", "image_url": ""},
        },
        "result": None, "payout": 1.9, "rake_pct": 0.10,
    },
    {
        "id": "election-br-2026", "title": "Brazil Election 2026",
        "icon": "🗳️", "category": "prediction", "sort_order": 2,
        "phase": "bet", "phase_ends_at": 1759536000,
        "sides": {
            "a": {"label": "Lula", "icon": "🔴", "image_url": ""},
            "b": {"label": "Flavio Bolsonaro", "icon": "🟡", "image_url": ""},
        },
        "result": None, "payout": 1.9, "rake_pct": 0.10,
    },
]

def _api_phase(elapsed):
    d = elapsed % _API_TOTAL
    if d < _API_BET: return "bet"
    if d < _API_BET + _API_COUNT: return "counting"
    if d < _API_BET + _API_COUNT + _API_RESULT: return "result"
    return "pause"

def _api_ends_in(elapsed, phase):
    d = elapsed % _API_TOTAL
    b = {"bet": _API_BET, "counting": _API_BET+_API_COUNT,
         "result": _API_BET+_API_COUNT+_API_RESULT, "pause": _API_TOTAL}
    return round(b[phase] - d, 1)

def _api_status():
    now = time.time()
    elapsed = now - tempo_rodada_inicio
    phase = _api_phase(elapsed)

    line_value = meta_rodada_atual
    if line_value is not None:
        try:
            line_int = int(line_value)
        except Exception:
            line_int = line_value
        over_min = line_int + 1 if isinstance(line_int, int) else None
        over_label = f"OVER {over_min}+" if over_min is not None else "OVER"
        under_label = f"UNDER ≤ {line_int}"
        line_label = f"Line {line_int}: OVER {over_min}+ / UNDER ≤ {line_int}" if over_min is not None else f"Line {line_int}"
        question = f"Will more than {line_int} cars pass in 1m30s?"
    else:
        line_int = None
        over_min = None
        over_label = "OVER"
        under_label = "UNDER"
        line_label = "Calculating adaptive line..."
        question = "Calculating adaptive car-flow line..."

    result = None
    if phase in ("result", "pause") and resultado_dados:
        d = resultado_dados
        result = {
            "winner": "OVER" if d.get("passou") else "UNDER",
            "count": d.get("contagem"),
            "line": d.get("meta"),
            "diff": (d.get("contagem", 0) - d.get("meta", 0)) if d.get("meta") else None,
            "rule": "OVER wins only when count > line; tie stays UNDER",
        }
    history = []
    for i, r in enumerate(historico_rodadas[-5:]):
        history.append({
            "round": len(historico_rodadas) - (4 - i),
            "winner": "OVER" if r.get("passou") else "UNDER",
            "count": r.get("contagem"),
            "line": r.get("meta"),
        })
    return {
        "market_id": "car-flow-001",
        "title": "Car Flow — Adaptive Highway",
        "question": question,
        "icon": "🚗",
        "phase": phase,
        "phase_ends_in": _api_ends_in(elapsed, phase),
        "round": len(historico_rodadas) + 1,
        "line": {
            "value": line_value,
            "over_min": over_min,
            "under_max": line_int,
            "label": line_label,
            "rule": "OVER if count > line; UNDER if count <= line",
            "x1": linha["x1"], "y1": linha["y1"],
            "x2": linha["x2"], "y2": linha["y2"],
            "srcW": LARGURA, "srcH": ALTURA,
        },
        "count": {"current": total_contado, "visible": phase == "counting"},
        "sides": {
            "over": {"label": over_label, "icon": "📈", "total_matched": 0, "total_waiting": 0},
            "under": {"label": under_label, "icon": "📉", "total_matched": 0, "total_waiting": 0},
        },
        "meta_debug": ultimo_debug_meta if isinstance(ultimo_debug_meta, dict) else {},
        "result": result,
        "history": history,
        "camera": {"stream_url": URL_CAMERA, "fps": round(fps_calculado, 1), "online": True},
        "payout": 1.9,
        "rake_pct": 0.10,
        "server_time": now,
    }

def _api_thread():
    if not _FLASK_OK:
        return
    app = Flask(__name__)
    CORS(app)

    @app.after_request
    def add_headers(response):
        response.headers['ngrok-skip-browser-warning'] = '1'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Content-Security-Policy'] = "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:; script-src * 'unsafe-inline' 'unsafe-eval'; style-src * 'unsafe-inline';"
        return response

    @app.route("/status")
    def r_status():
        resp = jsonify(_api_status())
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.route("/markets")
    def r_markets():
        return jsonify({"markets": _MARKETS})

    @app.route("/market/<mid>/status")
    def r_market(mid):
        if mid == "car-flow-001":
            return jsonify(_api_status())
        m = next((x for x in _MARKETS if x["id"] == mid), None)
        if not m:
            return jsonify({"error": "market not found"}), 404
        return jsonify({**m, "server_time": time.time()})

    def _send_predictmarket_html():
        from flask import send_file
        nomes_html = ("predictmarket.html",)
        pastas = (os.path.dirname(__file__), STATIC_DIR)
        for pasta in pastas:
            for name in nomes_html:
                html_path = os.path.join(pasta, name)
                if os.path.exists(html_path):
                    print(f"🌐 Servindo HTML: {html_path}")
                    return send_file(html_path)
        return "<h1>HTML não encontrado</h1><p>predictmarket.html não está na pasta.</p>", 404

    @app.route("/")
    def r_index():
        return _send_predictmarket_html()

    @app.route("/status.json")
    def r_status_json():
        from flask import send_file
        if os.path.exists(STATUS_JSON_PATH):
            resp = send_file(STATUS_JSON_PATH, mimetype="application/json")
            resp.headers["Cache-Control"] = "no-cache"
            return resp
        return jsonify(_api_status())

    @app.route("/hls/<path:filename>")
    def r_hls(filename):
        from flask import send_from_directory
        resp = send_from_directory(HLS_DIR, filename)
        if filename.endswith(".m3u8"):
            resp.headers["Cache-Control"] = "no-cache"
            resp.headers["Content-Type"] = "application/vnd.apple.mpegurl"
        elif filename.endswith(".ts"):
            resp.headers["Cache-Control"] = "public, max-age=30"
            resp.headers["Content-Type"] = "video/mp2t"
        return resp

    @app.route("/history")
    def r_history():
        rows = []
        for i, r in enumerate(historico_rodadas[-20:]):
            rows.append({
                "round": i + 1,
                "line": r.get("meta"),
                "count": r.get("contagem"),
                "winner": "OVER" if r.get("passou") else "UNDER",
                "diff": (r.get("contagem", 0) - r.get("meta", 0)) if r.get("meta") else None,
            })
        return jsonify({"rounds": rows, "total": len(rows)})

    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print("🌐 API rodando em http://localhost:8080")
    print("🌐 Externo: use Cloudflare Tunnel")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True, use_reloader=False)

# ══════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════

URL_CAMERA = "https://video.dot.state.mn.us/public/C9181.stream/chunklist_w1884308494.m3u8"

LARGURA = 854
ALTURA = 480

CONF_DETECCAO = 0.01
FRAME_SKIP = 1
IOU_DETECCAO = 0.25
IMGSZ = 640

AREA_MINIMA = 300
AREA_MAXIMA = 500000
PROPORCAO_MINIMA = 0.15
PROPORCAO_MAXIMA = 12.0

USAR_ZONA = True

INTERVALO_RESET = 90
MAX_HISTORICO = 5
META_INICIAL = None
META_MARGEM_PCT = 0.15
MOSTRAR_RESULTADO = True

# ══════════════════════════════════════════════════════════════

ARQUIVO_LINHA = "linha_config.json"
ARQUIVO_ZONA = "zona_config.json"
CLASSES_VEICULOS = [2, 3, 5, 7]
POSICAO_VERTICAL = 1.0
POSICAO_HORIZONTAL = 0.5

# Carrega modelo
print("Carregando modelo YOLO...")
model = YOLO("yolov8n.pt")

USE_GPU = torch.cuda.is_available()
YOLO_DEVICE = 0 if USE_GPU else "cpu"

if USE_GPU:
    try:
        model.to("cuda")
        dummy = torch.randn(1, 3, IMGSZ, IMGSZ).to("cuda")
        for _ in range(10):
            _ = model(dummy, verbose=False)
        print("✅ GPU aquecida e pronta!")
    except Exception as e:
        print(f"⚠️ Erro no warmup: {e}")

model.fuse()
print("✅ Modelo carregado!")

# Linha de contagem
linha = {"x1": 320, "y1": 180, "x2": 320, "y2": 360}

def carregar_linha():
    if os.path.exists(ARQUIVO_LINHA):
        try:
            with open(ARQUIVO_LINHA, "r", encoding="utf-8") as f:
                linha.update(json.load(f))
            print("✅ Linha carregada")
        except Exception as e:
            print(f"❌ Erro ao carregar linha: {e}")

carregar_linha()

# Zona
zona = {"cx": 210, "cy": 265, "r": 105}

def carregar_zona():
    global zona
    if os.path.exists(ARQUIVO_ZONA):
        try:
            with open(ARQUIVO_ZONA, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "cx" in data:
                    zona.update(data)
                else:
                    cx = (data.get("x1",50) + data.get("x2",590)) // 2
                    cy = (data.get("y1",50) + data.get("y2",310)) // 2
                    r = min(data.get("x2",590)-data.get("x1",50), data.get("y2",310)-data.get("y1",50)) // 2
                    zona.update({"cx": cx, "cy": cy, "r": r})
            print("✅ Zona carregada")
        except Exception as e:
            print(f"❌ Erro ao carregar zona: {e}")

carregar_zona()

# Detecta FPS
def detectar_fps(url):
    print("🔍 Detectando FPS...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and 5 < fps < 120:
        print(f"📷 FPS: {fps:.1f}")
        return fps
    return 25

# Thread de leitura
frame_queue = queue.Queue(maxsize=4)
stop_event = threading.Event()

def leitor(url, intervalo):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    falhas = 0
    ultimo_t = 0.0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            falhas += 1
            if falhas >= 60:
                print("🔄 Reconectando...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                falhas = 0
            time.sleep(0.01)
            continue
        falhas = 0
        agora = time.time()
        espera = intervalo - (agora - ultimo_t)
        if espera > 0:
            time.sleep(espera)
        ultimo_t = time.time()
        frame = cv2.resize(frame, (LARGURA, ALTURA))
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
    cap.release()

# Estado global
total_contado = 0
ids_contados = set()
posicoes_anteriores = {}
historico_posicoes = defaultdict(int)
historico_rodadas = []
tempo_rodada_inicio = time.time()
meta_rodada_atual = None
mostrar_resultado = False
resultado_timer = 0
resultado_dados = {}
PAUSA_ENTRE_RODADAS = 5
rodada_em_pausa = False
pausa_ate = 0
proxima_meta_pendente = None

# Geometria
def distancia_ponto(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5

def lado_da_linha(px, py):
    x1,y1,x2,y2 = linha["x1"],linha["y1"],linha["x2"],linha["y2"]
    return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)

def distancia_ponto_linha(px, py):
    x1,y1,x2,y2 = linha["x1"],linha["y1"],linha["x2"],linha["y2"]
    dx = x2 - x1
    dy = y2 - y1
    comprimento = (dx**2 + dy**2) ** 0.5
    if comprimento == 0:
        return ((px-x1)**2 + (py-y1)**2) ** 0.5
    return abs(dx*(y1-py) - dy*(x1-px)) / comprimento

def ponto_cruzou_linha(px_a, py_a, px_n, py_n):
    lado_a = lado_da_linha(px_a, py_a)
    lado_n = lado_da_linha(px_n, py_n)
    if lado_a * lado_n < 0:
        return True
    BUFFER_PX = 12
    if distancia_ponto_linha(px_n, py_n) < BUFFER_PX and abs(lado_a) > BUFFER_PX:
        return True
    for t in [0.2, 0.4, 0.6, 0.8]:
        xi = px_a + (px_n - px_a) * t
        yi = py_a + (py_n - py_a) * t
        if lado_a * lado_da_linha(xi, yi) < 0:
            return True
    return False

def obter_ponto_deteccao(x1, y1, x2, y2):
    return (int(x1 + (x2-x1)*POSICAO_HORIZONTAL),
            int(y1 + (y2-y1)*POSICAO_VERTICAL))

import random
import statistics

ALVO_OVER = 0.50
JANELA_META = 12
FORCA_CORRECAO_50_50 = 0.26
RUIDO_MIN = -0.025
RUIDO_MAX = 0.025
LIMITE_PULO_META_BAIXO = 0.78
LIMITE_PULO_META_ALTO = 1.18
ANTI_SEQUENCIA_ATIVO = True
MAX_SEQUENCIA_DESEJADA = 2
FORCA_ANTI_SEQUENCIA = 0.14
FORCA_ANTI_SEQUENCIA_MAX = 0.24
RANDOM_TENDENCIA_ATIVO = True
CHANCE_TENDENCIA = 0.25
FORCA_TENDENCIA_MIN = 0.03
FORCA_TENDENCIA_MAX = 0.07
CHANCE_TENDENCIA_CONTRA_SEQUENCIA = 0.65
META_TETO_ADAPTATIVO_ATIVO = True
META_TETO_JANELA = 14
META_TETO_MINIMO = 18
META_TETO_FOLGA_CARROS = 3
META_TETO_MULTIPLICADOR_MEDIANA = 1.18
META_TETO_MULTIPLICADOR_P80 = 1.10
META_ALTA_REFERENCIA = 40
META_ALTA_MIN_AMOSTRAS = 2
META_ALTA_TAXA_OVER_BAIXA = 0.30
META_TETO_FIXO_SEGURANCA = None

ultimo_debug_meta = {}

def clamp(v, mn, mx):
    return max(mn, min(mx, v))

def obter_sequencia_atual():
    rodadas_com_meta = [r for r in historico_rodadas if r.get("meta") is not None]
    if not rodadas_com_meta:
        return None, 0
    ultimo_lado = "OVER" if rodadas_com_meta[-1].get("passou") else "UNDER"
    tamanho = 0
    for r in reversed(rodadas_com_meta):
        lado = "OVER" if r.get("passou") else "UNDER"
        if lado == ultimo_lado:
            tamanho += 1
        else:
            break
    return ultimo_lado, tamanho

def escolher_tendencia_aleatoria(lado_seq=None, tam_seq=0):
    if not RANDOM_TENDENCIA_ATIVO:
        return None
    if random.random() > CHANCE_TENDENCIA:
        return None
    if lado_seq is not None and tam_seq >= MAX_SEQUENCIA_DESEJADA:
        if random.random() < CHANCE_TENDENCIA_CONTRA_SEQUENCIA:
            return "UNDER" if lado_seq == "OVER" else "OVER"
    return random.choice(["OVER", "UNDER"])

def percentil_simples(valores, pct):
    if not valores:
        return None
    vals = sorted(float(v) for v in valores)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac

def calcular_teto_adaptativo(base, recentes):
    if not META_TETO_ADAPTATIVO_ATIVO:
        return None, None
    janela = recentes[-META_TETO_JANELA:]
    contagens = [r.get("contagem", 0) for r in janela if r.get("contagem") is not None]
    if len(contagens) < 4:
        return None, None
    mediana_real = statistics.median(contagens)
    p80_real = percentil_simples(contagens, 0.80) or mediana_real
    teto_natural = max(
        META_TETO_MINIMO,
        mediana_real * META_TETO_MULTIPLICADOR_MEDIANA,
        p80_real * META_TETO_MULTIPLICADOR_P80,
        mediana_real + META_TETO_FOLGA_CARROS,
    )
    motivo = "fluxo"
    rodadas_com_meta = [r for r in janela if r.get("meta") is not None]
    metas_altas = [r for r in rodadas_com_meta if r.get("meta", 0) >= META_ALTA_REFERENCIA]
    if len(metas_altas) >= META_ALTA_MIN_AMOSTRAS:
        taxa_over_alta = sum(1 for r in metas_altas if r.get("passou")) / len(metas_altas)
        contagens_altas = [r.get("contagem", 0) for r in metas_altas]
        mediana_altas = statistics.median(contagens_altas) if contagens_altas else mediana_real
        if taxa_over_alta <= META_ALTA_TAXA_OVER_BAIXA:
            teto_teste_alto = max(META_TETO_MINIMO, mediana_altas + META_TETO_FOLGA_CARROS)
            teto_natural = min(teto_natural, teto_teste_alto)
            motivo = f"alta_under({taxa_over_alta:.2f})"
    if META_TETO_FIXO_SEGURANCA is not None:
        teto_natural = min(teto_natural, META_TETO_FIXO_SEGURANCA)
        motivo += "+fixo"
    return max(META_TETO_MINIMO, int(round(teto_natural))), motivo

def gerar_meta_rodada():
    global ultimo_debug_meta
    ultimo_debug_meta = {
        "base": None,
        "taxa_over": None,
        "sequencia_lado": None,
        "sequencia_tamanho": 0,
        "tendencia": None,
        "meta_antes": None,
        "teto_adaptativo": None,
        "teto_motivo": None,
        "meta_final": None,
    }
    if META_INICIAL is not None:
        meta_fixa = max(1, int(META_INICIAL))
        ultimo_debug_meta["meta_final"] = meta_fixa
        return meta_fixa
    if len(historico_rodadas) < 4:
        return None
    recentes = historico_rodadas[-JANELA_META:]
    contagens_ajustadas = []
    for r in recentes:
        contagem = r.get("contagem", 0)
        meta = r.get("meta")
        passou = r.get("passou", False)
        tempo_decorrido = r.get("tempo_decorrido", INTERVALO_RESET)
        segundos_restantes = r.get("segundos_restantes", 0)
        if meta is None:
            contagens_ajustadas.append(contagem)
            continue
        if passou and tempo_decorrido > 5:
            fluxo_por_segundo = contagem / tempo_decorrido
            contagem_equivalente = contagem + fluxo_por_segundo * segundos_restantes
        else:
            contagem_equivalente = contagem
        contagens_ajustadas.append(contagem_equivalente)
    if not contagens_ajustadas:
        return None
    base = statistics.median(contagens_ajustadas)
    ultimo_debug_meta["base"] = round(base, 2)
    rodadas_com_meta = [r for r in recentes if r.get("meta") is not None]
    if len(rodadas_com_meta) >= 3:
        overs = sum(1 for r in rodadas_com_meta if r.get("passou"))
        total = len(rodadas_com_meta)
        taxa_over = overs / total if total else ALVO_OVER
        ultimo_debug_meta["taxa_over"] = round(taxa_over, 3)
        erro = taxa_over - ALVO_OVER
        ajuste_equilibrio = 1.0 + (erro * FORCA_CORRECAO_50_50)
        over_early_strengths = [
            r.get("segundos_restantes", 0) / INTERVALO_RESET
            for r in rodadas_com_meta
            if r.get("passou") and r.get("segundos_restantes", 0) > 0
        ]
        media_forca_early = (sum(over_early_strengths) / len(over_early_strengths)) if over_early_strengths else 0
        ajuste_tempo = 1.0 + clamp(media_forca_early * 0.22, 0, 0.18)
    else:
        ajuste_equilibrio = 1.0
        ajuste_tempo = 1.0
    ruido = random.uniform(RUIDO_MIN, RUIDO_MAX)
    meta = base * ajuste_equilibrio * ajuste_tempo * (1.0 + ruido)
    lado_seq, tam_seq = obter_sequencia_atual()
    ultimo_debug_meta["sequencia_lado"] = lado_seq
    ultimo_debug_meta["sequencia_tamanho"] = tam_seq
    if ANTI_SEQUENCIA_ATIVO and lado_seq is not None and tam_seq >= MAX_SEQUENCIA_DESEJADA:
        excesso_seq = tam_seq - MAX_SEQUENCIA_DESEJADA + 1
        ajuste_seq = min(FORCA_ANTI_SEQUENCIA_MAX, FORCA_ANTI_SEQUENCIA * excesso_seq)
        if lado_seq == "OVER":
            meta *= (1.0 + ajuste_seq)
        elif lado_seq == "UNDER":
            meta *= (1.0 - ajuste_seq)
    tendencia = escolher_tendencia_aleatoria(lado_seq, tam_seq)
    ultimo_debug_meta["tendencia"] = tendencia
    if tendencia is not None:
        forca_tendencia = random.uniform(FORCA_TENDENCIA_MIN, FORCA_TENDENCIA_MAX)
        if tendencia == "OVER":
            meta *= (1.0 - forca_tendencia)
        elif tendencia == "UNDER":
            meta *= (1.0 + forca_tendencia)
    ultimo_debug_meta["meta_antes"] = round(meta, 2)
    ultima_meta = historico_rodadas[-1].get("meta")
    if ultima_meta is not None:
        limite_baixo = ultima_meta * LIMITE_PULO_META_BAIXO
        limite_alto = ultima_meta * LIMITE_PULO_META_ALTO
        meta = clamp(meta, limite_baixo, limite_alto)
    teto_adaptativo, teto_motivo = calcular_teto_adaptativo(base, recentes)
    if teto_adaptativo is not None:
        ultimo_debug_meta["teto_adaptativo"] = teto_adaptativo
        ultimo_debug_meta["teto_motivo"] = teto_motivo
        meta = min(meta, teto_adaptativo)
    meta_final = max(1, int(round(meta)))
    ultimo_debug_meta["meta_final"] = meta_final
    return meta_final

def desenhar_frame_web_limpo(frame_base):
    frame_web = frame_base.copy()
    cv2.line(frame_web, (linha["x1"], linha["y1"]), (linha["x2"], linha["y2"]), (0, 0, 0), 6)
    cv2.line(frame_web, (linha["x1"], linha["y1"]), (linha["x2"], linha["y2"]), (0, 255, 255), 3)
    label = f"CARS {total_contado}"
    x, y = 14, ALTURA - 18
    cv2.putText(frame_web, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                0.62, (0, 255, 255), 2, cv2.LINE_AA)
    return frame_web

def limpar_hls_antigo():
    os.makedirs(HLS_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    for name in os.listdir(HLS_DIR):
        if name.endswith(('.ts', '.m3u8', '.tmp')):
            try:
                os.remove(os.path.join(HLS_DIR, name))
            except OSError:
                pass

def iniciar_ffmpeg_hls():
    global ffmpeg_proc
    if shutil.which('ffmpeg') is None:
        print('❌ FFmpeg não encontrado. Instale: apt-get install ffmpeg')
        return None
    limpar_hls_antigo()
    playlist = os.path.join(HLS_DIR, 'stream.m3u8')
    segment = os.path.join(HLS_DIR, 'seg_%05d.ts')
    gop = max(1, HLS_FPS * 2)
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{HLS_WIDTH}x{HLS_HEIGHT}',
        '-r', str(HLS_FPS), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
        '-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-level', '3.0',
        '-b:v', HLS_BITRATE, '-maxrate', HLS_BITRATE, '-bufsize', '1700k',
        '-g', str(gop), '-keyint_min', str(gop), '-sc_threshold', '0',
        '-f', 'hls', '-hls_time', '1', '-hls_list_size', '5',
        '-hls_flags', 'delete_segments+append_list+omit_endlist+independent_segments',
        '-hls_segment_filename', segment, playlist,
    ]
    print('🎥 FFmpeg HLS iniciado')
    ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return ffmpeg_proc

def enviar_frame_para_hls(frame_web):
    global ffmpeg_proc, ultimo_hls_write
    agora = time.time()
    if agora - ultimo_hls_write < (1.0 / max(1, HLS_FPS)):
        return
    if ffmpeg_proc is None or ffmpeg_proc.poll() is not None:
        ffmpeg_proc = iniciar_ffmpeg_hls()
        if ffmpeg_proc is None:
            return
    frame_out = cv2.resize(frame_web, (HLS_WIDTH, HLS_HEIGHT))
    try:
        ffmpeg_proc.stdin.write(frame_out.tobytes())
        ffmpeg_proc.stdin.flush()
        ultimo_hls_write = agora
    except (BrokenPipeError, OSError):
        print('⚠️ FFmpeg reiniciando...')
        try:
            ffmpeg_proc.kill()
        except Exception:
            pass
        ffmpeg_proc = iniciar_ffmpeg_hls()

def escrever_status_json():
    global ultimo_status_write
    agora = time.time()
    try:
        status_history.append((agora, _api_status()))
        if agora - ultimo_status_write < STATUS_WRITE_INTERVAL:
            return
        alvo = agora - STATUS_DELAY_SECONDS
        data = None
        for ts, snap in reversed(status_history):
            if ts <= alvo:
                data = snap
                break
        if data is None and status_history:
            data = status_history[0][1]
        if data is None:
            return
        data = dict(data)
        os.makedirs(STATIC_DIR, exist_ok=True)
        tmp = STATUS_JSON_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, STATUS_JSON_PATH)
        ultimo_status_write = agora
    except Exception as e:
        pass

# Main
print("\n" + "="*60)
print("🚗 CONTAGEM DE VEÍCULOS")
print("="*60)
print(f"📡 Camera URL: {URL_CAMERA}")

fps_stream = detectar_fps(URL_CAMERA)
intervalo = 1.0 / max(fps_stream, 15)

t = threading.Thread(target=leitor, args=(URL_CAMERA, intervalo), daemon=True)
t.start()

threading.Thread(target=_api_thread, daemon=True).start()

print(f"✅ Configurações otimizadas:")
print(f"   Resolução: {LARGURA}x{ALTURA}")
print(f"   IMGSZ: {IMGSZ}")
print(f"   FPS destino: {fps_stream:.1f}")
print("MODO HEADLESS: API rodando, sem janela OpenCV")
print("-"*60)

frame_idx = 0
ultimo_results = None
fps_calculado = 0.0
ultimo_frame_web = None
tempo_inicio = time.time()
frames_fps = 0

while True:
    try:
        frame = frame_queue.get(timeout=1.0)
    except queue.Empty:
        continue

    frame_base_web = frame.copy()
    frame_idx += 1
    frames_fps += 1
    agora = time.time()
    elapsed = agora - tempo_inicio
    if elapsed >= 2.0:
        fps_calculado = frames_fps / elapsed
        frames_fps = 0
        tempo_inicio = agora

    if rodada_em_pausa and time.time() >= pausa_ate:
        total_contado = 0
        ids_contados.clear()
        posicoes_anteriores.clear()
        historico_posicoes.clear()
        ultimo_results = None
        tempo_rodada_inicio = time.time()
        meta_rodada_atual = proxima_meta_pendente
        proxima_meta_pendente = None
        mostrar_resultado = False
        resultado_dados = {}
        rodada_em_pausa = False
        print(f"🎯 Nova meta: {meta_rodada_atual}")

    meta_atingida = (meta_rodada_atual is not None and total_contado > meta_rodada_atual)
    tempo_esgotado = (time.time() - tempo_rodada_inicio >= INTERVALO_RESET)

    if (not rodada_em_pausa) and (meta_atingida or tempo_esgotado):
        passou = meta_atingida
        tempo_decorrido = time.time() - tempo_rodada_inicio
        segundos_restantes = max(0, INTERVALO_RESET - tempo_decorrido)
        motivo = f"OVER {total_contado}/{meta_rodada_atual}" if passou else f"UNDER {total_contado}/{meta_rodada_atual}"
        historico_rodadas.append({
            "contagem": total_contado,
            "meta": meta_rodada_atual,
            "passou": passou,
            "tempo_decorrido": tempo_decorrido,
            "segundos_restantes": segundos_restantes,
        })
        proxima_meta = gerar_meta_rodada()
        resultado_dados = {
            "passou": passou,
            "contagem": total_contado,
            "meta": meta_rodada_atual,
            "rodada": len(historico_rodadas),
            "proxima_meta": proxima_meta,
            "motivo": motivo,
            "tempo_decorrido": tempo_decorrido,
            "segundos_restantes": segundos_restantes,
        }
        mostrar_resultado = True
        resultado_timer = time.time()
        print(
            f"⏱️ Rodada #{len(historico_rodadas)}: "
            f"{total_contado}/{meta_rodada_atual} - "
            f"{'OVER' if passou else 'UNDER'} | "
            f"tempo={tempo_decorrido:.1f}s | faltou={segundos_restantes:.1f}s | "
            f"proxima_meta={proxima_meta}"
        )
        if proxima_meta is not None and ultimo_debug_meta:
            print(
                f"🧠 Meta debug: base={ultimo_debug_meta.get('base')} | "
                f"taxa_over={ultimo_debug_meta.get('taxa_over')} | "
                f"seq={ultimo_debug_meta.get('sequencia_lado')}x{ultimo_debug_meta.get('sequencia_tamanho')} | "
                f"tendencia={ultimo_debug_meta.get('tendencia')} | "
                f"teto={ultimo_debug_meta.get('teto_adaptativo')}({ultimo_debug_meta.get('teto_motivo')}) | "
                f"meta_final={ultimo_debug_meta.get('meta_final')}"
            )
        rodada_em_pausa = True
        pausa_ate = time.time() + PAUSA_ENTRE_RODADAS
        proxima_meta_pendente = proxima_meta

    if (not rodada_em_pausa) and frame_idx % FRAME_SKIP == 0:
        results = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=CLASSES_VEICULOS,
            conf=CONF_DETECCAO,
            iou=IOU_DETECCAO,
            imgsz=IMGSZ,
            half=USE_GPU,
            verbose=False,
            device=YOLO_DEVICE
        )
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            ultimo_results = results
        else:
            if frame_idx % (FRAME_SKIP * 10) == 0:
                ultimo_results = None

    if (not rodada_em_pausa) and ultimo_results and ultimo_results[0].boxes is not None and ultimo_results[0].boxes.id is not None:
        boxes = ultimo_results[0].boxes
        ids_neste_frame = set()
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            area = (x2-x1) * (y2-y1)
            if area < AREA_MINIMA or area > AREA_MAXIMA:
                continue
            prop = (x2-x1) / (y2-y1) if (y2-y1) > 0 else 0
            if prop < PROPORCAO_MINIMA or prop > PROPORCAO_MAXIMA:
                continue
            cx, cy = obter_ponto_deteccao(x1, y1, x2, y2)
            track_id = int(box.id[0])
            ids_neste_frame.add(track_id)
            if USAR_ZONA:
                dist = distancia_ponto(cx, cy, zona["cx"], zona["cy"])
                if dist > zona["r"]:
                    continue
            if track_id in posicoes_anteriores:
                ax, ay = posicoes_anteriores[track_id]
                if ponto_cruzou_linha(ax, ay, cx, cy) and track_id not in ids_contados:
                    total_contado += 1
                    ids_contados.add(track_id)
                    print(f"🚗 Veiculo #{track_id} | Total: {total_contado}")
            posicoes_anteriores[track_id] = (cx, cy)
            cv2.circle(frame_base_web, (cx, cy), 6, (0, 0, 0), -1)
            cor_ponto = (0, 80, 255) if track_id in ids_contados else (0, 255, 80)
            cv2.circle(frame_base_web, (cx, cy), 4, cor_ponto, -1)
        for tid in list(posicoes_anteriores.keys()):
            if tid not in ids_neste_frame:
                historico_posicoes[tid] += 1
                if historico_posicoes[tid] > 30:
                    posicoes_anteriores.pop(tid, None)
                    historico_posicoes.pop(tid, None)
            else:
                historico_posicoes[tid] = 0

    if mostrar_resultado:
        if time.time() - resultado_timer > 5:
            mostrar_resultado = False

    ultimo_frame_web = desenhar_frame_web_limpo(frame_base_web)
    
    with jpeg_lock:
        _, buf = cv2.imencode('.jpg', ultimo_frame_web, [cv2.IMWRITE_JPEG_QUALITY, WEB_JPEG_QUALITY])
        ultimo_jpeg_web = buf.tobytes()
    
    enviar_frame_para_hls(ultimo_frame_web)
    escrever_status_json()

# Limpeza ao sair (nunca chega aqui, mas mantido por segurança)
stop_event.set()
try:
    if ffmpeg_proc and ffmpeg_proc.stdin:
        ffmpeg_proc.stdin.close()
    if ffmpeg_proc:
        ffmpeg_proc.terminate()
except Exception:
    pass
print(f"\n{'='*60}\n📊 FINAL: {total_contado} veículos\n{'='*60}")