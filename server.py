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
import torch
import numpy as np

# GPU RTX 4090
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.set_device(0)
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")
else:
    print("⚠️ CPU mode")

try:
    from flask import Flask, jsonify
    from flask_cors import CORS
    _FLASK_OK = True
except ImportError:
    _FLASK_OK = False
    print("WARNING: flask/flask-cors not installed")

# ============================================================
#  CONFIG — CÂMERA e RTMP
# ============================================================
URL_CAMERA = "https://video.dot.state.mn.us/public/C9181.stream/chunklist_w1884308494.m3u8"

# Stream key atualizada
RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/msmg-cyyg-shsy-svhf-4q94"

RTMP_EXTRAS = []  # múltiplos destinos, se necessário

# Qualidade do stream (ajustada para 854x480, 20 FPS – mais leve)
STREAM_WIDTH   = 854
STREAM_HEIGHT  = 480
STREAM_FPS     = 20
STREAM_BITRATE = "1500k"   # menor bitrate para conexão estável

# Resolução interna (YOLO)
LARGURA = 640
ALTURA  = 360
IMGSZ   = 640

# Detecção
CONF_DETECCAO    = 0.08
IOU_DETECCAO     = 0.35
CLASSES_VEICULOS = [2, 3, 5, 7]
AREA_MINIMA      = 400   # reduzido para capturar veículos menores
AREA_MAXIMA      = 800000
PROPORCAO_MINIMA = 0.10
PROPORCAO_MAXIMA = 15.0

# Tracking
HISTORICO_POSICAO_FRAMES = 6
BUFFER_CRUZAMENTO_PX     = 22
FRAMES_ANTES_LIMPAR_ID   = 45

# Zona ativa, mas cobrindo a tela toda.
# Assim ela não corta nenhum veículo e também não aparece no stream.
USAR_ZONA                = True
MOSTRAR_ZONA_NO_VIDEO    = False

# Linha FINAL configurada manualmente na rodovia Minnesota C9181 em 640x360.
# A linha fica visível no vídeo para marcar onde conta.
# Não depende de linha_config.json/zona_config.json no RunPod.
linha = {"x1": 115, "y1": 234, "x2": 414, "y2": 156}

# Zona tela inteira em 640x360: centro da imagem + raio maior que a diagonal.
zona  = {"cx": 320, "cy": 180, "r": 500}

# Jogo
INTERVALO_RESET     = 90
PAUSA_ENTRE_RODADAS = 5
META_INICIAL        = None

# API
API_PORT     = 8080
STATUS_DELAY = 2.0

_API_BET    = 30
_API_COUNT  = 60
_API_RESULT = 5
_API_PAUSE  = 5
_API_TOTAL  = _API_BET + _API_COUNT + _API_RESULT + _API_PAUSE

# ============================================================
#  Carregar configs (opcional – se arquivos existirem)
# ============================================================
def carregar_configs():
    global linha, zona
    for arq, nome in [("linha_config.json","linha"), ("zona_config.json","zona")]:
        if os.path.exists(arq):
            try:
                with open(arq) as f:
                    d = json.load(f)
                if nome == "linha":
                    linha = d
                else:
                    zona = d
                print(f"✅ {nome}: {d}")
            except Exception as e:
                print(f"⚠️ {arq}: {e}")
        else:
            print(f"✅ {nome} padrão: {linha if nome=='linha' else zona}")

CARREGAR_CONFIGS_EXTERNAS = False
if CARREGAR_CONFIGS_EXTERNAS:
    carregar_configs()
else:
    print(f"✅ linha embutida: {linha}")
    print(f"✅ zona embutida: {zona}")
# ============================================================
#  Geometria
# ============================================================
def lado_da_linha(px, py):
    x1,y1,x2,y2 = linha["x1"],linha["y1"],linha["x2"],linha["y2"]
    return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)

def distancia_ponto_linha(px, py):
    x1,y1,x2,y2 = linha["x1"],linha["y1"],linha["x2"],linha["y2"]
    dx,dy = x2-x1, y2-y1
    comp = (dx**2 + dy**2)**0.5
    if comp == 0:
        return ((px-x1)**2 + (py-y1)**2)**0.5
    return abs(dx*(y1-py) - dy*(x1-px)) / comp

def dist2p(x1,y1,x2,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def cruzou_linha(historico):
    if len(historico) < 2:
        return False
    for i in range(len(historico)-1):
        ax,ay = historico[i]
        nx,ny = historico[i+1]
        la = lado_da_linha(ax,ay)
        ln = lado_da_linha(nx,ny)
        if la * ln < 0:
            return True
        if distancia_ponto_linha(nx,ny) < BUFFER_CRUZAMENTO_PX and abs(la) > BUFFER_CRUZAMENTO_PX*0.4:
            return True
        for t in [0.2,0.4,0.6,0.8]:
            xi = ax + (nx-ax)*t
            yi = ay + (ny-ay)*t
            if la * lado_da_linha(xi,yi) < 0:
                return True
    return False

def na_zona(cx,cy):
    if not USAR_ZONA:
        return True
    return ((cx - zona["cx"])**2 + (cy - zona["cy"])**2) ** 0.5 <= zona["r"]

def centro_inferior(x1,y1,x2,y2):
    return int((x1+x2)/2), int(y1 + (y2-y1)*0.9)

# ============================================================
#  Modelo YOLO
# ============================================================
print("Carregando modelo YOLO...")
MODEL_PATH = "yolov8n.pt"
if not os.path.exists(MODEL_PATH):
    print("Baixando yolov8n.pt...")
    from ultralytics.utils.downloads import safe_download
    safe_download("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt", file=MODEL_PATH)

model = YOLO(MODEL_PATH)
USE_GPU = torch.cuda.is_available()
YOLO_DEVICE = 0 if USE_GPU else "cpu"

if USE_GPU:
    model.to("cuda")
    print("🔥 Aquecendo GPU...")
    dummy = torch.randn(1,3,IMGSZ,IMGSZ).to("cuda")
    for _ in range(10):
        _ = model(dummy, verbose=False)
    print("✅ GPU pronta!")
model.fuse()
print(f"✅ Modelo: {MODEL_PATH}")

# ============================================================
#  Estado global
# ============================================================
total_contado    = 0
ids_contados     = set()
hist_pos         = defaultdict(lambda: deque(maxlen=HISTORICO_POSICAO_FRAMES))
frames_sem_ver   = defaultdict(int)
historico_rodadas= []
tempo_rodada_inicio = time.time()
meta_rodada_atual= None
resultado_dados  = {}
rodada_em_pausa  = False
pausa_ate        = 0
proxima_meta_pendente = None
ultimo_debug_meta= {}
fps_calculado    = 0.0

status_history   = deque(maxlen=500)
ultimo_status_write = 0.0

# ============================================================
#  Meta adaptativa (código original, não modifiquei)
# ============================================================
import random, statistics

ALVO_OVER=0.50; JANELA_META=12; FORCA_CORRECAO_50_50=0.26
RUIDO_MIN=-0.025; RUIDO_MAX=0.025
LIMITE_PULO_META_BAIXO=0.78; LIMITE_PULO_META_ALTO=1.18
ANTI_SEQUENCIA_ATIVO=True; MAX_SEQUENCIA_DESEJADA=2
FORCA_ANTI_SEQUENCIA=0.14; FORCA_ANTI_SEQUENCIA_MAX=0.24
RANDOM_TENDENCIA_ATIVO=True; CHANCE_TENDENCIA=0.25
FORCA_TENDENCIA_MIN=0.03; FORCA_TENDENCIA_MAX=0.07
CHANCE_TENDENCIA_CONTRA_SEQUENCIA=0.65
META_TETO_ADAPTATIVO_ATIVO=True; META_TETO_JANELA=14
META_TETO_MINIMO=18; META_TETO_FOLGA_CARROS=3
META_TETO_MULTIPLICADOR_MEDIANA=1.18; META_TETO_MULTIPLICADOR_P80=1.10
META_ALTA_REFERENCIA=40; META_ALTA_MIN_AMOSTRAS=2
META_ALTA_TAXA_OVER_BAIXA=0.30; META_TETO_FIXO_SEGURANCA=None

def clamp(v,mn,mx): return max(mn, min(mx, v))

def obter_sequencia_atual():
    rc = [r for r in historico_rodadas if r.get("meta") is not None]
    if not rc: return None,0
    last = "OVER" if rc[-1].get("passou") else "UNDER"; tam=0
    for r in reversed(rc):
        if ("OVER" if r.get("passou") else "UNDER") == last: tam+=1
        else: break
    return last, tam

def escolher_tendencia(ls=None,ts=0):
    if not RANDOM_TENDENCIA_ATIVO or random.random()>CHANCE_TENDENCIA: return None
    if ls and ts>=MAX_SEQUENCIA_DESEJADA and random.random()<CHANCE_TENDENCIA_CONTRA_SEQUENCIA:
        return "UNDER" if ls=="OVER" else "OVER"
    return random.choice(["OVER","UNDER"])

def percentil(vals,pct):
    if not vals: return None
    vs = sorted(float(v) for v in vals); pos=(len(vs)-1)*pct; lo=int(pos); hi=min(lo+1,len(vs)-1)
    return vs[lo]*(1-(pos-lo)) + vs[hi]*(pos-lo)

def calcular_teto(base,recentes):
    if not META_TETO_ADAPTATIVO_ATIVO: return None, None
    jan = recentes[-META_TETO_JANELA:]
    cont = [r.get("contagem",0) for r in jan if r.get("contagem") is not None]
    if len(cont)<4: return None, None
    med = statistics.median(cont); p80 = percentil(cont,0.80) or med
    teto = max(META_TETO_MINIMO, med*META_TETO_MULTIPLICADOR_MEDIANA, p80*META_TETO_MULTIPLICADOR_P80, med+META_TETO_FOLGA_CARROS)
    motivo = "fluxo"
    rcm = [r for r in jan if r.get("meta") is not None]
    altas = [r for r in rcm if r.get("meta",0) >= META_ALTA_REFERENCIA]
    if len(altas) >= META_ALTA_MIN_AMOSTRAS:
        tax = sum(1 for r in altas if r.get("passou")) / len(altas)
        med_altas = statistics.median([r.get("contagem",0) for r in altas])
        if tax <= META_ALTA_TAXA_OVER_BAIXA:
            teto = min(teto, max(META_TETO_MINIMO, med_altas+META_TETO_FOLGA_CARROS))
            motivo = f"alta_under({tax:.2f})"
    if META_TETO_FIXO_SEGURANCA:
        teto = min(teto, META_TETO_FIXO_SEGURANCA)
        motivo += "+fixo"
    return max(META_TETO_MINIMO, int(round(teto))), motivo

def gerar_meta():
    global ultimo_debug_meta
    ultimo_debug_meta = {"base":None, "taxa_over":None, "sequencia_lado":None,
                         "sequencia_tamanho":0, "tendencia":None, "meta_antes":None,
                         "teto_adaptativo":None, "teto_motivo":None, "meta_final":None}
    if META_INICIAL is not None:
        mf = max(1, int(META_INICIAL))
        ultimo_debug_meta["meta_final"] = mf
        return mf
    if len(historico_rodadas) < 4:
        return None
    rec = historico_rodadas[-JANELA_META:]
    cajust = []
    for r in rec:
        c = r.get("contagem",0)
        m = r.get("meta")
        p = r.get("passou",False)
        td = r.get("tempo_decorrido", INTERVALO_RESET)
        sr = r.get("segundos_restantes",0)
        if m and p and td>5:
            cajust.append(c + (c/td)*sr)
        else:
            cajust.append(c)
    if not cajust:
        return None
    base = statistics.median(cajust)
    ultimo_debug_meta["base"] = round(base,2)
    rcm = [r for r in rec if r.get("meta") is not None]
    ae = 1.0; at = 1.0
    if len(rcm) >= 3:
        ov = sum(1 for r in rcm if r.get("passou"))
        tx = ov / len(rcm)
        ultimo_debug_meta["taxa_over"] = round(tx,3)
        ae = 1.0 + (tx - ALVO_OVER) * FORCA_CORRECAO_50_50
        early = [r.get("segundos_restantes",0)/INTERVALO_RESET for r in rcm if r.get("passou") and r.get("segundos_restantes",0)>0]
        if early:
            at = 1.0 + clamp((sum(early)/len(early)) * 0.22, 0, 0.18)
    meta = base * ae * at * (1 + random.uniform(RUIDO_MIN, RUIDO_MAX))
    ls, ts = obter_sequencia_atual()
    ultimo_debug_meta["sequencia_lado"] = ls
    ultimo_debug_meta["sequencia_tamanho"] = ts
    if ANTI_SEQUENCIA_ATIVO and ls and ts >= MAX_SEQUENCIA_DESEJADA:
        aj = min(FORCA_ANTI_SEQUENCIA_MAX, FORCA_ANTI_SEQUENCIA * (ts - MAX_SEQUENCIA_DESEJADA + 1))
        meta *= (1+aj) if ls == "OVER" else (1-aj)
    tend = escolher_tendencia(ls, ts)
    ultimo_debug_meta["tendencia"] = tend
    if tend:
        ft = random.uniform(FORCA_TENDENCIA_MIN, FORCA_TENDENCIA_MAX)
        meta *= (1-ft) if tend == "OVER" else (1+ft)
    ultimo_debug_meta["meta_antes"] = round(meta,2)
    ultima = historico_rodadas[-1].get("meta")
    if ultima:
        meta = clamp(meta, ultima*LIMITE_PULO_META_BAIXO, ultima*LIMITE_PULO_META_ALTO)
    teto, motivo = calcular_teto(base, rec)
    if teto:
        ultimo_debug_meta["teto_adaptativo"] = teto
        ultimo_debug_meta["teto_motivo"] = motivo
        meta = min(meta, teto)
    mf = max(1, int(round(meta)))
    ultimo_debug_meta["meta_final"] = mf
    return mf

# ============================================================
#  API Flask
# ============================================================
def _api_phase(elapsed):
    d = elapsed % _API_TOTAL
    if d < _API_BET: return "bet"
    if d < _API_BET + _API_COUNT: return "counting"
    if d < _API_BET + _API_COUNT + _API_RESULT: return "result"
    return "pause"

def _api_ends_in(elapsed,phase):
    d = elapsed % _API_TOTAL
    b = {"bet": _API_BET, "counting": _API_BET+_API_COUNT,
         "result": _API_BET+_API_COUNT+_API_RESULT, "pause": _API_TOTAL}
    return round(b[phase] - d, 1)

def _api_status():
    now = time.time()
    elapsed = now - tempo_rodada_inicio
    phase = _api_phase(elapsed)
    lv = meta_rodada_atual
    if lv is not None:
        li = int(lv)
        ol = f"OVER {li+1}+"
        ul = f"UNDER ≤ {li}"
        ll = f"Line {li}: OVER {li+1}+ / UNDER ≤ {li}"
        q = f"Will more than {li} cars pass in 1m30s?"
    else:
        li = None
        ol = "OVER"
        ul = "UNDER"
        ll = "Calculating adaptive line..."
        q = "Calculating adaptive car-flow line..."
    result = None
    if phase in ("result","pause") and resultado_dados:
        d = resultado_dados
        result = {"winner": "OVER" if d.get("passou") else "UNDER",
                  "count": d.get("contagem"), "line": d.get("meta")}
    history = []
    for i,r in enumerate(historico_rodadas[-5:]):
        history.append({
            "round": len(historico_rodadas) - (min(4,len(historico_rodadas)-1)-i),
            "winner": "OVER" if r.get("passou") else "UNDER",
            "count": r.get("contagem"),
            "line": r.get("meta")
        })
    return {
        "market_id": "car-flow-001",
        "title": "Car Flow — Kingvale Highway",
        "question": q,
        "icon": "🚗",
        "phase": phase,
        "phase_ends_in": _api_ends_in(elapsed, phase),
        "round": len(historico_rodadas) + 1,
        "line": {
            "value": lv,
            "over_min": li+1 if li else None,
            "under_max": li,
            "label": ll,
            "rule": "OVER if count > line; UNDER if count <= line",
            "x1": linha["x1"], "y1": linha["y1"],
            "x2": linha["x2"], "y2": linha["y2"],
            "srcW": LARGURA, "srcH": ALTURA,
        },
        "count": {"current": total_contado, "visible": True},
        "sides": {
            "over": {"label": ol, "icon": "📈", "total_matched": 0, "total_waiting": 0},
            "under": {"label": ul, "icon": "📉", "total_matched": 0, "total_waiting": 0}
        },
        "meta_debug": ultimo_debug_meta if isinstance(ultimo_debug_meta, dict) else {},
        "result": result,
        "history": history,
        "camera": {"fps": round(fps_calculado, 1), "online": True},
        "payout": 1.9,
        "rake_pct": 0.10,
        "server_time": now,
    }

def criar_flask_app():
    app = Flask(__name__)
    CORS(app)

    @app.after_request
    def add_cors(r):
        r.headers['Access-Control-Allow-Origin'] = '*'
        r.headers['Access-Control-Allow-Headers'] = '*'
        r.headers['ngrok-skip-browser-warning'] = '1'
        r.headers['Content-Security-Policy'] = "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:;"
        return r

    @app.route("/status")
    def status():
        resp = jsonify(_api_status())
        resp.headers["Cache-Control"] = "no-cache,no-store"
        return resp

    @app.route("/status.json")
    def status_json():
        resp = jsonify(_api_status())
        resp.headers["Cache-Control"] = "no-cache,no-store"
        return resp

    @app.route("/history")
    def history():
        rows = [{"round": i+1,
                 "line": r.get("meta"),
                 "count": r.get("contagem"),
                 "winner": "OVER" if r.get("passou") else "UNDER"}
                for i,r in enumerate(historico_rodadas[-20:])]
        return jsonify({"rounds": rows, "total": len(rows)})

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "cars": total_contado, "fps": round(fps_calculado,1)})

    @app.route("/")
    def index():
        from flask import send_file
        for folder in (os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "web")):
            html_path = os.path.join(folder, "predictmarket.html")
            if os.path.exists(html_path):
                return send_file(html_path)
        return "<h1>PredictMarket API</h1><p>Use /status for data</p>", 200

    return app

app = criar_flask_app()

def _api_thread():
    if not _FLASK_OK:
        return
    # Não iniciamos o Flask aqui porque o Gunicorn ou o script principal já vai rodar.
    # Mantemos apenas para compatibilidade.
    pass

# ============================================================
#  RTMP STREAMING (FFmpeg)
# ============================================================
rtmp_proc = None
rtmp_lock = threading.Lock()

def iniciar_rtmp():
    global rtmp_proc
    if not shutil.which('ffmpeg'):
        print('❌ FFmpeg não encontrado. Instale: apt-get install ffmpeg')
        return None

    gop = STREAM_FPS * 2
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        '-re',                                 # respeita taxa de frames
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{STREAM_WIDTH}x{STREAM_HEIGHT}',
        '-r', str(STREAM_FPS), '-i', '-',
        '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
        '-c:v', 'libx264', '-preset', 'fast', '-tune', 'zerolatency',
        '-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-level', '4.0',
        '-b:v', STREAM_BITRATE, '-maxrate', STREAM_BITRATE, '-bufsize', '8000k',  # buffer maior
        '-g', str(gop), '-keyint_min', str(gop), '-sc_threshold', '0',
        '-c:a', 'aac', '-b:a', '128k', '-ar', '44100',
        '-f', 'flv', RTMP_URL,
    ]

    if RTMP_EXTRAS:
        outputs = '|'.join([RTMP_URL] + RTMP_EXTRAS)
        cmd = cmd[:-2] + ['-f', 'tee', f'[f=flv]{outputs}']

    print(f'📡 FFmpeg RTMP iniciado → {RTMP_URL}')
    rtmp_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return rtmp_proc

def enviar_rtmp(frame):
    global rtmp_proc
    with rtmp_lock:
        if rtmp_proc is None or rtmp_proc.poll() is not None:
            print("⚠️ FFmpeg RTMP caiu, reiniciando...")
            try:
                if rtmp_proc:
                    rtmp_proc.kill()
            except:
                pass
            rtmp_proc = iniciar_rtmp()
            if rtmp_proc is None:
                return
        try:
            # Redimensiona para o tamanho de streaming
            out_frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
            rtmp_proc.stdin.write(out_frame.tobytes())
            rtmp_proc.stdin.flush()
        except (BrokenPipeError, OSError):
            print("⚠️ Pipe RTMP quebrado, reiniciando...")
            try:
                rtmp_proc.kill()
            except:
                pass
            rtmp_proc = iniciar_rtmp()

def desenhar_frame(frame_base):
    f = frame_base.copy()
    # Zona continua ativa para a contagem, mas fica invisível no vídeo/YouTube.
    if USAR_ZONA and MOSTRAR_ZONA_NO_VIDEO:
        cv2.circle(f, (zona["cx"], zona["cy"]), zona["r"], (0, 0, 0), 3)
        cv2.circle(f, (zona["cx"], zona["cy"]), zona["r"], (255, 120, 0), 2)
    # Linha de contagem configurada
    cv2.line(f, (linha["x1"],linha["y1"]), (linha["x2"],linha["y2"]), (0,0,0), 6)
    cv2.line(f, (linha["x1"],linha["y1"]), (linha["x2"],linha["y2"]), (0,255,255), 3)
    # Contador de carros
    label = f"CARS {total_contado}"
    cv2.putText(f, label, (15, ALTURA-18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(f, label, (14, ALTURA-19), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if meta_rodada_atual:
        meta_txt = f"LINE {meta_rodada_atual}"
        cv2.putText(f, meta_txt, (15, ALTURA-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(f, meta_txt, (14, ALTURA-46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 1, cv2.LINE_AA)
    return f

# ============================================================
#  Leitura de frames (câmera)
# ============================================================
frame_queue = queue.Queue(maxsize=6)
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
                print("🔄 Reconectando câmera...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                falhas = 0
            time.sleep(0.01)
            continue
        falhas = 0
        now = time.time()
        espera = intervalo - (now - ultimo_t)
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

def detectar_fps(url):
    print("🔍 Detectando FPS...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and 5 < fps < 120:
        print(f"📷 FPS câmera: {fps:.1f}")
        return fps
    print("📷 FPS padrão: 30")
    return 30

# ============================================================
#  MAIN
# ============================================================
print("\n" + "="*60)
print("🚗 PREDICTMARKET — CAR FLOW COUNTER")
print("="*60)
print(f"📡 Câmera: {URL_CAMERA}")
print(f"📺 RTMP:   {RTMP_URL}")
print(f"🤖 Modelo: {MODEL_PATH} | IMGSZ: {IMGSZ}")
print(f"📐 Stream: {STREAM_WIDTH}x{STREAM_HEIGHT}@{STREAM_FPS}fps | {STREAM_BITRATE}")
print()

fps_stream = detectar_fps(URL_CAMERA)
intervalo = 1.0 / max(fps_stream, 15)

threading.Thread(target=leitor, args=(URL_CAMERA, intervalo), daemon=True).start()
threading.Thread(target=_api_thread, daemon=True).start()

rtmp_proc = iniciar_rtmp()

print(f"✅ API rodando em http://0.0.0.0:{API_PORT}")
print(f"✅ RTMP stream iniciado")
print(f"✅ USAR_ZONA={USAR_ZONA} | Buffer={BUFFER_CRUZAMENTO_PX}px | Histórico={HISTORICO_POSICAO_FRAMES}f")
print("CTRL+C para sair")
print("-"*60)

frame_idx = 0
ultimo_results = None
tempo_inicio = time.time()
frames_fps = 0

while True:
    try:
        frame = frame_queue.get(timeout=1.0)
    except queue.Empty:
        continue

    frame_base = frame.copy()
    frame_idx += 1
    frames_fps += 1
    now = time.time()
    if now - tempo_inicio >= 2.0:
        fps_calculado = frames_fps / (now - tempo_inicio)
        frames_fps = 0
        tempo_inicio = now

    # Reset após pausa
    if rodada_em_pausa and time.time() >= pausa_ate:
        total_contado = 0
        ids_contados.clear()
        hist_pos.clear()
        frames_sem_ver.clear()
        ultimo_results = None
        tempo_rodada_inicio = time.time()
        meta_rodada_atual = proxima_meta_pendente
        proxima_meta_pendente = None
        resultado_dados = {}
        rodada_em_pausa = False
        print(f"🎯 Nova meta: {meta_rodada_atual}")

    meta_atingida = (meta_rodada_atual is not None and total_contado > meta_rodada_atual)
    tempo_esgotado = (time.time() - tempo_rodada_inicio >= INTERVALO_RESET)

    if (not rodada_em_pausa) and (meta_atingida or tempo_esgotado):
        passou = meta_atingida
        td = time.time() - tempo_rodada_inicio
        sr = max(0, INTERVALO_RESET - td)
        historico_rodadas.append({
            "contagem": total_contado,
            "meta": meta_rodada_atual,
            "passou": passou,
            "tempo_decorrido": td,
            "segundos_restantes": sr
        })
        pm = gerar_meta()
        resultado_dados = {
            "passou": passou,
            "contagem": total_contado,
            "meta": meta_rodada_atual,
            "proxima_meta": pm
        }
        print(f"⏱️  Rodada #{len(historico_rodadas)}: {total_contado}/{meta_rodada_atual} "
              f"{'OVER ✅' if passou else 'UNDER ❌'} | tempo={td:.1f}s | proxima={pm}")
        if pm and ultimo_debug_meta:
            print(f"🧠 base={ultimo_debug_meta.get('base')} | "
                  f"teto={ultimo_debug_meta.get('teto_adaptativo')} | "
                  f"final={ultimo_debug_meta.get('meta_final')}")
        rodada_em_pausa = True
        pausa_ate = time.time() + PAUSA_ENTRE_RODADAS
        proxima_meta_pendente = pm

    # YOLO tracking
    if not rodada_em_pausa:
        results = model.track(
            frame, persist=True, tracker="botsort.yaml",
            classes=CLASSES_VEICULOS, conf=CONF_DETECCAO, iou=IOU_DETECCAO,
            imgsz=IMGSZ, half=USE_GPU, verbose=False, device=YOLO_DEVICE
        )
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            ultimo_results = results

    # Contagem de cruzamentos
    ids_neste_frame = set()
    if (not rodada_em_pausa) and ultimo_results and ultimo_results[0].boxes is not None and ultimo_results[0].boxes.id is not None:
        for box in ultimo_results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            area = (x2-x1) * (y2-y1)
            if area < AREA_MINIMA or area > AREA_MAXIMA:
                continue
            prop = (x2-x1) / (y2-y1) if (y2-y1) > 0 else 0
            if prop < PROPORCAO_MINIMA or prop > PROPORCAO_MAXIMA:
                continue
            cx, cy = centro_inferior(x1, y1, x2, y2)
            tid = int(box.id[0])
            ids_neste_frame.add(tid)
            if not na_zona(cx, cy):
                continue
            hist_pos[tid].append((cx, cy))
            frames_sem_ver[tid] = 0
            if tid not in ids_contados and len(hist_pos[tid]) >= 2:
                if cruzou_linha(list(hist_pos[tid])):
                    total_contado += 1
                    ids_contados.add(tid)
                    print(f"🚗 Veiculo #{tid} | Total: {total_contado}")
            # Desenha ponto no frame (opcional, não envia para YouTube)
            cv2.circle(frame_base, (cx, cy), 7, (0,0,0), -1)
            cv2.circle(frame_base, (cx, cy), 5, (0,80,255) if tid in ids_contados else (0,255,80), -1)

        for tid in list(hist_pos.keys()):
            if tid not in ids_neste_frame:
                frames_sem_ver[tid] += 1
                if frames_sem_ver[tid] > FRAMES_ANTES_LIMPAR_ID:
                    del hist_pos[tid]
                    del frames_sem_ver[tid]

    # Preparar frame com overlay e enviar para RTMP
    frame_out = desenhar_frame(frame_base)
    enviar_rtmp(frame_out)

# Cleanup (nunca alcançado, mas mantido)
stop_event.set()
try:
    if rtmp_proc and rtmp_proc.stdin:
        rtmp_proc.stdin.close()
    if rtmp_proc:
        rtmp_proc.terminate()
except:
    pass
print(f"\n{'='*60}\n📊 FINAL: {total_contado} veículos\n{'='*60}")