"use strict";

// Главный вид: светящийся шар и больше ничего. Телефон держит микрофон
// открытым и льёт поток на плату по WebSocket; wake word и конец фразы
// считает плата — теми же моделями, что и для железного микрофона.
// Здесь только звук, состояния и рисование.
//
// Ограничения iOS, из которых вырос этот код:
//   * микрофон и звук стартуют только из жеста — отсюда «коснитесь,
//     чтобы разбудить» при первом открытии;
//   * со свёрнутой вкладкой или на заблокированном экране Safari
//     останавливает захват — поток встаёт и поднимается при возврате;
//   * пока микрофон открыт, ответ уходит в разговорный динамик и звучит
//     еле слышно — поэтому на время ответа микрофон закрывается совсем.

const TOKEN_KEY = "jarvisToken";
const MENU_HIDE_MS = 4000;
const SEND_LIMIT_BYTES = 512 * 1024; // не копим отставание в сокете
const RECONNECT_MAX_MS = 10000;

const $ = (id) => document.getElementById(id);
const canvas = $("orb");
const caption = $("caption");
const player = $("player");
const sheet = $("sheet");
const menuBtn = $("menuBtn");

// 0.05 с тишины: «разблокирует» <audio> внутри жеста, иначе iOS не даст
// сыграть ответ, пришедший уже вне жеста.
const SILENT_WAV = "data:audio/wav;base64,UklGRkQDAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YSADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";

let token = load(TOKEN_KEY);
let targetRate = 16000;

let state = "asleep";        // asleep|connecting|idle|listening|thinking|speaking|offline
let level = 0;               // громкость 0..1, по ней «дышит» шар
let envelope = null;         // огибающая ответа Piper: шар говорит ртом Джарвиса
let ws = null;
let reconnectMs = 1000;
let running = false;         // пользователь разбудил и не выключал
let ctx = null;
let workletReady = false;
let mic = null;              // { stream, source, node }
let resampler = null;
let menuTimer = null;

// ------------------------------------------------------------------ утиль

function load(key) {
  try { return localStorage.getItem(key) || ""; } catch { return ""; }
}
function save(key, value) {
  try { localStorage.setItem(key, value); } catch { /* приватный режим */ }
}

const CAPTIONS = {
  asleep: "Коснитесь, чтобы разбудить",
  connecting: "Соединение…",
  idle: "",
  listening: "",
  thinking: "",
  speaking: "",
  offline: "Нет связи с платой",
};

function setState(next, text) {
  state = next;
  caption.textContent = text !== undefined ? text : CAPTIONS[next] || "";
  $("sheetStatus").textContent = {
    asleep: "Джарвис спит",
    connecting: "Соединение…",
    idle: "Джарвис слушает",
    listening: "Слушаю команду",
    thinking: "Думаю",
    speaking: "Отвечаю",
    offline: "Нет связи с платой",
  }[next] || "";
}

// ------------------------------------------------------------------ шар

const g = canvas.getContext("2d");
const PALETTE = {
  asleep:     ["#2a3444", "#1d2735", "#151c27"],
  connecting: ["#3b6cff", "#2a4ba8", "#16233f"],
  idle:       ["#3b6cff", "#2f8fff", "#123056"],
  listening:  ["#22d3ee", "#38bdf8", "#0b3b4a"],
  thinking:   ["#a855f7", "#6366f1", "#2a1d4d"],
  speaking:   ["#34d399", "#22d3ee", "#0d3b34"],
  offline:    ["#ef4444", "#7f1d1d", "#2a1212"],
};

function fitCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const size = Math.min(window.innerWidth, window.innerHeight) * 0.78;
  canvas.style.width = size + "px";
  canvas.style.height = size + "px";
  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);
}
fitCanvas();
window.addEventListener("resize", fitCanvas);

function blob(cx, cy, r, amp, t, phase, color, alpha) {
  const steps = 72;
  g.beginPath();
  for (let i = 0; i <= steps; i++) {
    const a = (i / steps) * Math.PI * 2;
    const wob =
      Math.sin(a * 3 + t * 1.1 + phase) * 0.5 +
      Math.sin(a * 5 - t * 0.8 + phase * 1.7) * 0.3 +
      Math.sin(a * 2 + t * 1.6 + phase * 0.5) * 0.2;
    const rr = r * (1 + amp * wob);
    const x = cx + Math.cos(a) * rr;
    const y = cy + Math.sin(a) * rr;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.closePath();
  // Плотная середина и мягкий край: иначе виден не шар, а размытое
  // пятно, и колебание контура не читается вовсе.
  const grad = g.createRadialGradient(cx, cy, r * 0.05, cx, cy, r);
  grad.addColorStop(0, color);
  grad.addColorStop(0.55, color);
  grad.addColorStop(1, "rgba(0,0,0,0)");
  g.globalAlpha = alpha;
  g.fillStyle = grad;
  g.fill();
  g.globalAlpha = 1;
}

let shown = 0; // сглаженный уровень: без него шар дёргается
function draw(now) {
  requestAnimationFrame(draw);

  const t = now / 1000;
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;

  let target = level;
  if (state === "speaking") target = speakingLevel(t);
  else if (state === "thinking") target = 0.35 + 0.15 * Math.sin(t * 3.2);
  else if (state === "idle") target = 0.12 + 0.05 * Math.sin(t * 1.1);
  else if (state === "asleep" || state === "offline") target = 0.06;
  else if (state === "connecting") target = 0.2 + 0.1 * Math.sin(t * 2.4);

  // Вверх быстро (голос), вниз плавно — так шар «дышит», а не мигает.
  shown += (target - shown) * (target > shown ? 0.35 : 0.06);

  g.clearRect(0, 0, w, h);

  const [c1, c2, c3] = PALETTE[state] || PALETTE.idle;
  const base = Math.min(w, h) * 0.30 * (1 + shown * 0.35);
  const amp = 0.06 + shown * 0.16;

  g.globalCompositeOperation = "lighter";

  // Ореол — он один даёт «свечение в темноте». Радиус упирается в край
  // холста, иначе на границе виден светлый квадрат вместо свечения.
  const haloR = Math.min(base * 1.9, Math.min(w, h) / 2);
  const halo = g.createRadialGradient(cx, cy, base * 0.5, cx, cy, haloR);
  halo.addColorStop(0, c3);
  halo.addColorStop(1, "rgba(0,0,0,0)");
  g.globalAlpha = 0.45 + shown * 0.25;
  g.fillStyle = halo;
  g.beginPath();
  g.arc(cx, cy, haloR, 0, Math.PI * 2);
  g.fill();
  g.globalAlpha = 1;

  // Три доли на медленно вращающемся треугольнике: их пересечения
  // смешивают цвета, и шар «переливается», а не просто пульсирует.
  const spin = t * (state === "thinking" ? 0.9 : 0.25);
  const off = base * (0.18 + shown * 0.12);
  const lobes = [c1, c2, c3];
  for (let i = 0; i < 3; i++) {
    const a = spin + (i * Math.PI * 2) / 3;
    blob(
      cx + Math.cos(a) * off,
      cy + Math.sin(a) * off,
      base * 0.82,
      amp,
      t * (1 + i * 0.15),
      i * 2.1,
      lobes[i],
      0.6
    );
  }

  // Ядро: плотный блик, иначе центр выглядит пустым.
  const core = g.createRadialGradient(cx, cy, 0, cx, cy, base * 0.6);
  core.addColorStop(0, "rgba(255,255,255," + (0.2 + shown * 0.5).toFixed(3) + ")");
  core.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = core;
  g.beginPath();
  g.arc(cx, cy, base * 0.6, 0, Math.PI * 2);
  g.fill();
  g.globalCompositeOperation = "source-over";
}
requestAnimationFrame(draw);

// Пока играет ответ, шар движется по реальной громкости этого ответа.
function speakingLevel() {
  if (!envelope || !player.duration) return 0.4 + 0.2 * Math.sin(performance.now() / 120);
  const i = Math.floor((player.currentTime / player.duration) * envelope.length);
  return envelope[Math.max(0, Math.min(envelope.length - 1, i))];
}

// ------------------------------------------------------------------ меню

function showMenu() {
  menuBtn.classList.add("show");
  clearTimeout(menuTimer);
  menuTimer = setTimeout(() => {
    if (!sheet.classList.contains("show")) menuBtn.classList.remove("show");
  }, MENU_HIDE_MS);
}

document.addEventListener("click", (e) => {
  if (sheet.contains(e.target) || menuBtn.contains(e.target)) return;
  if (!running) { start(); return; }   // первый тап будит Джарвиса
  showMenu();
});

menuBtn.addEventListener("click", () => {
  sheet.classList.add("show");
  $("sheetHint").textContent = "Поток 16 кГц, wake word считает плата.";
});
$("closeSheet").addEventListener("click", () => {
  sheet.classList.remove("show");
  menuBtn.classList.remove("show");
});
$("toggleListen").addEventListener("click", () => {
  if (running) stop("asleep");
  else start();
  $("toggleListen").textContent = running ? "Не слушать" : "Слушать";
});
$("tokenRow").addEventListener("submit", (e) => {
  e.preventDefault();
  token = $("token").value.trim();
  save(TOKEN_KEY, token);
  $("token").value = "";
  $("tokenRow").hidden = true;
  sheet.classList.remove("show");
  start();
});

function askToken(message) {
  $("tokenRow").hidden = false;
  sheet.classList.add("show");
  menuBtn.classList.add("show");
  $("sheetHint").textContent = message || "";
}

// ------------------------------------------------------------------ звук

async function start() {
  if (running) return;
  running = true;
  $("toggleListen").textContent = "Не слушать";

  // Всё, что требует жеста, — синхронно, до первого await.
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  ctx.resume();
  player.src = SILENT_WAV;
  player.play().catch(() => {});

  keepAwake();
  setState("connecting");
  connect();
  try {
    await openMic();
  } catch (err) {
    // Сокет тоже гасим: слушать нечем, держать соединение незачем.
    stop("asleep");
    $("toggleListen").textContent = "Слушать";
    setState("asleep", err.name === "NotAllowedError"
      ? "Микрофон запрещён. Разрешите его в настройках Safari."
      : "Микрофон недоступен: " + err.message);
  }
}

function stop(nextState) {
  running = false;
  closeMic();
  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify({ type: "bye" })); } catch { /* уже закрыт */ }
  }
  if (ws) { ws.onclose = null; ws.close(); ws = null; }
  setState(nextState || "asleep");
  level = 0;
}

async function openMic() {
  if (mic) return;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });

  if (!workletReady) {
    await ctx.audioWorklet.addModule("/static/recorder-worklet.js");
    workletReady = true;
  }

  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "recorder");
  resampler = new StreamResampler(ctx.sampleRate, targetRate);

  node.port.onmessage = (e) => {
    const samples = e.data.samples;
    level = rms(samples) * 3;
    const pcm = resampler.push(samples);
    if (pcm.length && ws && ws.readyState === WebSocket.OPEN && ws.bufferedAmount < SEND_LIMIT_BYTES) {
      ws.send(pcm.buffer);
    }
  };

  source.connect(node);
  node.connect(ctx.destination); // без выхода Safari не гоняет process()
  mic = { stream, source, node };
}

function closeMic() {
  if (!mic) return;
  mic.node.port.onmessage = null;
  mic.source.disconnect();
  mic.node.disconnect();
  mic.stream.getTracks().forEach((t) => t.stop());
  mic = null;
  level = 0;
}

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / (samples.length || 1));
}

// ------------------------------------------------------------------ сокет

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/api/stream";
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    reconnectMs = 1000;
    ws.send(JSON.stringify({ type: "auth", token }));
  };

  ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }

    if (msg.type === "ready") {
      targetRate = msg.sample_rate;
      if (resampler) resampler = new StreamResampler(ctx.sampleRate, targetRate);
      setState("idle");
    } else if (msg.type === "state") {
      // «speaking» ставим сами, когда реально начали играть ответ.
      if (msg.state !== "speaking") setState(msg.state);
    } else if (msg.type === "reply") {
      playReply(msg.audio);
    } else if (msg.type === "error") {
      if (msg.code === "auth") {
        stop("asleep");
        askToken(msg.message);
      } else {
        caption.textContent = msg.message || "Ошибка";
        setTimeout(() => { if (state === "idle") caption.textContent = ""; }, 3000);
      }
    }
  };

  ws.onclose = () => {
    ws = null;
    if (!running) return;
    setState("offline");
    setTimeout(connect, reconnectMs);
    reconnectMs = Math.min(reconnectMs * 2, RECONNECT_MAX_MS);
  };

  ws.onerror = () => { /* закрытие придёт следом */ };
}

// ------------------------------------------------------------------ ответ

async function playReply(b64) {
  const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));

  // Микрофон закрываем целиком: иначе iOS уводит ответ в разговорный
  // динамик, да и Джарвис услышал бы сам себя.
  closeMic();
  envelope = await buildEnvelope(bytes.buffer.slice(0));
  setState("speaking");

  const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
  await new Promise((resolve) => {
    const done = () => { URL.revokeObjectURL(url); resolve(); };
    player.onended = done;
    player.onerror = done;
    player.src = url;
    player.play().catch(done);
  });

  envelope = null;
  setState("idle");
  if (running) {
    openMic().catch(() => {
      stop("asleep");
      setState("asleep", "Микрофон недоступен");
    });
  }
}

// Громкость ответа по 40 мс — по ней шар шевелится в такт голосу.
async function buildEnvelope(buffer) {
  try {
    const audio = await ctx.decodeAudioData(buffer);
    const data = audio.getChannelData(0);
    const step = Math.max(1, Math.floor(audio.sampleRate * 0.04));
    const out = new Float32Array(Math.ceil(data.length / step));
    for (let i = 0; i < out.length; i++) {
      out[i] = Math.min(1, rms(data.subarray(i * step, (i + 1) * step)) * 4);
    }
    return out;
  } catch {
    return null; // не беда: шар будет качаться сам по себе
  }
}

// ------------------------------------------------------------------ экран

async function keepAwake() {
  try { await navigator.wakeLock?.request("screen"); } catch { /* не критично */ }
}

document.addEventListener("visibilitychange", async () => {
  if (document.hidden) {
    // Safari всё равно остановит захват — отпускаем микрофон сами.
    closeMic();
    return;
  }
  if (!running) return;
  keepAwake();
  await ctx?.resume();
  if (ctx && ctx.state !== "running") {
    stop("asleep");
    return;
  }
  connect();
  openMic().catch(() => stop("asleep"));
});

// ------------------------------------------------------- ресемплер потока

// Тот же оконный sinc, что и в классическом виде, но с памятью между
// чанками: без неё на каждой границе (каждые ~85 мс) был бы щелчок,
// а wake word ловил бы его вместо голоса.
class StreamResampler {
  constructor(from, to) {
    this.ratio = from / to;
    this.half = 16;
    this.taps = new Float32Array(2 * this.half + 1);
    const cutoff = (0.45 * to) / from;
    let norm = 0;
    for (let k = -this.half; k <= this.half; k++) {
      const sinc = k === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * k) / (Math.PI * k);
      const w = 0.54 + 0.46 * Math.cos((Math.PI * k) / this.half);
      this.taps[k + this.half] = sinc * w;
      norm += this.taps[k + this.half];
    }
    for (let i = 0; i < this.taps.length; i++) this.taps[i] /= norm;

    this.buf = new Float32Array(0);
    this.absIn = 0;     // индекс первого отсчёта buf в потоке
    this.nextT = this.half; // позиция следующего выходного отсчёта
  }

  push(chunk) {
    if (this.ratio === 1) return floatToInt16(chunk);

    const merged = new Float32Array(this.buf.length + chunk.length);
    merged.set(this.buf, 0);
    merged.set(chunk, this.buf.length);
    this.buf = merged;

    const end = this.absIn + this.buf.length;
    const out = [];
    while (Math.round(this.nextT) + this.half < end) {
      const c = Math.round(this.nextT) - this.absIn;
      let acc = 0;
      for (let k = -this.half; k <= this.half; k++) acc += this.taps[k + this.half] * this.buf[c + k];
      out.push(acc);
      this.nextT += this.ratio;
    }

    const keepFrom = Math.max(0, Math.round(this.nextT) - this.half - this.absIn);
    this.buf = this.buf.slice(keepFrom);
    this.absIn += keepFrom;

    return floatToInt16(out);
  }
}

function floatToInt16(values) {
  const out = new Int16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = Math.max(-1, Math.min(1, values[i]));
    out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return out;
}

// ------------------------------------------------------------------ старт

if (!navigator.mediaDevices?.getUserMedia) {
  setState("offline", "Микрофон доступен только по HTTPS");
} else {
  setState("asleep");
}
