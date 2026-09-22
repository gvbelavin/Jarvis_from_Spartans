"use strict";

// Нажал — пишем, нажал ещё раз — отправляем. Плата отвечает транскриптом,
// текстом ответа и WAV от Piper, который играет здесь же.

const MAX_SECONDS = 30;
const TOKEN_KEY = "jarvisToken";

// 0.05 с тишины: «разблокирует» <audio> внутри нажатия, иначе iOS не даст
// потом сыграть ответ, пришедший уже вне жеста пользователя.
const SILENT_WAV = "data:audio/wav;base64,UklGRkQDAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YSADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";

const $ = (id) => document.getElementById(id);
const talkBtn = $("talk");
const statusEl = $("status");
const logEl = $("log");
const connEl = $("conn");
const player = $("player");
const authForm = $("auth");
const tokenInput = $("token");

let targetRate = 16000;
let ctx = null;
let workletReady = false;
let playerUnlocked = false;
let rec = null; // { stream, source, node, chunks, length, flushed }
let busy = false;

// ---------------------------------------------------------------- токен

function loadToken() {
  try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
}
function saveToken(t) {
  try { localStorage.setItem(TOKEN_KEY, t); } catch { /* приватный режим */ }
}
let token = loadToken();

authForm.addEventListener("submit", (e) => {
  e.preventDefault();
  token = tokenInput.value.trim();
  saveToken(token);
  tokenInput.value = "";
  checkStatus();
});

// ---------------------------------------------------------------- статус

async function checkStatus() {
  try {
    const r = await fetch("/api/status", {
      headers: { "X-Jarvis-Token": token },
      cache: "no-store",
    });
    const s = await r.json();
    targetRate = s.sample_rate;
    connEl.textContent = s.busy ? "отвечает" : "на связи";
    connEl.className = "online";
    authForm.classList.toggle("show", s.auth_required && !s.authorized);
  } catch {
    connEl.textContent = "нет связи";
    connEl.className = "offline";
  }
}
checkStatus();
setInterval(() => { if (!busy && !rec) checkStatus(); }, 15000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) checkStatus();
});

// ---------------------------------------------------------------- лента

function addMessage(kind, text, who) {
  $("empty")?.remove();
  const el = document.createElement("div");
  el.className = "msg " + kind;
  if (who) {
    const w = document.createElement("span");
    w.className = "who";
    w.textContent = who;
    el.appendChild(w);
  }
  el.appendChild(document.createTextNode(text));
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
}

function setState(state, text) {
  talkBtn.classList.toggle("recording", state === "recording");
  talkBtn.classList.toggle("busy", state === "busy");
  talkBtn.setAttribute("aria-label", state === "recording" ? "Отправить" : "Говорить");
  statusEl.textContent = text || "";
}

// ---------------------------------------------------------------- запись

talkBtn.addEventListener("click", () => {
  if (busy) return;
  if (rec) { stopAndSend(); return; }
  startRecording();
});

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    addMessage("err", "Браузер не даёт микрофон: страницу нужно открыть по HTTPS.");
    return;
  }

  // Всё, что требует жеста пользователя, — синхронно, до первого await.
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  ctx.resume();
  if (!playerUnlocked) {
    player.src = SILENT_WAV;
    player.play().then(() => { playerUnlocked = true; }).catch(() => {});
  }

  busy = true;
  setState("busy", "Включаю микрофон…");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    if (!workletReady) {
      await ctx.audioWorklet.addModule("/static/recorder-worklet.js");
      workletReady = true;
    }
    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "recorder");
    const r = { stream, source, node, chunks: [], length: 0, flushed: null, started: performance.now() };

    let onLast;
    r.flushed = new Promise((res) => { onLast = res; });
    node.port.onmessage = (e) => {
      const { samples, last } = e.data;
      r.chunks.push(samples);
      r.length += samples.length;
      showLevel(samples);
      if (last) onLast();
      else if (r.length / ctx.sampleRate >= MAX_SECONDS && rec === r) stopAndSend();
    };

    source.connect(node);
    node.connect(ctx.destination); // без выхода Safari не гоняет process()
    rec = r;
    busy = false;
    setState("recording", "Слушаю… нажмите, чтобы отправить");
    tickTimer(r);
  } catch (err) {
    busy = false;
    setState("idle");
    addMessage("err", err.name === "NotAllowedError"
      ? "Доступ к микрофону запрещён. Разрешите его в настройках Safari для этого сайта."
      : "Не удалось включить микрофон: " + err.message);
  }
}

function tickTimer(r) {
  if (rec !== r) return;
  const s = Math.floor((performance.now() - r.started) / 1000);
  statusEl.textContent = `Слушаю… ${s} с — нажмите, чтобы отправить`;
  setTimeout(() => tickTimer(r), 500);
}

function showLevel(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / (samples.length || 1));
  talkBtn.style.setProperty("--ring", Math.min(28, rms * 300) + "px");
}

async function stopAndSend() {
  const r = rec;
  rec = null;
  busy = true;
  setState("busy", "Думаю…");
  talkBtn.style.setProperty("--ring", "0px");

  r.node.port.postMessage("flush");
  await r.flushed;
  r.source.disconnect();
  r.node.disconnect();
  // Микрофон отпускаем сразу: пока он открыт, iOS выводит звук в
  // разговорный динамик, и ответ еле слышно.
  r.stream.getTracks().forEach((t) => t.stop());

  const pcm = toInt16(resample(concat(r.chunks, r.length), ctx.sampleRate, targetRate));
  if (pcm.length < targetRate * 0.4) {
    busy = false;
    setState("idle", "Слишком коротко — попробуйте ещё раз");
    return;
  }

  try {
    const resp = await fetch("/api/talk", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Sample-Rate": String(targetRate),
        "X-Jarvis-Token": token,
      },
      body: pcm.buffer,
    });
    const data = await resp.json().catch(() => ({}));

    if (resp.status === 401) {
      authForm.classList.add("show");
      tokenInput.focus();
      throw new Error("Нужен токен доступа — введите его сверху.");
    }
    if (!resp.ok) throw new Error(data.error || `Ошибка сервера (${resp.status})`);

    if (data.transcript) addMessage("me", data.transcript, data.speaker?.name);
    for (const reply of data.replies || []) {
      addMessage("jarvis", reply.text, "Джарвис");
      setState("busy", "Говорю…");
      await playWav(reply.audio);
    }
    setState("idle");
  } catch (err) {
    addMessage("err", err.message || "Нет связи с платой.");
    setState("idle");
  } finally {
    busy = false;
  }
}

// ---------------------------------------------------------------- звук

function concat(chunks, length) {
  const out = new Float32Array(length);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

// Ресемплинг до частоты модели (16 кГц) с НЧ-фильтром, чтобы Whisper и
// Speaker ID не получали алиасинг. Окно Хэмминга, 33 отвода.
function resample(input, from, to) {
  if (from === to) return input;
  const ratio = from / to;
  const cutoff = (0.45 * to) / from; // доля от частоты входа
  const half = 16;
  const taps = new Float32Array(2 * half + 1);
  let norm = 0;
  for (let k = -half; k <= half; k++) {
    const sinc = k === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * k) / (Math.PI * k);
    const w = 0.54 + 0.46 * Math.cos((Math.PI * k) / half);
    taps[k + half] = sinc * w;
    norm += taps[k + half];
  }
  for (let i = 0; i < taps.length; i++) taps[i] /= norm;

  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const c = Math.round(i * ratio);
    let acc = 0;
    for (let k = -half; k <= half; k++) {
      const j = c + k;
      if (j >= 0 && j < input.length) acc += taps[k + half] * input[j];
    }
    out[i] = acc;
  }
  return out;
}

function toInt16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const v = Math.max(-1, Math.min(1, f32[i]));
    out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return out;
}

function playWav(b64) {
  const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
  return new Promise((resolve) => {
    const done = () => { URL.revokeObjectURL(url); resolve(); };
    player.onended = done;
    player.onerror = done;
    player.src = url;
    player.play().catch(done);
  });
}
