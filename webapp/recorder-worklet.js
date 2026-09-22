// Забирает сырой PCM с микрофона и пачками отдаёт в основной поток.
// MediaRecorder не подходит: на iOS он пишет AAC, и плате пришлось бы
// тащить ffmpeg ради декодирования.

const BATCH = 4096;

class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(BATCH);
    this.n = 0;
    this.port.onmessage = (e) => {
      if (e.data === "flush") {
        this.port.postMessage({ samples: this.buf.slice(0, this.n), last: true });
        this.n = 0;
      }
    };
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.buf[this.n++] = ch[i];
        if (this.n === BATCH) {
          this.port.postMessage({ samples: this.buf.slice(0), last: false });
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("recorder", RecorderProcessor);
