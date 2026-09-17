// Shares the client's authenticated lobby socket; never logs or retains credentials.
(() => {
  if (globalThis.__akagiRecordTransport || !globalThis.WebSocket) return;
  if (!['maj-soul.com', 'mahjongsoul.com'].some(host => location.hostname === host || location.hostname.endsWith('.' + host))) return;
  const NativeSocket = globalThis.WebSocket;
  const nativeSend = NativeSocket.prototype.send;
  const sockets = new Map();
  const synthetic = new WeakSet();
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const MAX_BYTES = 8 * 1024 * 1024;
  const authVersionFields = new Map([
    ['.lq.Lobby.login', 11], ['.lq.Lobby.oauth2Login', 10], ['.lq.Lobby.fastLogin', 1],
  ]);
  function bytes(value) {
    if (value instanceof ArrayBuffer) return new Uint8Array(value);
    if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    return null;
  }
  function fields(data) {
    const result = new Map();
    let p = 0;
    function varint() {
      let value = 0;
      for (let shift = 0; shift < 35 && p < data.length; shift += 7) {
        const b = data[p++]; value += (b & 127) * 2 ** shift;
        if (!(b & 128)) return value;
      }
      throw new Error('invalid protobuf');
    }
    while (p < data.length) {
      const tag = varint(), wire = tag % 8, key = Math.floor(tag / 8);
      if (!key) throw new Error('invalid protobuf field');
      if (wire === 2) {
        const length = varint();
        if (p + length > data.length) throw new Error('truncated protobuf');
        result.set(key, data.subarray(p, p + length)); p += length;
      } else if (wire === 0) {
        // Skip uint64 values without converting them to JavaScript numbers.
        let count = 0, value = 0, b;
        do { if (p >= data.length || count === 10) throw new Error('invalid varint'); b = data[p++]; if (count < 4) value += (b & 127) * 2 ** (7 * count); count++; } while (b & 128);
        result.set(key, value);
      } else if (wire === 1 || wire === 5) {
        p += wire === 1 ? 8 : 4;
        if (p > data.length) throw new Error('truncated protobuf');
      } else throw new Error('unsupported protobuf field');
    }
    return result;
  }
  function field(key, value) {
    const data = typeof value === 'string' ? encoder.encode(value) : value;
    const prefix = [key * 8 + 2]; let length = data.length;
    do { prefix.push((length & 127) | (length > 127 ? 128 : 0)); length >>>= 7; } while (length);
    return new Uint8Array([...prefix, ...data]);
  }
  function withId(data, id) { const copy = data.slice(); copy[1] = id & 255; copy[2] = id >>> 8; return copy; }
  function freeId(state) {
    for (let id = 65535; id >= 0; id--) if (!state.pending.has(id)) return id;
    throw new Error('record_request_busy');
  }
  function finish(entry, value) { clearTimeout(entry.timer); entry.resolve?.(value); entry.resolve = null; }
  function watch(socket) {
    let state = sockets.get(socket);
    if (state) return state;
    let url;
    try { url = new URL(socket.url); } catch { return null; }
    if (!['maj-soul.com', 'mahjongsoul.com'].some(host => url.hostname === host || url.hostname.endsWith('.' + host))
        || url.pathname !== '/gateway' || url.protocol !== 'wss:') return null;
    state = { pending: new Map(), authenticated: false, version: '' };
    sockets.set(socket, state);
    socket.addEventListener('close', () => {
      for (const entry of state.pending.values()) if (entry.kind === 'record') finish(entry, { error: 'login_required' });
      sockets.delete(socket);
    });
    socket.addEventListener('message', (event) => {
      if (synthetic.has(event)) return;
      const data = bytes(event.data);
      if (!data || data.length < 3 || data[0] !== 3) return;
      const id = data[1] | data[2] << 8, entry = state.pending.get(id);
      if (!entry) return;
      state.pending.delete(id);
      if (entry.kind === 'record') {
        // Private RPC replies must not enter the Unity client's dispatcher.
        event.stopImmediatePropagation();
        if (data.length > MAX_BYTES) { finish(entry, { error: 'download_failed' }); return; }
        let binary = '';
        for (let p = 0; p < data.length; p += 32768) binary += String.fromCharCode(...data.subarray(p, p + 32768));
        finish(entry, { data: btoa(binary) });
      } else {
        if (entry.version) {
          try {
            const body = fields(fields(data.subarray(3)).get(2));
            const error = body.get(1);
            state.authenticated = !error || !(fields(error).get(1) || 0);
            if (state.authenticated) state.version = entry.version;
          } catch { state.authenticated = false; }
        }
        if (id !== entry.original) {
          event.stopImmediatePropagation();
          const restored = new MessageEvent('message', { data: withId(data, entry.original).buffer, origin: event.origin });
          synthetic.add(restored); socket.dispatchEvent(restored);
        }
      }
    }, true);
    return state;
  }
  NativeSocket.prototype.send = function (value) {
    const state = watch(this), data = bytes(value);
    if (!state || !data || data.length < 3 || data[0] !== 2) return nativeSend.call(this, value);
    const original = data[1] | data[2] << 8;
    const id = state.pending.has(original) ? freeId(state) : original;
    let version = '';
    try {
      const wrapper = fields(data.subarray(3));
      const fieldId = authVersionFields.get(decoder.decode(wrapper.get(1)));
      if (fieldId) version = decoder.decode(fields(wrapper.get(2)).get(fieldId));
    } catch { /* Normal client traffic remains opaque to this helper. */ }
    state.pending.set(id, { kind: 'client', original, version });
    try { return nativeSend.call(this, id === original ? value : withId(data, id)); }
    catch (error) { state.pending.delete(id); throw error; }
  };
  globalThis.__akagiRecordTransport = {
    async fetch(uuid) {
      if (typeof uuid !== 'string' || !/^[A-Za-z0-9_-]{10,160}$/.test(uuid)) return { error: 'missing_game_uuid' };
      const selected = [...sockets].reverse().find(([socket, state]) => socket.readyState === 1 && socket.binaryType === 'arraybuffer' && state.authenticated && state.version);
      if (!selected) return { error: 'login_required' };
      const [socket, state] = selected;
      let id;
      try { id = freeId(state); } catch { return { error: 'download_failed' }; }
      const body = new Uint8Array([...field(1, uuid), ...field(2, state.version)]);
      const wire = new Uint8Array([2, id & 255, id >>> 8, ...field(1, '.lq.Lobby.fetchGameRecord'), ...field(2, body)]);
      return new Promise(resolve => {
        const entry = { kind: 'record', resolve, timer: null };
        // Keep a tombstone after timeout, so a late private response is swallowed.
        entry.timer = setTimeout(() => finish(entry, { error: 'record_not_ready' }), 15000);
        state.pending.set(id, entry);
        try { nativeSend.call(socket, wire); }
        catch { state.pending.delete(id); finish(entry, { error: 'download_failed' }); }
      });
    },
  };
})()
