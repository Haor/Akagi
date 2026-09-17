/// <reference types="node" />
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { runInNewContext } from 'node:vm'
import { afterEach, expect, it, vi } from 'vitest'

const script = readFileSync(resolve(process.cwd(), '../src/history/majsoul_record_transport.js'), 'utf8')
const encoder = new TextEncoder()
const field = (key: number, value: string | Uint8Array) => {
  const data = typeof value === 'string' ? encoder.encode(value) : value
  return new Uint8Array([key * 8 + 2, data.length, ...data])
}
const request = (id: number, method = '.lq.Lobby.fetchFriendList', body = new Uint8Array()) =>
  new Uint8Array([2, id & 255, id >>> 8, ...field(1, method), ...field(2, body)])
const response = (id: number, body = new Uint8Array()) =>
  new Uint8Array([3, id & 255, id >>> 8, ...field(2, body)])
const idOf = (data: Uint8Array) => data[1] | data[2] << 8

function setup() {
  class Socket extends EventTarget {
    url = 'wss://route-2.maj-soul.com/gateway'
    readyState = 1
    binaryType = 'arraybuffer'
    sent: (Uint8Array | string)[] = []
    send(value: Uint8Array | string) { this.sent.push(value) }
    reply(value: Uint8Array) { this.dispatchEvent(new MessageEvent('message', { data: value.buffer, origin: this.url })) }
  }
  const context = { WebSocket: Socket, location: { hostname: 'game.maj-soul.com' }, URL,
    ArrayBuffer, Uint8Array, TextEncoder, TextDecoder, MessageEvent, setTimeout, clearTimeout, btoa,
    __akagiRecordTransport: undefined as { fetch: (uuid: string) => Promise<{ data?: string; error?: string }> } | undefined }
  runInNewContext(script, context)
  const socket = new Socket()
  const incoming: number[] = []
  socket.addEventListener('message', (event) => incoming.push(idOf(new Uint8Array((event as MessageEvent).data))))
  const login = () => { socket.send(request(1, '.lq.Lobby.oauth2Login', field(10, 'WebGL_2022-0.16.275'))); socket.reply(response(1)) }
  return { socket, incoming, login, fetch: () => context.__akagiRecordTransport!.fetch('260917-synthetic-game') }
}

afterEach(() => vi.useRealTimers())

it('requires an observed authenticated lobby and leaves normal client traffic unchanged', async () => {
  const { socket, incoming, login, fetch } = setup()
  expect(await fetch()).toEqual({ error: 'login_required' })
  const frame = request(7)
  socket.send(frame)
  expect(socket.sent[0]).toBe(frame)
  socket.reply(response(7))
  expect(incoming).toEqual([7])
  login()
  socket.send('text')
  expect(socket.sent.at(-1)).toBe('text')
})

it('allocates around pending client requests and keeps private replies out of the client', async () => {
  const { socket, incoming, login, fetch } = setup()
  login()
  socket.send(request(65535))
  const pending = fetch()
  const id = idOf(socket.sent.at(-1) as Uint8Array)
  expect(id).toBe(65534)
  const reply = response(id, field(4, 'private-record'))
  socket.reply(reply)
  expect(await pending).toEqual({ data: btoa(String.fromCharCode(...reply)) })
  expect(incoming).toEqual([1])
  socket.reply(response(65535))
  expect(incoming).toEqual([1, 65535])
})

it('translates a client id collision and restores the response before client delivery', async () => {
  const { socket, incoming, login, fetch } = setup()
  login()
  const pending = fetch()
  const id = idOf(socket.sent.at(-1) as Uint8Array)
  socket.send(request(id))
  const translated = idOf(socket.sent.at(-1) as Uint8Array)
  expect(translated).not.toBe(id)
  socket.reply(response(translated))
  expect(incoming).toEqual([1, id])
  socket.reply(response(id))
  expect((await pending).data).toBeTruthy()
  expect(incoming).toEqual([1, id])
})

it('swallows late private replies after timeout and cancels requests on disconnect', async () => {
  vi.useFakeTimers()
  const { socket, incoming, login, fetch } = setup()
  login()
  const pending = fetch()
  const id = idOf(socket.sent.at(-1) as Uint8Array)
  await vi.advanceTimersByTimeAsync(15000)
  expect(await pending).toEqual({ error: 'record_not_ready' })
  socket.reply(response(id))
  expect(incoming).toEqual([1])
  const disconnected = fetch()
  socket.dispatchEvent(new Event('close'))
  expect(await disconnected).toEqual({ error: 'login_required' })
  expect(await fetch()).toEqual({ error: 'login_required' })
})
