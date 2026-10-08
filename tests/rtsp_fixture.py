"""Small local H.264 RTSP/TCP fixture; no camera SDK or RTSP server dependency.

Only used for integration testing: OPTIONS/DESCRIBE/SETUP/PLAY and Basic auth,
Annex-B H.264 input with AUD, RTP single NAL/FU-A packets, optional disconnects.
"""
import argparse
import base64
from pathlib import Path
import re
import socket
import socketserver
import struct
import threading
import time


def access_units(path):
    nals = [x for x in re.split(b"\x00\x00\x00?\x01", Path(path).read_bytes()) if x]
    sps = next(n for n in nals if n[0]&31 == 7)
    pps = next(n for n in nals if n[0]&31 == 8)
    frames, current = [], []
    for nal in nals:
        if nal[0]&31 == 9 and current:
            frames.append(current)
            current = []
        current.append(nal)
    if current:
        frames.append(current)
    return frames, sps, pps


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        server = self.server
        stop = threading.Event()
        lock = threading.Lock()
        sender = None
        self.request.settimeout(.5)

        def send(data):
            with lock:
                self.request.sendall(data)

        def video():
            seq, index = 0, 0
            start = time.monotonic()
            try:
                while not stop.is_set():
                    if server.disconnect and time.monotonic()-start > server.disconnect:
                        self.request.shutdown(socket.SHUT_RDWR)
                        break
                    frame = server.frames[index % len(server.frames)]
                    for ni, nal in enumerate(frame):
                        if len(nal) < 1200:
                            pieces = [nal]
                        else:
                            data, pieces = nal[1:], []
                            for off in range(0, len(data), 1198):
                                end = off+1198 >= len(data)
                                header = (nal[0]&31) | (128 if off == 0 else 0) | (64 if end else 0)
                                pieces.append(bytes([(nal[0]&224)|28, header])+data[off:off+1198])
                        for pi, payload in enumerate(pieces):
                            marker = ni == len(frame)-1 and pi == len(pieces)-1
                            packet = struct.pack('!BBHII', 128, 96|(128 if marker else 0),
                                                 seq % 65536, index*3600 % (2**32), 12345)+payload
                            send(b'$\x00'+struct.pack('!H',len(packet))+packet)
                            seq += 1
                    index += 1
                    stop.wait(max(0, start+index/25-time.monotonic()))
            except OSError:
                pass
            finally:
                stop.set()

        buffered = b''
        try:
            while not stop.is_set():
                try:
                    data = self.request.recv(8192)
                except socket.timeout:
                    continue
                if not data:
                    break
                buffered += data
                while buffered:
                    if buffered.startswith(b'$'):
                        if len(buffered) < 4:
                            break
                        size = struct.unpack('!H', buffered[2:4])[0]+4
                        if len(buffered) < size:
                            break
                        buffered = buffered[size:]
                        continue
                    end = buffered.find(b'\r\n\r\n')
                    if end < 0:
                        break
                    header = buffered[:end].decode()
                    buffered = buffered[end+4:]
                    lines = header.split('\r\n')
                    method, url, _ = lines[0].split(' ',2)
                    headers = {k.lower():v.strip() for k,v in (line.split(':',1) for line in lines[1:])}
                    extra, body, code = '', b'', '200 OK'
                    expected = 'Basic '+base64.b64encode(b'test:testpass').decode()
                    if server.auth and headers.get('authorization') != expected and method != 'OPTIONS':
                        code, extra = '401 Unauthorized', 'WWW-Authenticate: Basic realm="fixture"\r\n'
                    elif method == 'DESCRIBE':
                        sps, pps = (base64.b64encode(v).decode() for v in (server.sps, server.pps))
                        body = (f'v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=fixture\r\nt=0 0\r\n'
                                f'a=control:*\r\nm=video 0 RTP/AVP 96\r\nc=IN IP4 127.0.0.1\r\n'
                                f'a=rtpmap:96 H264/90000\r\na=fmtp:96 packetization-mode=1;sprop-parameter-sets={sps},{pps}\r\n'
                                f'a=control:trackID=0\r\n').encode()
                        extra = f'Content-Type: application/sdp\r\nContent-Base: {url}/\r\n'
                    elif method == 'SETUP':
                        extra = 'Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\nSession: 123456;timeout=60\r\n'
                    elif method == 'PLAY':
                        extra = 'Session: 123456\r\nRange: npt=0-\r\n'
                    elif method == 'OPTIONS':
                        extra = 'Public: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN, GET_PARAMETER\r\n'
                    send((f'RTSP/1.0 {code}\r\nCSeq: {headers.get("cseq","1")}\r\n{extra}'
                          f'Content-Length: {len(body)}\r\n\r\n').encode()+body)
                    if method == 'PLAY' and code == '200 OK' and sender is None:
                        sender = threading.Thread(target=video, daemon=True)
                        sender.start()
                    if method == 'TEARDOWN':
                        stop.set()
        except (OSError, ValueError):
            pass
        finally:
            stop.set()
            if sender:
                sender.join(timeout=2)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port, path, disconnect=0, auth=False):
        self.frames, self.sps, self.pps = access_units(path)
        self.disconnect, self.auth = disconnect, auth
        super().__init__(('127.0.0.1',port), Handler)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('--port',type=int,default=18554)
    p.add_argument('--disconnect',type=float,default=0)
    p.add_argument('--auth',action='store_true')
    a=p.parse_args()
    with Server(a.port,a.video,a.disconnect,a.auth) as server:
        print(f'RTSP fixture listening on {server.server_address}',flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
