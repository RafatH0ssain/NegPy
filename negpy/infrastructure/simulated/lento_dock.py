"""A Lento Dock serial port stand-in: prints BOOT on open and answers the continuous-light commands."""

import queue

_IDENT = "IDENT serial=035SIM/0001/26 model=35 hwrev=v1.0"
_VER = "VER fw=1.0.224 model=35"


class SimDockSerial:
    port = "sim-dock"

    def __init__(self, *, saved_mode: str = "AUTO", saved_expmode: str = "BRGB") -> None:
        self.is_open = True
        self.lines: list[str] = []  # every command received, in order
        self.mode = saved_mode
        self.expmode = saved_expmode
        self.idle = "AUTO"
        self.hold = 0
        self.duty = (0, 0, 0)
        self._out: queue.Queue[bytes] = queue.Queue()
        self._buf = b""
        self._say("BOOT fw=1.0.224 model=35 reason=nrst film=0 entrance=0 cal=persisted")
        self._say(self._state())

    @property
    def lit(self) -> tuple[int, int, int]:
        """The duty the panel shows: continuous mode, idle light on, held."""
        if self.expmode == "CONT" and self.idle != "OFF" and self.hold:
            return self.duty
        return (0, 0, 0)

    def _say(self, line: str) -> None:
        self._out.put((line + "\r\n").encode("ascii"))

    def _state(self) -> str:
        return f"STATE link=1 seq=1 mode={self.mode} st=IDLE film=0 busy=0 expmode={self.expmode} shutter=JACK"

    def write(self, data: bytes) -> None:
        self._buf += data
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            line = raw.decode("ascii").strip()
            if line:
                self.lines.append(line)
                self._answer(line)

    def _answer(self, line: str) -> None:
        cmd, _, arg = line.partition(" ")
        if cmd == "IDENT":
            self._say(_IDENT)
        elif cmd == "VER":
            self._say(_VER)
        elif cmd in ("STATE?", "PING"):
            self._say(self._state())
        elif cmd == "MODE" and arg in ("SEMI", "AUTO", "CONFIG"):
            self.mode = arg
            self._say(f"MODE={arg}")
        elif cmd == "EXPMODE" and arg in ("FIX", "BWHITE", "BRGB", "SEQ", "BRGBPF", "CONT"):
            self.expmode = arg
            self._say(f"EXPMODE={arg}")
        elif cmd == "IDLE" and arg:
            self.idle = arg
            self._say(f"IDLE={arg}")
        elif cmd == "CONTHOLD" and arg in ("0", "1"):
            self.hold = int(arg)
            self._say(f"CONTHOLD {arg}")
        elif cmd == "CONTRGB":
            parts = arg.split()
            if len(parts) == 3 and all(p.isdigit() and int(p) <= 1000 for p in parts):
                self.duty = (int(parts[0]), int(parts[1]), int(parts[2]))
                self._say(f"CONTRGB {arg}")
            else:
                self._say("ERR")
        else:
            self._say("ERR ?")

    def read(self, _n: int) -> bytes:
        try:
            return self._out.get(timeout=0.02)
        except queue.Empty:
            return b""

    def close(self) -> None:
        self.is_open = False
