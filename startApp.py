from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONT_DIR = ROOT / "Front"
PYTHON_EXEC = Path(sys.executable)


def get_local_ip() -> str:
    ip = "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
    except OSError:
        pass
    return ip


def _stream_output(stream, title: str) -> None:
    for line in stream:
        print(f"[{title}] {line}", end="")


def start_process(cmd: list[str], cwd: Path, title: str, pipe: bool = False) -> subprocess.Popen:
    print(f"[startApp] Iniciando {title}: {' '.join(cmd)} (cwd={cwd})")
    stdout = subprocess.PIPE if pipe else None
    stderr = subprocess.STDOUT if pipe else None
    proc = subprocess.Popen(cmd, cwd=cwd, env=os.environ.copy(), stdout=stdout, stderr=stderr, text=True)
    if pipe and proc.stdout:
        threading.Thread(target=_stream_output, args=(proc.stdout, title), daemon=True).start()
    return proc


def stop_process(proc: subprocess.Popen | None, title: str) -> None:
    if proc and proc.poll() is None:
        print(f"[startApp] Encerrando {title}...")
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGINT)
                time.sleep(0.5)
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    if not FRONT_DIR.exists():
        print("[startApp] Pasta 'Front' não encontrada.")
        return 1
    if not (ROOT / "scripts" / "server.py").exists():
        print("[startApp] scripts/server.py não encontrado.")
        return 1

    python = str(PYTHON_EXEC)
    print(f"[startApp] Usando interpretador: {python}")
    if 'venv' not in python:
        print("[startApp] Aviso: parece que o venv não está ativo. Continue apenas se as dependências estiverem instaladas neste Python.")

    server_cmd = [python, "scripts/server.py"]
    server_proc = start_process(server_cmd, ROOT, "Flask (front + API em 5000)")

    tunnel_proc = None
    cloudflared_path = shutil.which("cloudflared")
    if cloudflared_path:
        tunnel_cmd = [cloudflared_path, "tunnel", "--url", "http://127.0.0.1:5000", "--no-autoupdate"]
        tunnel_proc = start_process(tunnel_cmd, ROOT, "Tunnel", pipe=True)
        print("[startApp] Aguarde o link HTTPS (trycloudflare) aparecer acima.")
    else:
        print("[startApp] Para expor via HTTPS, instale o cloudflared (https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/).")

    local_ip = get_local_ip()
    print("\n[ready] Acesse pelo navegador:")
    print(f"  Front+API (LAN):  http://{local_ip}:5000/")
    print("Pressione Ctrl+C para encerrar.")

    processes: list[tuple[str, subprocess.Popen]] = [("Flask", server_proc)]
    if tunnel_proc:
        processes.append(("Tunnel", tunnel_proc))

    exit_code = 0
    try:
        while True:
            time.sleep(0.5)
            for label, proc in processes:
                if proc.poll() is not None:
                    exit_code = proc.returncode or 1
                    raise RuntimeError(f"{label} finalizou inesperadamente (código {proc.returncode}).")
    except KeyboardInterrupt:
        print("\n[startApp] Interrompido pelo usuário.")
    except RuntimeError as exc:
        print(f"\n[startApp] {exc}")
    finally:
        stop_process(tunnel_proc, "Tunnel HTTPS")
        stop_process(server_proc, "Flask")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

