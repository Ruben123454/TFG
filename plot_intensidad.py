#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def cargar_curva_pixel(path_bin: str, ancho: int, alto: int, x: int, y: int, spp: float) -> np.ndarray:
    if not os.path.isfile(path_bin):
        raise FileNotFoundError(f"No existe el archivo: {path_bin}")

    datos = np.fromfile(path_bin, dtype=np.float32)
    pixeles_por_frame = ancho * alto

    if pixeles_por_frame <= 0:
        raise ValueError("ancho y alto deben ser > 0")

    if datos.size == 0:
        raise ValueError(f"El archivo está vacío: {path_bin}")

    if datos.size % pixeles_por_frame != 0:
        raise ValueError(
            f"Tamaño inválido en {path_bin}: {datos.size} floats no es múltiplo de "
            f"ancho*alto ({pixeles_por_frame})"
        )

    num_frames = datos.size // pixeles_por_frame
    volumen = datos.reshape(num_frames, alto, ancho)

    curva = volumen[:, y, x].astype(np.float64)

    if spp > 0:
        curva = curva / spp

    return curva


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara dos curvas de intensidad temporal en un píxel (x, y)."
    )
    parser.add_argument("--bin1", required=True, help="Primer .bin")
    parser.add_argument("--bin2", required=True, help="Segundo .bin")
    parser.add_argument("--ancho", type=int, required=True, help="Ancho de imagen")
    parser.add_argument("--alto", type=int, required=True, help="Alto de imagen")
    parser.add_argument("--x", type=int, required=True, help="Coordenada X del píxel")
    parser.add_argument("--y", type=int, required=True, help="Coordenada Y del píxel")
    parser.add_argument(
        "--spp",
        type=float,
        default=1.0,
        help="Normalización por samples per pixel (si <= 0 no normaliza)",
    )
    parser.add_argument("--label1", default="Bin 1", help="Etiqueta curva 1")
    parser.add_argument("--label2", default="Bin 2", help="Etiqueta curva 2")
    parser.add_argument("--titulo", default=None, help="Título personalizado")
    parser.add_argument(
        "--guardar",
        default=None,
        help="Ruta para guardar la figura (ej: comparacion.png). Si no se indica, abre ventana.",
    )

    args = parser.parse_args()

    if args.x < 0 or args.x >= args.ancho or args.y < 0 or args.y >= args.alto:
        print(
            f"Error: píxel fuera de rango. x debe estar en [0, {args.ancho - 1}] "
            f"e y en [0, {args.alto - 1}]",
            file=sys.stderr,
        )
        return 1

    try:
        curva1 = cargar_curva_pixel(args.bin1, args.ancho, args.alto, args.x, args.y, args.spp)
        curva2 = cargar_curva_pixel(args.bin2, args.ancho, args.alto, args.x, args.y, args.spp)
    except Exception as e:
        print(f"Error al cargar los datos: {e}", file=sys.stderr)
        return 1

    n = min(curva1.shape[0], curva2.shape[0])
    if curva1.shape[0] != curva2.shape[0]:
        print(
            f"Aviso: distinto número de frames ({curva1.shape[0]} vs {curva2.shape[0]}). "
            f"Se comparan solo los primeros {n}.",
            file=sys.stderr,
        )

    frames = np.arange(n)
    c1 = curva1[:n]
    c2 = curva2[:n]

    plt.figure(figsize=(10, 5))
    plt.plot(frames, c1, label=args.label1, linewidth=1.8)
    plt.plot(frames, c2, label=args.label2, linewidth=1.8)
    plt.xlabel("Frame temporal")
    plt.ylabel("Intensidad")

    titulo = args.titulo or f"Intensidad temporal en píxel (x={args.x}, y={args.y})"
    if args.spp > 0:
        titulo += f" [normalizado por spp={args.spp:g}]"
    plt.title(titulo)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.guardar:
        plt.savefig(args.guardar, dpi=160)
        print(f"Figura guardada en: {args.guardar}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
