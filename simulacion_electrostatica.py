"""
================================================================
SIMULACIÓN DE ENERGÍA ELECTROSTÁTICA 2D
Proyecto: Electricidad y Magnetismo — Física III — Mayo 2026
================================================================

FÍSICA IMPLEMENTADA (k = 1, unidades naturales):

  Energía total:
      U = k · Σ_{i<j}  q_i · q_j / |r_i − r_j|

  Potencial eléctrico en r:
      V(r) = k · Σ_i  q_i / |r − r_i|

  Campo eléctrico en r:
      E(r) = k · Σ_i  q_i · (r − r_i) / |r − r_i|³

ALGORITMO: Monte Carlo Greedy (T = 0 K)
  En cada intento se desplaza una carga aleatoria δr.
  El movimiento se acepta SOLO si:
    · la nueva posición permanece en el dominio [-L, L]²
    · la energía total DISMINUYE (U_nueva < U_actual)

CARACTERÍSTICAS:
  · Entrada interactiva del número de cargas (+/−)
  · Agregar y eliminar cargas en tiempo de ejecución
  · Máximo 200 cargas activas simultáneas
  · Registro de datos en cada movimiento aceptado
  · Generación de video GIF con la evolución
  · Análisis completo: U(t), histogramas, mapa de carga libre,
    U vs distancia promedio, potencial V(x,y), campo E(x,y)
================================================================
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Forzar UTF-8 en la salida de consola (necesario en Windows cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ═══════════════════════════════════════════════════════════════
# PARÁMETROS GLOBALES
# ═══════════════════════════════════════════════════════════════
K_E        = 1.0          # Constante de Coulomb (unidades naturales)
L          = 5.0          # Semilado del dominio → dominio = [-L, L]²
DOMAIN     = (-L, L)      # Tupla del dominio
DELTA      = 0.3          # Desplazamiento máximo por iteración [u.a.]
N_ITER_DEF = 30_000       # Iteraciones por defecto
GRID_SIZE  = 120          # Resolución de malla para visualización
MAX_CARGAS = 200          # Máximo de cargas simultáneas
EPS        = 0.05         # Radio de regularización (evita singularidades en malla)
OUT_DIR       = Path(__file__).parent / 'resultados'   # carpeta de salida
GIF_POS       = str(OUT_DIR / 'video_posiciones.gif')  # GIF: posiciones de cargas
GIF_V         = str(OUT_DIR / 'video_potencial.gif')   # GIF: mapa de calor V(x,y)
GIF_E         = str(OUT_DIR / 'video_campo.gif')       # GIF: magnitud |E| + flechas
GIF_DPI       = 72                     # DPI de cada frame en memoria
GIF_SIZE      = (4.0, 4.0)            # Tamaño de figura en pulgadas
GIF_GRID      = 28                     # Resolucion de malla en frames
GIF_MAX_FRAMES = 150                   # Frames maximos por GIF (submuestreo si excede)


def _limpiar_archivos_anteriores() -> None:
    """Elimina la carpeta de resultados anterior y la recrea vacía."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        print(f"  Carpeta anterior eliminada: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True)
    print(f"  Carpeta de resultados creada: {OUT_DIR}")


# Exclusión de volumen duro — las cargas son esferas de radio R_CARGA.
# Dos centros NUNCA pueden estar a menos de R_MIN = 2·R_CARGA.
# Cargas opuestas se atraen hasta "tocarse" (d → R_MIN).
# Cargas iguales se repelen: la energía crece al acercarse,
# pero la restricción de volumen duro impide que se superpongan.
R_CARGA     = 0.20          # Radio fisico de cada carga [u.a.]
R_MIN       = 2 * R_CARGA   # Distancia minima centro-centro = 0.40 [u.a.]
R_DIBUJO_PNG = 0.10          # Radio de dibujo en figuras PNG (más legible)

# ── Coordenadas fijas de presentación: 25(+) + 25(−) ──────────────────────
# Positivas: grid 5×5 en cluster izquierdo  x ∈ [-4.5, -1.5]
# Negativas: grid 5×5 en cluster derecho    x ∈ [ 1.5,  4.5]
# Separación intra-cluster: 0.75 u.a. > R_MIN = 0.40  ✓
_XS_POS = np.array([-4.50, -3.75, -3.00, -2.25, -1.50])
_XS_NEG = np.array([ 1.50,  2.25,  3.00,  3.75,  4.50])
_YS_FIJ = np.array([-1.50, -0.75,  0.00,  0.75,  1.50])

POS_FIJAS = np.array(
    [(x, y) for x in _XS_POS for y in _YS_FIJ] +
    [(x, y) for x in _XS_NEG for y in _YS_FIJ],
    dtype=float
)
Q_FIJAS = np.array([+1.0] * 25 + [-1.0] * 25)
# ──────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
# MÓDULO 1: FÍSICA ELECTROSTÁTICA
# ═══════════════════════════════════════════════════════════════

def calcular_energia(posiciones: np.ndarray, cargas: np.ndarray) -> float:
    """
    Energía electrostática total del sistema.

    U = k · Σ_{i<j} q_i · q_j / |r_i − r_j|

    Implementación vectorizada con NumPy:
      · diff[i,j] = r_i − r_j  (tensor NxNx2)
      · dist[i,j] = |diff[i,j]|  (matriz NxN)
      · Q[i,j]    = q_i · q_j    (matriz NxN)
      · Solo triángulo superior (pares únicos i < j)

    U > 0  pares repulsivos (mismo signo)
    U < 0  pares atractivos (signos opuestos)
    """
    N = len(cargas)
    if N < 2:
        return 0.0
    diff = posiciones[:, np.newaxis, :] - posiciones[np.newaxis, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dist, np.inf)          # auto-interacción → ∞
    Q = np.outer(cargas, cargas)
    sup = np.triu(np.ones((N, N), dtype=bool), k=1)
    return float(K_E * np.sum(Q[sup] / dist[sup]))


def delta_U(posiciones: np.ndarray, cargas: np.ndarray,
            idx: int, nueva_pos: np.ndarray) -> float:
    """
    Cambio de energía al mover la carga idx a nueva_pos.

    DeltaU = k * q_i * sum_{j != i} q_j * (1/r_nuevo - 1/r_viejo)

    Solo recalcula las N-1 interacciones de esa carga -> O(N).
    25x más eficiente que recalcular U completa en O(N^2).
    """
    mask    = np.arange(len(cargas)) != idx
    pos_j   = posiciones[mask]
    q_j     = cargas[mask]
    r_old   = np.maximum(np.linalg.norm(pos_j - posiciones[idx], axis=1), EPS)
    r_new   = np.maximum(np.linalg.norm(pos_j - nueva_pos,        axis=1), EPS)
    return float(K_E * cargas[idx] * np.sum(q_j * (1.0 / r_new - 1.0 / r_old)))


def calcular_potencial(X: np.ndarray, Y: np.ndarray,
                        posiciones: np.ndarray, cargas: np.ndarray) -> np.ndarray:
    """
    Potencial eléctrico escalar en la malla (X, Y).

    V(r) = k · Σ_i q_i / |r − r_i|

    El potencial se suma algebraicamente (superposición lineal).
    V > 0 cerca de cargas +, V < 0 cerca de cargas −.
    """
    V = np.zeros_like(X, dtype=float)
    for q, (xi, yi) in zip(cargas, posiciones):
        r = np.sqrt((X - xi)**2 + (Y - yi)**2)
        V += K_E * q / np.maximum(r, EPS)
    return V


def calcular_campo_electrico(X: np.ndarray, Y: np.ndarray,
                              posiciones: np.ndarray,
                              cargas: np.ndarray) -> tuple:
    """
    Campo eléctrico vectorial en la malla (X, Y).

    E(r) = k · Σ_i q_i · (r − r_i) / |r − r_i|³

    El campo es el negativo del gradiente del potencial: E = −∇V.
    Apunta desde cargas + hacia cargas −.
    """
    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(Y, dtype=float)
    for q, (xi, yi) in zip(cargas, posiciones):
        dx = X - xi
        dy = Y - yi
        r2 = dx**2 + dy**2
        r3 = np.maximum(r2, EPS**2) ** 1.5
        Ex += K_E * q * dx / r3
        Ey += K_E * q * dy / r3
    return Ex, Ey


def V_en_carga(idx: int, posiciones: np.ndarray, cargas: np.ndarray) -> float:
    """
    Potencial eléctrico en la posición de la carga 'idx'
    debido a TODAS las demás (excluye auto-interacción).
    """
    xi, yi = posiciones[idx]
    V = 0.0
    for j, (q, (xj, yj)) in enumerate(zip(cargas, posiciones)):
        if j == idx:
            continue
        r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        V += K_E * q / max(r, EPS)
    return V


def E_en_carga(idx: int, posiciones: np.ndarray, cargas: np.ndarray) -> float:
    """
    Magnitud del campo eléctrico en la posición de la carga 'idx'
    debido a TODAS las demás (excluye auto-campo).
    """
    xi, yi = posiciones[idx]
    Ex = Ey = 0.0
    for j, (q, (xj, yj)) in enumerate(zip(cargas, posiciones)):
        if j == idx:
            continue
        dx = xi - xj
        dy = yi - yj
        r3 = max(dx**2 + dy**2, EPS**2) ** 1.5
        Ex += K_E * q * dx / r3
        Ey += K_E * q * dy / r3
    return float(np.hypot(Ex, Ey))


def distancia_promedio(posiciones: np.ndarray) -> float:
    """Distancia euclídea promedio entre todos los pares de cargas."""
    N = len(posiciones)
    if N < 2:
        return 0.0
    diff = posiciones[:, np.newaxis, :] - posiciones[np.newaxis, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    sup  = np.triu(np.ones((N, N), dtype=bool), k=1)
    return float(np.mean(dist[sup]))


def _hay_colision(nueva_pos: np.ndarray, idx_excluir: int,
                  posiciones: np.ndarray) -> bool:
    """
    Restriccion de volumen duro (hard-core exclusion).

    Retorna True si 'nueva_pos' violaría la distancia mínima R_MIN
    con cualquier otra carga del sistema.

    Fisica:
      - Cada carga es una esfera de radio R_CARGA.
      - Dos cargas colisionan cuando sus centros están a < R_MIN = 2·R_CARGA.
      - Aplica por igual a pares de igual O distinto signo:
          · Opuesto (+/−): se atraen hasta tocarse (d → R_MIN), nunca se unen.
          · Igual (+/+ o −/−): se repelen; la barrera impide solapamiento.
    """
    if len(posiciones) == 0:
        return False
    diffs = posiciones - nueva_pos                       # shape (N, 2)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    if 0 <= idx_excluir < len(posiciones):
        dists[idx_excluir] = np.inf                      # no comparar consigo misma
    return bool((dists < R_MIN).any())


# ═══════════════════════════════════════════════════════════════
# MÓDULO 2: GESTIÓN DEL SISTEMA DE CARGAS
# ═══════════════════════════════════════════════════════════════

class SistemaCargas:
    """
    Encapsula el estado y la gestión de un sistema de cargas
    eléctricas puntuales ±1 en el dominio 2D [-L, L]².

    Límite: MAX_CARGAS = 200 cargas activas simultáneas.
    """

    def __init__(self, n_pos: int = 0, n_neg: int = 0, seed: int = 42,
                 pos_ext: np.ndarray = None, q_ext: np.ndarray = None):
        self.posiciones = np.empty((0, 2), dtype=float)
        self.cargas     = np.empty(0, dtype=float)
        self.seed       = seed
        np.random.seed(seed)
        if pos_ext is not None and q_ext is not None:
            self.posiciones = pos_ext.copy()
            self.cargas     = q_ext.copy()
        elif n_pos + n_neg > 0:
            self._insertar(n_pos, n_neg)

    # ── propiedades de lectura ───────────────────────────────────
    @property
    def N(self) -> int:
        return len(self.cargas)

    @property
    def n_pos(self) -> int:
        return int((self.cargas > 0).sum())

    @property
    def n_neg(self) -> int:
        return int((self.cargas < 0).sum())

    @property
    def energia(self) -> float:
        return calcular_energia(self.posiciones, self.cargas)

    # ── inserción interna ────────────────────────────────────────
    def _insertar(self, n_pos: int, n_neg: int) -> None:
        """
        Inserta n_pos cargas +1 y n_neg cargas −1 usando muestreo de
        rechazo: cada nueva carga se coloca en una posicion aleatoria
        dentro del dominio que no viole la restriccion de volumen duro
        (distancia a cualquier carga existente >= R_MIN).

        Garantiza que desde el primer instante ninguna carga se solapa.
        """
        N_add  = n_pos + n_neg
        q_new  = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])
        perm   = np.random.permutation(N_add)
        q_new  = q_new[perm]

        pos_colocadas = list(self.posiciones)            # posiciones ya ocupadas
        max_intentos  = 20_000

        for k in range(N_add):
            colocada = False
            for _ in range(max_intentos):
                candidato = np.random.uniform(DOMAIN[0], DOMAIN[1], 2)
                if len(pos_colocadas) == 0:
                    pos_colocadas.append(candidato)
                    colocada = True
                    break
                ocupadas = np.array(pos_colocadas)
                dists    = np.sqrt(((ocupadas - candidato) ** 2).sum(axis=1))
                if dists.min() >= R_MIN:
                    pos_colocadas.append(candidato)
                    colocada = True
                    break
            if not colocada:
                # Fallback: colocar aunque haya solapamiento parcial
                pos_colocadas.append(candidato)
                print(f"  Advertencia: carga {k} no pudo colocarse sin "
                      f"solapamiento (dominio muy denso).")

        nuevas = np.array(pos_colocadas)[self.N:]
        self.posiciones = (np.vstack([self.posiciones, nuevas])
                           if self.N > 0 else nuevas.copy())
        self.cargas     = np.concatenate([self.cargas, q_new])

    # ── API pública ──────────────────────────────────────────────
    def agregar_cargas(self, n_pos: int = 0, n_neg: int = 0) -> None:
        """Agrega cargas positivas y/o negativas respetando MAX_CARGAS."""
        N_req = n_pos + n_neg
        if N_req == 0:
            return
        disp = MAX_CARGAS - self.N
        if disp <= 0:
            print(f"  Sistema lleno ({self.N}/{MAX_CARGAS}).")
            return
        if N_req > disp:
            print(f"  Limite: solo se agregan {disp} de {N_req} solicitadas.")
            ratio = disp / N_req
            n_pos = int(n_pos * ratio)
            n_neg = disp - n_pos
        self._insertar(n_pos, n_neg)
        print(f"  Sistema: {self.n_pos}(+) + {self.n_neg}(−) = {self.N} cargas")

    def agregar_carga_en_pos(self, x: float, y: float, signo: float = 1.0) -> bool:
        """
        Agrega una carga en coordenadas específicas (x, y).
        signo: +1.0 (positiva) o -1.0 (negativa).
        Retorna True si fue colocada, False si viola el dominio o el volumen duro.
        """
        if self.N >= MAX_CARGAS:
            print(f"  Sistema lleno ({self.N}/{MAX_CARGAS}).")
            return False
        pos_nueva = np.array([x, y], dtype=float)
        if abs(x) > L or abs(y) > L:
            print(f"  ({x}, {y}) fuera del dominio [{-L}, {L}]². Ingrese de nuevo.")
            return False
        if _hay_colision(pos_nueva, -1, self.posiciones):
            print(f"  ({x:.3f}, {y:.3f}) viola volumen duro (distancia mínima = {R_MIN}). "
                  f"Ingrese de nuevo.")
            return False
        self.posiciones = (np.vstack([self.posiciones, pos_nueva])
                           if self.N > 0 else pos_nueva.reshape(1, 2))
        self.cargas = np.append(self.cargas, float(signo))
        sg = '+1' if signo > 0 else '−1'
        print(f"  Carga ({sg}) colocada en ({x:.3f}, {y:.3f}).  "
              f"Total: {self.n_pos}(+) + {self.n_neg}(−) = {self.N}")
        return True

    def eliminar_carga(self, idx: int = None, tipo: str = None) -> None:
        """
        Elimina una carga del sistema.

        Args:
            idx:  índice de la carga (None = aleatoria)
            tipo: '+' carga positiva, '-' negativa, None = cualquiera
        """
        if self.N == 0:
            print("  No hay cargas para eliminar.")
            return
        if tipo == '+':
            cands = np.where(self.cargas > 0)[0]
            if len(cands) == 0:
                print("  No hay cargas positivas.")
                return
            idx = np.random.choice(cands)
        elif tipo == '-':
            cands = np.where(self.cargas < 0)[0]
            if len(cands) == 0:
                print("  No hay cargas negativas.")
                return
            idx = np.random.choice(cands)
        elif idx is None:
            idx = np.random.randint(0, self.N)

        signo = '+' if self.cargas[idx] > 0 else '−'
        xi, yi = self.posiciones[idx]
        self.posiciones = np.delete(self.posiciones, idx, axis=0)
        self.cargas     = np.delete(self.cargas, idx)
        print(f"  Eliminada carga ({signo}1) idx={idx}  "
              f"pos=({xi:.3f},{yi:.3f}). "
              f"Total: {self.n_pos}(+) + {self.n_neg}(−) = {self.N}")

    def info(self) -> None:
        """Muestra estado completo del sistema en consola."""
        U = self.energia
        d = distancia_promedio(self.posiciones)
        print(f"\n  {'─'*50}")
        print(f"  ESTADO DEL SISTEMA")
        print(f"  {'─'*50}")
        print(f"  Cargas totales    : {self.N} / {MAX_CARGAS}")
        print(f"  Cargas positivas  : {self.n_pos}")
        print(f"  Cargas negativas  : {self.n_neg}")
        print(f"  Energia total U   : {U:+.6f} u.a.")
        print(f"  Dist. prom. cargas: {d:.5f} u.a.")
        print(f"  {'─'*50}")
        print(f"  {'Idx':>4}  {'Carga':>6}  {'x':>9}  {'y':>9}  "
              f"{'V en pos':>10}  {'|E| en pos':>10}")
        print(f"  {'─'*50}")
        for i in range(min(self.N, 20)):  # mostrar hasta 20
            q = self.cargas[i]
            xi, yi = self.posiciones[i]
            Vi  = V_en_carga(i, self.posiciones, self.cargas)
            Ei  = E_en_carga(i, self.posiciones, self.cargas)
            sg  = '+1' if q > 0 else '−1'
            print(f"  {i:4d}  {sg:>6}  {xi:9.4f}  {yi:9.4f}  "
                  f"{Vi:10.4f}  {Ei:10.4f}")
        if self.N > 20:
            print(f"  ... ({self.N - 20} cargas más no mostradas)")
        print(f"  {'─'*50}\n")


# ═══════════════════════════════════════════════════════════════
# MÓDULO 3: ALGORITMO DE MINIMIZACIÓN — Monte Carlo Greedy
# ═══════════════════════════════════════════════════════════════

def minimizar_energia(sistema: SistemaCargas,
                       n_iter: int = N_ITER_DEF,
                       delta: float = DELTA,
                       guardar_frames: bool = False,
                       verbose: bool = True) -> dict:
    """
    Minimiza la energía electrostática del sistema.

    ALGORITMO (T = 0 K — descenso puro de energía):
    ┌──────────────────────────────────────────────────────────┐
    │ U₀ = U(posiciones_iniciales)                             │
    │ Para n = 1, 2, …, N_iter:                                │
    │   1. Elegir índice i  ∈ [0, N−1] uniformemente          │
    │   2. δr ← U[−δ, δ]²  (desplazamiento aleatorio 2D)      │
    │   3. pos_nueva[i] = pos[i] + δr                         │
    │   4. Si pos_nueva[i] ∉ [-L,L]²  → RECHAZAR              │
    │   5. Calcular U_nueva                                    │
    │   6. Si U_nueva < U_actual       → ACEPTAR (guardar)     │
    │      En otro caso                → RECHAZAR              │
    └──────────────────────────────────────────────────────────┘

    En cada movimiento ACEPTADO se registran:
      · número de iteración
      · energía total U
      · snapshot de posiciones
      · distancia promedio entre cargas

    Args:
        sistema        : SistemaCargas a optimizar (se modifica in-place)
        n_iter         : Número total de intentos de movimiento
        delta          : Desplazamiento máximo por paso [u.a.]
        guardar_frames : Guardar PNG en FRAMES_DIR para video
        verbose        : Mostrar progreso por consola

    Returns:
        dict con claves:
          pos_inicial, pos_final,
          U_historial   (todas las iteraciones),
          iter_aceptadas, U_aceptadas, dist_prom_acept,
          n_aceptados, tasa_aceptacion
    """
    pos      = sistema.posiciones.copy()
    cargas   = sistema.cargas.copy()
    N        = len(cargas)
    pos_ini  = pos.copy()
    U_actual = calcular_energia(pos, cargas)

    # Arrays de registro
    U_hist     = np.empty(n_iter + 1)
    U_hist[0]  = U_actual
    iter_acept = [0]
    U_acept    = [U_actual]
    dist_acept = [distancia_promedio(pos)]
    snaps      = [pos.copy()]        # snapshots de posiciones aceptadas
    n_acept    = 0

    # Listas de frames PIL en memoria (sin archivos intermedios en disco)
    _fp, _fv, _fe = [], [], []
    if guardar_frames:
        try:
            from PIL import Image as _PIL_check  # noqa: F401 — verificar disponibilidad
            _fp.append(_frame_pos(pos, cargas, U_actual, 0))
            _fv.append(_frame_potencial(pos, cargas, U_actual, 0))
            _fe.append(_frame_campo(pos, cargas, U_actual, 0))
            print("  Capturando frames en memoria para 3 GIFs...")
        except ImportError:
            guardar_frames = False
            print("  Pillow no disponible — instale con: pip install pillow")

    for n in range(1, n_iter + 1):
        i  = np.random.randint(0, N)
        dr = np.random.uniform(-delta, delta, size=2)
        pos_nueva_i = pos[i] + dr

        # Criterio 1: la nueva posición debe estar dentro del dominio
        if not (DOMAIN[0] <= pos_nueva_i[0] <= DOMAIN[1] and
                DOMAIN[0] <= pos_nueva_i[1] <= DOMAIN[1]):
            U_hist[n] = U_actual
            continue

        # Criterio 2: restriccion de volumen duro — sin solapamiento.
        # La carga i no puede acercarse a ninguna otra a menos de R_MIN.
        # Fisica:
        #   · Cargas opuestas (+/-): se atraen hasta tocarse (d -> R_MIN).
        #     El rechazo de esta condicion hace que "choquen" y permanezcan
        #     lado a lado sin fusionarse.
        #   · Cargas iguales: se repelen; esta barrera impide solapamiento
        #     aunque la energia no fuera suficiente para separarlas.
        if _hay_colision(pos_nueva_i, i, pos):
            U_hist[n] = U_actual
            continue

        # Calcular DeltaU de forma eficiente: O(N) en vez de O(N^2)
        dU = delta_U(pos, cargas, i, pos_nueva_i)

        # Criterio 3: aceptar solo si U disminuye (descenso puro, T = 0 K)
        if dU < 0.0:
            pos[i]    = pos_nueva_i
            U_actual += dU
            n_acept += 1
            iter_acept.append(n)
            U_acept.append(U_actual)
            dist_acept.append(distancia_promedio(pos))
            snaps.append(pos.copy())
            if guardar_frames:
                _fp.append(_frame_pos(pos, cargas, U_actual, n))
                _fv.append(_frame_potencial(pos, cargas, U_actual, n))
                _fe.append(_frame_campo(pos, cargas, U_actual, n))

        U_hist[n] = U_actual

        if verbose and (n % max(1, n_iter // 10) == 0 or n == n_iter):
            print(f"    Iter {n:5d}/{n_iter}  |  U = {U_actual:+12.5f}  |  "
                  f"Acept. {n_acept}/{n} ({100*n_acept/n:.1f}%)")

    # Actualizar sistema con la configuración optimizada
    sistema.posiciones = pos

    # Guardar los 3 GIFs directamente desde memoria (sin PNGs intermedios)
    gifs_guardados = []
    if guardar_frames and _fp:
        fps_gif = 8
        print(f"\n  Guardando {len(_fp)} frames → 3 GIFs animados...")
        try:
            _compilar_gif(_fp, GIF_POS, fps_gif)
            _compilar_gif(_fv, GIF_V,   fps_gif)
            _compilar_gif(_fe, GIF_E,   fps_gif)
            gifs_guardados = [GIF_POS, GIF_V, GIF_E]
        except Exception as _e:
            print(f"  Error al guardar GIFs: {_e}")

    return {
        'pos_inicial'      : pos_ini,
        'pos_final'        : pos.copy(),
        'U_historial'      : U_hist[:n + 1],
        'iter_aceptadas'   : np.array(iter_acept),
        'U_aceptadas'      : np.array(U_acept),
        'dist_prom_acept'  : np.array(dist_acept),
        'snapshots'        : snaps,
        'n_aceptados'      : n_acept,
        'tasa_aceptacion'  : n_acept / n_iter,
        'gifs'             : gifs_guardados,
    }


def _dibujar_cargas(ax, posiciones: np.ndarray, cargas: np.ndarray,
                    alpha: float = 1.0, mostrar_radio: bool = True,
                    radio: float = None) -> None:
    """
    Dibuja cada carga como un circulo fisico.
    radio: radio de dibujo en unidades de datos (None → R_CARGA).
    """
    from matplotlib.patches import Circle
    r = radio if radio is not None else R_CARGA
    for (xi, yi), q in zip(posiciones, cargas):
        color   = '#ef4444' if q > 0 else '#3b82f6'
        circ    = Circle((xi, yi), r,
                         facecolor=color, edgecolor='white',
                         linewidth=0.8, alpha=alpha, zorder=5)
        ax.add_patch(circ)
        signo = '+' if q > 0 else '−'
        ax.text(xi, yi, signo, ha='center', va='center',
                fontsize=max(4, int(r * 28)), color='white',
                fontweight='bold', zorder=6, alpha=alpha)


def _plano_cartesiano_gif(ax) -> None:
    """
    Agrega ejes cartesianos (x=0, y=0) y cuadrícula por unidad a un frame de GIF.
    Ejes principales en azul claro; cuadrícula fina en azul oscuro.
    """
    ax.axhline(0, color='#5599cc', lw=0.9, alpha=0.7, zorder=2)
    ax.axvline(0, color='#5599cc', lw=0.9, alpha=0.7, zorder=2)
    ticks = np.arange(int(DOMAIN[0]), int(DOMAIN[1]) + 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(colors='#aaaaaa', labelsize=6)
    ax.grid(True, color='#1e3a5f', lw=0.5, alpha=0.65)


def _plano_cartesiano_png(ax) -> None:
    """
    Plano cartesiano profesional para figuras PNG estáticas.
    Ejes x=0/y=0 resaltados, grid principal por unidad, grid menor cada 0.5 u.
    """
    ax.axhline(0, color='#7ab8e8', lw=1.2, alpha=0.85, zorder=2)
    ax.axvline(0, color='#7ab8e8', lw=1.2, alpha=0.85, zorder=2)
    ticks_may = np.arange(int(DOMAIN[0]), int(DOMAIN[1]) + 1)
    ax.set_xticks(ticks_may)
    ax.set_yticks(ticks_may)
    ticks_men = np.arange(int(DOMAIN[0]), int(DOMAIN[1]) + 1, 0.5)
    ax.set_xticks(ticks_men, minor=True)
    ax.set_yticks(ticks_men, minor=True)
    ax.tick_params(axis='both', which='major', colors='#bbbbbb', labelsize=9, length=4)
    ax.tick_params(axis='both', which='minor', colors='#555555', labelsize=0, length=2)
    ax.grid(True, which='major', color='#1a3050', lw=0.6, alpha=0.80)
    ax.grid(True, which='minor', color='#15263d', lw=0.3, alpha=0.50)
    for sp in ax.spines.values():
        sp.set_edgecolor('#3b5a7d')
        sp.set_linewidth(1.0)


def _fig_a_pil(fig) -> object:
    """
    Renderiza una figura matplotlib a PIL Image completamente en memoria.
    No escribe ningún archivo en disco. Cierra la figura al terminar.
    Usa FigureCanvasAgg directamente para evitar ventanas emergentes.
    """
    import io
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=GIF_DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()   # .copy() desvincula la imagen del buffer
    buf.close()
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    return img


def _frame_pos(posiciones: np.ndarray, cargas: np.ndarray,
               U: float, iteracion: int) -> object:
    """
    Genera un frame PIL de las posiciones de cargas (sin disco).

    Muestra los círculos físicos de radio R_CARGA con su signo.
    """
    from matplotlib.figure import Figure
    fig = Figure(figsize=GIF_SIZE)
    fig.patch.set_facecolor('#0d1b2a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0d1b2a')

    _dibujar_cargas(ax, posiciones, cargas)

    _plano_cartesiano_gif(ax)
    ax.set_xlim(DOMAIN); ax.set_ylim(DOMAIN)
    ax.set_title(f'Posiciones  |  Iter {iteracion}  |  U={U:+.3f}',
                 color='white', fontsize=9, fontweight='bold')
    ax.set_xlabel('x', color='#aaaaaa', fontsize=8)
    ax.set_ylabel('y', color='#aaaaaa', fontsize=8)
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2d4a6d')
    return _fig_a_pil(fig)


def _frame_potencial(posiciones: np.ndarray, cargas: np.ndarray,
                     U: float, iteracion: int) -> object:
    """
    Genera un frame PIL del mapa de calor del potencial V(x,y) (sin disco).

    Usa imshow sobre malla GIF_GRID×GIF_GRID para máxima velocidad.
    Rojo = V alto (cerca de +), Azul = V bajo (cerca de −).
    """
    from matplotlib.figure import Figure
    x1 = np.linspace(DOMAIN[0], DOMAIN[1], GIF_GRID)
    y1 = np.linspace(DOMAIN[0], DOMAIN[1], GIF_GRID)
    X, Y  = np.meshgrid(x1, y1)
    V     = calcular_potencial(X, Y, posiciones, cargas)

    fig = Figure(figsize=GIF_SIZE)
    fig.patch.set_facecolor('#0d1b2a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0d1b2a')

    ax.imshow(np.clip(V, -15, 15),
              extent=[DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]],
              origin='lower', cmap='RdBu_r', vmin=-15, vmax=15,
              aspect='equal', interpolation='bilinear')
    _dibujar_cargas(ax, posiciones, cargas)

    _plano_cartesiano_gif(ax)
    ax.set_xlim(DOMAIN); ax.set_ylim(DOMAIN)
    ax.set_title(f'Potencial V(x,y)  |  Iter {iteracion}  |  U={U:+.3f}',
                 color='white', fontsize=9, fontweight='bold')
    ax.set_xlabel('x', color='#aaaaaa', fontsize=8)
    ax.set_ylabel('y', color='#aaaaaa', fontsize=8)
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2d4a6d')
    return _fig_a_pil(fig)


def _frame_campo(posiciones: np.ndarray, cargas: np.ndarray,
                 U: float, iteracion: int) -> object:
    """
    Genera un frame PIL del mapa |E(x,y)| + flechas de dirección (sin disco).

    Fondo inferno = intensidad del campo en escala log.
    Flechas blancas normalizadas = dirección del campo eléctrico.
    """
    from matplotlib.figure import Figure
    x1 = np.linspace(DOMAIN[0], DOMAIN[1], GIF_GRID)
    y1 = np.linspace(DOMAIN[0], DOMAIN[1], GIF_GRID)
    X, Y   = np.meshgrid(x1, y1)
    Ex, Ey = calcular_campo_electrico(X, Y, posiciones, cargas)
    Emag   = np.hypot(Ex, Ey)
    En     = np.maximum(Emag, 1e-12)

    fig = Figure(figsize=GIF_SIZE)
    fig.patch.set_facecolor('#0d1b2a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0d1b2a')

    ax.imshow(np.log1p(Emag),
              extent=[DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]],
              origin='lower', cmap='inferno', aspect='equal',
              interpolation='bilinear')

    sk = 2  # submuestreo: 1 de cada sk filas/columnas
    ax.quiver(X[::sk, ::sk], Y[::sk, ::sk],
              Ex[::sk, ::sk] / En[::sk, ::sk],
              Ey[::sk, ::sk] / En[::sk, ::sk],
              color='white', alpha=0.42, scale=52, width=0.004)

    _dibujar_cargas(ax, posiciones, cargas)

    _plano_cartesiano_gif(ax)
    ax.set_xlim(DOMAIN); ax.set_ylim(DOMAIN)
    ax.set_title(f'Campo |E(x,y)|  |  Iter {iteracion}  |  U={U:+.3f}',
                 color='white', fontsize=9, fontweight='bold')
    ax.set_xlabel('x', color='#aaaaaa', fontsize=8)
    ax.set_ylabel('y', color='#aaaaaa', fontsize=8)
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2d4a6d')
    return _fig_a_pil(fig)


def _compilar_gif(frames: list, salida: str, fps: int = 8) -> None:
    """
    Guarda una lista de PIL Images como GIF animado.
    Si hay más de GIF_MAX_FRAMES frames, submuestrea para no saturar.
    Abre el GIF automáticamente con el visor del sistema al terminar.
    """
    from PIL import Image
    if not frames:
        return
    # Submuestreo si se excede el limite de frames
    if len(frames) > GIF_MAX_FRAMES:
        step   = max(1, len(frames) // GIF_MAX_FRAMES)
        frames = frames[::step]
    dur_ms = max(1, int(1000 / fps))
    frames[0].save(
        salida, save_all=True, append_images=frames[1:],
        duration=dur_ms, loop=0, optimize=True
    )
    tam = os.path.getsize(salida) / 1024
    print(f"  {salida:30s}  {len(frames):3d} frames | "
          f"{fps} fps | {tam:6.1f} KB")
    # Abrir con el visor predeterminado del sistema
    try:
        os.startfile(os.path.abspath(salida))
    except Exception:
        pass  # en sistemas sin startfile se ignora silenciosamente


# ═══════════════════════════════════════════════════════════════
# MÓDULO 4: ANÁLISIS ESTADÍSTICO Y FÍSICO
# ═══════════════════════════════════════════════════════════════

def analizar_carga_libre(sistema: SistemaCargas,
                          idx_libre: int = 0,
                          n_grid: int = 60) -> tuple:
    """
    Fija todas las cargas excepto 'idx_libre' y calcula la
    energía total del sistema en función de la posición de
    esa carga sobre una malla.

    U(x, y) = U_fijas + k · q_libre · Σ_{j≠libre} q_j / |r − r_j|

    Permite identificar la región de mínima energía para la
    carga libre dado el resto del sistema fijo.

    Retorna: X, Y (mallas 2D), U_mapa (energía en cada punto)
    """
    pos    = sistema.posiciones.copy()
    cargas = sistema.cargas.copy()
    N      = len(cargas)
    if N < 2 or idx_libre >= N:
        return None, None, None

    # Cargas fijas (todas menos idx_libre)
    pos_f  = np.delete(pos, idx_libre, axis=0)
    q_f    = np.delete(cargas, idx_libre)
    q_lib  = cargas[idx_libre]

    # Energía constante entre cargas fijas
    U_fijas = calcular_energia(pos_f, q_f)

    x1 = np.linspace(DOMAIN[0], DOMAIN[1], n_grid)
    y1 = np.linspace(DOMAIN[0], DOMAIN[1], n_grid)
    X, Y = np.meshgrid(x1, y1)

    # Interacción de la carga libre con cada carga fija (vectorizado)
    U_inter = np.zeros_like(X)
    for (xj, yj), qj in zip(pos_f, q_f):
        r = np.sqrt((X - xj)**2 + (Y - yj)**2)
        U_inter += K_E * q_lib * qj / np.maximum(r, EPS)

    return X, Y, U_fijas + U_inter


def comparar_semillas(n_pos: int, n_neg: int,
                       n_iter: int = 600,
                       seeds: tuple = (7, 42, 123)) -> list:
    """
    Ejecuta la minimización con distintas semillas iniciales
    para comparar la distribución de energías aceptadas y
    analizar la dependencia con las condiciones iniciales.

    Retorna lista de dicts con resultados por semilla.
    """
    resultados = []
    for s in seeds:
        np.random.seed(s)
        sys_tmp = SistemaCargas(pos_ext=POS_FIJAS, q_ext=Q_FIJAS, seed=s)
        res = minimizar_energia(sys_tmp, n_iter=n_iter,
                                delta=DELTA, verbose=False)
        resultados.append({'seed': s, **res})
        print(f"    Semilla {s:5d}: U0={res['U_aceptadas'][0]:+.3f}  "
              f"Uf={res['U_aceptadas'][-1]:+.3f}  "
              f"Acept={res['n_aceptados']}")
    return resultados


# ═══════════════════════════════════════════════════════════════
# MÓDULO 5: VISUALIZACIÓN PRINCIPAL (6 subplots)
# ═══════════════════════════════════════════════════════════════

def _estilo(ax, titulo: str) -> None:
    """Aplica tema oscuro estándar a un eje."""
    ax.set_facecolor('#0d1b2a')
    ax.set_title(titulo, color='white', fontweight='bold', fontsize=12, pad=9)
    ax.tick_params(colors='#bbbbbb', labelsize=9)
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2d4a6d')
    ax.grid(True, color='#1a3050', lw=0.5, alpha=0.7)


def _colorbar(fig, im, ax, label: str) -> None:
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color='#aaaaaa', fontsize=8)
    cb.ax.tick_params(colors='#aaaaaa', labelsize=7)


def visualizar_estado_completo(sistema: SistemaCargas,
                                resultado: dict,
                                guardar: str = str(OUT_DIR / 'resultado_simulacion.png')) -> None:
    """
    Figura educativa principal con 6 subplots.

    Fila superior:
      [1] Scatter posiciones ini/fin   [2] Potencial V(x,y)   [3] |E(x,y)|
    Fila inferior:
      [4] U(t) vs iter. aceptada       [5] Histograma U        [6] U vs dist.

    Muestra en pantalla los valores de V y |E| en cada carga.
    """
    pos    = sistema.posiciones
    q      = sistema.cargas
    pos_0  = resultado['pos_inicial']
    U_hist = resultado['U_historial']
    U_ac   = resultado['U_aceptadas']
    it_ac  = resultado['iter_aceptadas']
    d_ac   = resultado['dist_prom_acept']
    U0, Uf = U_hist[0], U_hist[-1]
    nIt    = len(U_hist) - 1

    mp, mn = q > 0, q < 0

    # Malla para campos
    x1  = np.linspace(DOMAIN[0], DOMAIN[1], GRID_SIZE)
    y1  = np.linspace(DOMAIN[0], DOMAIN[1], GRID_SIZE)
    X, Y = np.meshgrid(x1, y1)

    fig = plt.figure(figsize=(26, 15))
    fig.patch.set_facecolor('#060d1a')

    dU  = Uf - U0
    pct = 100 * dU / (abs(U0) + 1e-12)
    fig.suptitle(
        f'Simulacion de Energia Electrostatica  ·  '
        f'{sistema.n_pos}(+) + {sistema.n_neg}(-) = {sistema.N} cargas\n'
        f'U inicial = {U0:+.4f}  →  U final = {Uf:+.4f}  '
        f'(DU = {dU:+.4f} u.a. | {pct:+.1f}%)',
        fontsize=14, fontweight='bold', color='#e2e8f0', y=0.99
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32,
                  top=0.93, bottom=0.06, left=0.045, right=0.975)

    # ────────────────────────────────────────────────────────────
    # SUB 1: Scatter de posiciones
    # ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _estilo(ax1, 'Posiciones  (inicial → final)')

    # Flechas de desplazamiento ini → fin
    for i in range(len(q)):
        ax1.annotate('', xy=pos[i], xytext=pos_0[i],
                     arrowprops=dict(arrowstyle='->', color='#4b6080',
                                     lw=0.7, alpha=0.5, mutation_scale=8))

    # Posiciones iniciales (semitransparentes, sin simbolo)
    from matplotlib.patches import Circle as _Circ, Patch
    for (xi, yi), qi in zip(pos_0, q):
        c = '#fca5a5' if qi > 0 else '#93c5fd'
        ax1.add_patch(_Circ((xi, yi), R_DIBUJO_PNG, facecolor=c,
                            edgecolor='none', alpha=0.28, zorder=3))

    # Posiciones finales con circulos fisicos
    _dibujar_cargas(ax1, pos, q, radio=R_DIBUJO_PNG)

    # Leyenda manual (Patch para los circulos)
    legend_handles = [
        Patch(facecolor='#ef4444', edgecolor='white',
              label=f'+1  n={sistema.n_pos}'),
        Patch(facecolor='#3b82f6', edgecolor='white',
              label=f'-1  n={sistema.n_neg}'),
    ]

    # Texto con U, V promedio y |E| promedio del sistema
    V_prom = np.mean([abs(V_en_carga(i, pos, q)) for i in range(len(q))])
    E_prom = np.mean([E_en_carga(i, pos, q) for i in range(len(q))])
    ax1.text(0.02, 0.02,
             f'U = {Uf:+.4f} u.a.\n'
             f'<|V|> = {V_prom:.4f}\n'
             f'<|E|> = {E_prom:.4f}\n'
             f'R_carga = {R_CARGA}  R_min = {R_MIN}',
             transform=ax1.transAxes, color='#22d3ee', fontsize=8,
             va='bottom',
             bbox=dict(facecolor='#0f3460', edgecolor='#334155',
                       alpha=0.85, boxstyle='round,pad=0.3'))

    _plano_cartesiano_png(ax1)
    ax1.set_xlim(DOMAIN); ax1.set_ylim(DOMAIN)
    ax1.set_xlabel('x [u.a.]'); ax1.set_ylabel('y [u.a.]')
    ax1.set_aspect('equal')
    lg = ax1.legend(handles=legend_handles, fontsize=9,
                    facecolor='#1e3a5f', edgecolor='#334155')
    for t in lg.get_texts():
        t.set_color('white')

    # ────────────────────────────────────────────────────────────
    # SUB 2: Potencial V(x, y)
    # ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _estilo(ax2, 'Potencial Electrico  V(x, y)')

    V      = calcular_potencial(X, Y, pos, q)
    V_clip = np.clip(V, -20, 20)
    cf2    = ax2.contourf(X, Y, V_clip, levels=60, cmap='RdBu_r', alpha=0.93)
    ax2.contour(X, Y, V_clip, levels=12, colors='white', alpha=0.10, linewidths=0.3)
    _colorbar(fig, cf2, ax2, 'V [u.a.]')

    # Circulos fisicos + etiqueta de V en cada carga
    _dibujar_cargas(ax2, pos, q, radio=R_DIBUJO_PNG)
    for i, (xi, yi) in enumerate(pos):
        Vi = V_en_carga(i, pos, q)
        ax2.annotate(f'{Vi:.1f}', (xi, yi + R_DIBUJO_PNG), fontsize=6,
                     color='#facc15', ha='center', va='bottom',
                     xytext=(0, 2), textcoords='offset points')

    _plano_cartesiano_png(ax2)
    ax2.set_xlim(DOMAIN); ax2.set_ylim(DOMAIN)
    ax2.set_xlabel('x [u.a.]'); ax2.set_ylabel('y [u.a.]')
    ax2.set_aspect('equal')

    # ────────────────────────────────────────────────────────────
    # SUB 3: Magnitud del campo |E(x, y)| + quiver
    # ────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _estilo(ax3, 'Campo Electrico  |E(x, y)|  +  direccion')

    nq = 26
    xq = np.linspace(DOMAIN[0], DOMAIN[1], nq)
    yq = np.linspace(DOMAIN[0], DOMAIN[1], nq)
    Xq, Yq = np.meshgrid(xq, yq)
    Ex, Ey  = calcular_campo_electrico(Xq, Yq, pos, q)
    Emag    = np.hypot(Ex, Ey)

    cf3 = ax3.contourf(Xq, Yq, np.log1p(Emag), levels=50,
                        cmap='inferno', alpha=0.88)
    _colorbar(fig, cf3, ax3, 'ln(1+|E|)')

    # Flechas de dirección (normalizadas)
    sk = 2
    En = np.maximum(Emag, 1e-12)
    ax3.quiver(Xq[::sk, ::sk], Yq[::sk, ::sk],
               Ex[::sk, ::sk] / En[::sk, ::sk],
               Ey[::sk, ::sk] / En[::sk, ::sk],
               color='white', alpha=0.3, scale=48, width=0.003)

    # Circulos fisicos + etiqueta de |E| en cada carga
    _dibujar_cargas(ax3, pos, q, radio=R_DIBUJO_PNG)
    for i, (xi, yi) in enumerate(pos):
        Ei = E_en_carga(i, pos, q)
        ax3.annotate(f'{Ei:.1f}', (xi, yi + R_DIBUJO_PNG), fontsize=6,
                     color='#facc15', ha='center', va='bottom',
                     xytext=(0, 2), textcoords='offset points')

    _plano_cartesiano_png(ax3)
    ax3.set_xlim(DOMAIN); ax3.set_ylim(DOMAIN)
    ax3.set_xlabel('x [u.a.]'); ax3.set_ylabel('y [u.a.]')
    ax3.set_aspect('equal')

    # ────────────────────────────────────────────────────────────
    # SUB 4: U(t) vs iteración aceptada
    # ────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _estilo(ax4, '|U(t)| vs Iteracion Aceptada  (log-log)')

    # Escala log-log sobre |U| (iter 0 → 1 para evitar log(0))
    iters_all = np.maximum(np.arange(len(U_hist)), 1)
    ax4.plot(iters_all, np.abs(U_hist),
             color='#4b6080', lw=0.5, alpha=0.35, label='|U| (todas iter.)')
    it_ac_log = np.maximum(it_ac, 1)
    ax4.plot(it_ac_log, np.abs(U_ac), '-', color='#22d3ee', lw=1.4,
             alpha=0.85, label='|U| (aceptadas)')
    ax4.plot(it_ac_log, np.abs(U_ac), 'o', color='#22d3ee', ms=3.0,
             alpha=0.70, zorder=4)
    ax4.axhline(abs(U0), color='#facc15', ls='--', lw=1.3,
                label=f'|U0| = {abs(U0):.3f}')
    ax4.axhline(abs(Uf), color='#f87171', ls='--', lw=1.3,
                label=f'|Uf| = {abs(Uf):.3f}')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.text(0.98, 0.97,
             f'DU = {dU:+.4f}\n({pct:+.1f}%)',
             transform=ax4.transAxes, color='white', fontsize=9,
             ha='right', va='top',
             bbox=dict(facecolor='#0f3460', edgecolor='#334155',
                       alpha=0.85, boxstyle='round,pad=0.3'))
    ax4.set_xlabel('Iteracion n')
    ax4.set_ylabel('|U| [u.a.]')
    ax4.grid(True, which='both', color='#1a3050', lw=0.4, alpha=0.65)
    ax4.grid(True, which='minor', color='#111e33', lw=0.25, alpha=0.45)
    lg4 = ax4.legend(fontsize=8, facecolor='#1e3a5f', edgecolor='#334155')
    for t in lg4.get_texts():
        t.set_color('white')

    # ────────────────────────────────────────────────────────────
    # SUB 5: Histograma de energías aceptadas
    # ────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _estilo(ax5, 'Distribucion de |U| Aceptadas  (log-log)')

    U_abs = np.abs(U_ac)
    nbins = min(40, max(8, len(U_abs) // 3))
    counts, edges = np.histogram(U_abs, bins=nbins)
    centros = (edges[:-1] + edges[1:]) / 2.0
    mask_pos = counts > 0
    sc5 = ax5.scatter(centros[mask_pos], counts[mask_pos],
                      c=centros[mask_pos], cmap='cividis',
                      s=32, edgecolors='#22d3ee', linewidths=0.7,
                      alpha=0.92, zorder=4)
    ax5.plot(centros[mask_pos], counts[mask_pos],
             color='#22d3ee', lw=0.9, alpha=0.45, zorder=2)
    mu = np.mean(U_abs)
    ax5.axvline(mu, color='#facc15', ls='--', lw=1.5,
                label=f'Media |U| = {mu:.3f}')
    ax5.axvline(abs(Uf), color='#f87171', ls='--', lw=1.5,
                label=f'|Uf| = {abs(Uf):.3f}')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel('|U| [u.a.]')
    ax5.set_ylabel('Frecuencia (counts)')
    ax5.grid(True, which='both', color='#1a3050', lw=0.4, alpha=0.65)
    ax5.grid(True, which='minor', color='#111e33', lw=0.25, alpha=0.45)
    ax5.text(0.02, 0.97,
             f'N muestras = {len(U_abs)}\nsigma = {np.std(U_abs):.4f}',
             transform=ax5.transAxes, color='#aaaaaa', fontsize=8,
             va='top',
             bbox=dict(facecolor='#0f3460', edgecolor='#334155',
                       alpha=0.7, boxstyle='round,pad=0.3'))
    lg5 = ax5.legend(fontsize=8, facecolor='#1e3a5f', edgecolor='#334155')
    for t in lg5.get_texts():
        t.set_color('white')

    # ────────────────────────────────────────────────────────────
    # SUB 6: Energía vs distancia promedio
    # ────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    _estilo(ax6, 'Energia vs Distancia Promedio entre Cargas')

    sc = ax6.scatter(d_ac, U_ac, c=it_ac, cmap='plasma',
                     s=24, alpha=0.80, edgecolors='white',
                     linewidths=0.3, zorder=3)
    cb6 = fig.colorbar(sc, ax=ax6, fraction=0.046, pad=0.04)
    cb6.set_label('Iteracion', color='#cccccc', fontsize=9)
    cb6.ax.tick_params(colors='#aaaaaa', labelsize=8)

    if len(d_ac) > 3:
        coef = np.polyfit(d_ac, U_ac, 1)
        dx   = np.linspace(d_ac.min(), d_ac.max(), 200)
        ax6.plot(dx, np.polyval(coef, dx), '--',
                 color='#facc15', lw=1.6, alpha=0.9, label='Tendencia lineal')
        lg6 = ax6.legend(fontsize=8, facecolor='#1e3a5f', edgecolor='#334155')
        for t in lg6.get_texts():
            t.set_color('white')

    ax6.set_xlabel('Distancia promedio [u.a.]')
    ax6.set_ylabel('Energia U [u.a.]')

    plt.savefig(guardar, dpi=130, bbox_inches='tight', facecolor='#060d1a')
    plt.show()
    print(f"\n  Figura guardada -> {guardar}")


# ═══════════════════════════════════════════════════════════════
# MÓDULO 6: VISUALIZACIÓN DE ANÁLISIS AVANZADO
# ═══════════════════════════════════════════════════════════════

def visualizar_analisis_avanzado(sistema: SistemaCargas,
                                  resultado: dict,
                                  seeds: tuple = (7, 42, 123),
                                  guardar: str = str(OUT_DIR / 'analisis_avanzado.png')) -> None:
    """
    Figura de análisis complementario con 3 subplots:

    [1] Comparación de histogramas de U con 3 semillas distintas
    [2] Mapa de calor U(x,y) con una carga libre y el resto fija
    [3] Potencial V(x,y) + streamlines del campo E
    """
    pos = sistema.posiciones
    q   = sistema.cargas

    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor('#060d1a')
    fig.suptitle('Analisis Avanzado — Condiciones Iniciales | Carga Libre | V+E',
                 fontsize=13, fontweight='bold', color='#e2e8f0', y=0.99)

    gs = GridSpec(1, 3, figure=fig, wspace=0.35,
                  top=0.90, bottom=0.10, left=0.05, right=0.97)

    # ── 1: Comparación histogramas por semilla ───────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _estilo(ax1, 'Histogramas — Distintas Condiciones Iniciales')

    colores = ['#22d3ee', '#f87171', '#a3e635']
    print("\n  Calculando 3 configuraciones iniciales distintas...")
    res_seeds = comparar_semillas(sistema.n_pos, sistema.n_neg,
                                  n_iter=600, seeds=seeds)
    for res_s, col in zip(res_seeds, colores):
        U_ac  = res_s['U_aceptadas']
        nb    = min(25, max(5, len(U_ac) // 3))
        Uf_s  = U_ac[-1]
        ax1.hist(U_ac, bins=nb, color=col, alpha=0.55,
                 edgecolor=col, lw=0.5, density=True,
                 label=f"Semilla {res_s['seed']}  Uf={Uf_s:.2f}")

    ax1.set_xlabel('Energia U [u.a.]')
    ax1.set_ylabel('Densidad de probabilidad')
    lg1 = ax1.legend(fontsize=8, facecolor='#1e3a5f', edgecolor='#334155')
    for t in lg1.get_texts():
        t.set_color('white')
    ax1.text(0.02, 0.97,
             'La distribucion tiende a desplazarse\nhacia menores U '
             'con convergencia.',
             transform=ax1.transAxes, color='#aaaaaa', fontsize=7,
             va='top',
             bbox=dict(facecolor='#0f3460', edgecolor='#334155',
                       alpha=0.7, boxstyle='round,pad=0.3'))

    # ── 2: Mapa de calor con carga 0 libre ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _estilo(ax2, 'U(x,y) — Carga #0 libre | resto fijas')

    print("  Calculando mapa con carga libre...")
    X_m, Y_m, U_m = analizar_carga_libre(sistema, idx_libre=0, n_grid=60)

    if X_m is not None:
        cf2 = ax2.contourf(X_m, Y_m, U_m, levels=50,
                            cmap='RdYlBu_r', alpha=0.90)
        ax2.contour(X_m, Y_m, U_m, levels=12, colors='white',
                    alpha=0.10, linewidths=0.3)
        _colorbar(fig, cf2, ax2, 'U [u.a.]')

        # Mínimo de U → posición óptima para la carga libre
        min_ij = np.unravel_index(np.argmin(U_m), U_m.shape)
        ax2.scatter(X_m[min_ij], Y_m[min_ij], marker='*',
                    c='#facc15', s=220, zorder=10, label='Min U (optimo)')

        # Cargas fijas como circulos fisicos
        q_f = np.delete(q, 0)
        p_f = np.delete(pos, 0, axis=0)
        _dibujar_cargas(ax2, p_f, q_f, radio=R_DIBUJO_PNG)

        # Carga libre como circulo diferenciado (borde amarillo)
        from matplotlib.patches import Circle as _C2
        c0 = '#ef4444' if q[0] > 0 else '#3b82f6'
        ax2.add_patch(_C2((pos[0, 0], pos[0, 1]), R_DIBUJO_PNG,
                          facecolor=c0, edgecolor='#facc15',
                          linewidth=2.0, zorder=7))
        ax2.scatter([], [], marker='o', c=c0, s=60, edgecolors='#facc15',
                    linewidths=2, label='Carga libre (actual)')
        ax2.scatter([], [], marker='*', c='#facc15', s=120,
                    label='Min U (optimo)')
        lg2 = ax2.legend(fontsize=8, facecolor='#1e3a5f', edgecolor='#334155')
        for t in lg2.get_texts():
            t.set_color('white')

    _plano_cartesiano_png(ax2)
    ax2.set_xlim(DOMAIN); ax2.set_ylim(DOMAIN)
    ax2.set_xlabel('x [u.a.]'); ax2.set_ylabel('y [u.a.]')
    ax2.set_aspect('equal')

    # ── 3: V(x,y) + streamlines E ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _estilo(ax3, 'Potencial V(x,y) + Lineas de Campo E')

    x1  = np.linspace(DOMAIN[0], DOMAIN[1], GRID_SIZE)
    y1  = np.linspace(DOMAIN[0], DOMAIN[1], GRID_SIZE)
    X, Y = np.meshgrid(x1, y1)
    V_fin = calcular_potencial(X, Y, pos, q)
    cf3   = ax3.contourf(X, Y, np.clip(V_fin, -20, 20),
                          levels=60, cmap='RdBu_r', alpha=0.88)
    _colorbar(fig, cf3, ax3, 'V [u.a.]')

    ns  = 28
    xs  = np.linspace(DOMAIN[0], DOMAIN[1], ns)
    ys  = np.linspace(DOMAIN[0], DOMAIN[1], ns)
    Xs, Ys = np.meshgrid(xs, ys)
    Exs, Eys = calcular_campo_electrico(Xs, Ys, pos, q)
    Emag_s   = np.maximum(np.hypot(Exs, Eys), 1e-12)
    ax3.streamplot(xs, ys, Exs, Eys,
                   color=np.log1p(Emag_s), cmap='YlOrRd',
                   density=1.3, linewidth=1.0, arrowsize=1.2)

    _dibujar_cargas(ax3, pos, q, radio=R_DIBUJO_PNG)
    _plano_cartesiano_png(ax3)
    ax3.set_xlim(DOMAIN); ax3.set_ylim(DOMAIN)
    ax3.set_xlabel('x [u.a.]'); ax3.set_ylabel('y [u.a.]')
    ax3.set_aspect('equal')

    plt.savefig(guardar, dpi=130, bbox_inches='tight', facecolor='#060d1a')
    plt.show()
    print(f"\n  Figura guardada -> {guardar}")


# ═══════════════════════════════════════════════════════════════
# MÓDULO 7: GENERACIÓN DE VIDEO
# ═══════════════════════════════════════════════════════════════

def abrir_gifs() -> None:
    """
    Abre los 3 GIFs animados con el visor predeterminado del sistema.
    Los GIFs se generan automáticamente al final de la simulación
    cuando se elige guardar frames (opcion 1).
    """
    gifs = [GIF_POS, GIF_V, GIF_E]
    encontrados = [g for g in gifs if os.path.exists(g)]
    if not encontrados:
        print("  No se encontraron GIFs. Ejecute la simulacion (opcion 1)"
              " con la opcion de guardar frames.")
        return
    print(f"  Abriendo {len(encontrados)} GIFs...")
    for gif in encontrados:
        tam = os.path.getsize(gif) / 1024
        print(f"    {gif}  ({tam:.1f} KB)")
        try:
            os.startfile(os.path.abspath(gif))
        except Exception as e:
            print(f"    No se pudo abrir automaticamente: {e}")


# ═══════════════════════════════════════════════════════════════
# MÓDULO 8: INTERFAZ INTERACTIVA (CLI)
# ═══════════════════════════════════════════════════════════════

def _leer_int(prompt: str, lo: int = 0,
              hi: int = MAX_CARGAS, default: int = None) -> int:
    """Lee un entero del usuario con validación de rango."""
    hint = f' [default={default}]' if default is not None else ''
    while True:
        try:
            raw = input(f"  {prompt}{hint}: ").strip()
            if raw == '' and default is not None:
                return default
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"  Valor fuera de rango [{lo}, {hi}].")
        except ValueError:
            print("  Ingrese un numero entero.")


def _leer_yn(prompt: str, default: bool = False) -> bool:
    """Lee una respuesta s/n del usuario."""
    d = 's' if default else 'n'
    r = input(f"  {prompt} (s/n) [{d}]: ").strip().lower()
    if r == '':
        return default
    return r == 's'


def configurar_sistema_inicial() -> 'SistemaCargas':
    """Crea el sistema con las coordenadas fijas de presentación."""
    sistema = SistemaCargas(pos_ext=POS_FIJAS, q_ext=Q_FIJAS)
    print(f"\n  {'═'*52}")
    print("  SISTEMA DE PRESENTACION (coordenadas fijas)")
    print(f"  {'═'*52}")
    print(f"  {sistema.n_pos} cargas (+1) · cluster izquierdo  x ∈ [-4.5, -1.5]")
    print(f"  {sistema.n_neg} cargas (−1) · cluster derecho   x ∈ [ 1.5,  4.5]")
    print(f"  Total: {sistema.N} cargas  |  U inicial = {sistema.energia:+.4f} u.a.")
    print(f"  {'═'*52}")
    return sistema


def _submenu_agregar(sistema: SistemaCargas) -> None:
    disp = MAX_CARGAS - sistema.N
    if disp <= 0:
        print(f"  Sistema lleno ({sistema.N}/{MAX_CARGAS}).")
        return
    print(f"\n  Disponibles: {disp} ranuras")
    print("  1. Agregar cargas en posiciones aleatorias")
    print("  2. Agregar una carga en coordenadas específicas")
    modo = _leer_int("Modo", lo=1, hi=2, default=1)

    if modo == 2:
        print("  Signo:  1 = positiva (+1),  -1 = negativa (−1)")
        sg_raw = _leer_int("Signo de la carga (1 o -1)", lo=-1, hi=1, default=1)
        if sg_raw == 0:
            print("  Signo inválido.")
            return
        print(f"  Dominio: x, y ∈ [{DOMAIN[0]}, {DOMAIN[1]}]")
        while True:
            try:
                raw = input("  Coordenadas → x y: ").strip()
                partes = raw.replace(',', ' ').split()
                x, y = float(partes[0]), float(partes[1])
                if sistema.agregar_carga_en_pos(x, y, float(sg_raw)):
                    break
            except (ValueError, IndexError):
                print("  Formato inválido. Ejemplo:  2.5 -1.0")
    else:
        n_pos = _leer_int("Agregar cargas positivas (+1)", lo=0, hi=disp, default=0)
        disp2 = disp - n_pos
        n_neg = _leer_int("Agregar cargas negativas (-1)", lo=0, hi=disp2, default=0)
        sistema.agregar_cargas(n_pos=n_pos, n_neg=n_neg)


def _submenu_eliminar(sistema: SistemaCargas) -> None:
    if sistema.N == 0:
        print("  No hay cargas.")
        return
    print(f"\n  Cargas: {sistema.n_pos}(+) + {sistema.n_neg}(−) = {sistema.N}")
    print("  Opciones:")
    print("    1. Eliminar una carga positiva (aleatoria)")
    print("    2. Eliminar una carga negativa (aleatoria)")
    print("    3. Eliminar una carga aleatoria (cualquier signo)")
    print("    4. Eliminar por indice especifico")
    opc = _leer_int("Opcion", lo=1, hi=4, default=3)
    if opc == 1:
        sistema.eliminar_carga(tipo='+')
    elif opc == 2:
        sistema.eliminar_carga(tipo='-')
    elif opc == 3:
        sistema.eliminar_carga()
    else:
        idx = _leer_int("Indice de la carga", lo=0, hi=sistema.N - 1)
        sistema.eliminar_carga(idx=idx)


def _submenu_simular(sistema: SistemaCargas) -> dict:
    """Configura y ejecuta la minimización de energía."""
    if sistema.N < 2:
        print("  Se necesitan al menos 2 cargas.")
        return None
    print(f"\n  Sistema: {sistema.n_pos}(+) + {sistema.n_neg}(−) = {sistema.N} cargas")
    print(f"  Energia actual: {sistema.energia:+.5f} u.a.")
    print()
    n_iter = _leer_int(f"Numero de iteraciones (1000 – 200000)",
                        lo=1000, hi=200_000, default=N_ITER_DEF)
    gen_gifs = _leer_yn("Generar GIFs animados (posiciones, V, E)", default=False)
    if gen_gifs:
        print("  Se generaran 3 GIFs al finalizar (sin archivos PNG intermedios).")

    print(f"\n  Ejecutando minimizacion: {n_iter} iter | delta={DELTA}...")
    res = minimizar_energia(sistema, n_iter=n_iter, delta=DELTA,
                             guardar_frames=gen_gifs, verbose=True)

    U0  = res['U_historial'][0]
    Uf  = res['U_historial'][-1]
    dU  = Uf - U0
    pct = 100 * dU / (abs(U0) + 1e-12)
    print(f"\n  {'─'*50}")
    print(f"  Energia inicial U0   : {U0:+.6f}  u.a.")
    print(f"  Energia final  Uf    : {Uf:+.6f}  u.a.")
    print(f"  Reduccion DU         : {dU:+.6f}  u.a.  ({pct:+.2f}%)")
    print(f"  Movim. aceptados     : {res['n_aceptados']}/{n_iter}"
          f"  ({100*res['tasa_aceptacion']:.1f}%)")
    if res.get('gifs'):
        print(f"  GIFs generados       : {', '.join(res['gifs'])}")
    print(f"  {'─'*50}")

    return res


def main() -> None:
    """Punto de entrada — interfaz interactiva de menú."""
    _limpiar_archivos_anteriores()
    print("\n" + "═"*56)
    print("  SIMULACION DE ENERGIA ELECTROSTATICA 2D")
    print("  Proyecto Electricidad y Magnetismo — Fisica III")
    print(f"  Algoritmo: Monte Carlo Greedy | k={K_E} | T=0 K")
    print(f"  Dominio [-{L},{L}]^2 | 25(+) + 25(−) = 50 cargas | delta={DELTA}")
    print("═"*56)

    sistema   = configurar_sistema_inicial()
    resultado = None

    while True:
        print(f"\n  {'─'*52}")
        print("  MENU PRINCIPAL")
        print(f"  {'─'*52}")
        print(f"  Sistema actual: {sistema.n_pos}(+) + {sistema.n_neg}(−)"
              f" = {sistema.N} cargas"
              + (f"  |  U = {resultado['U_historial'][-1]:+.4f}"
                 if resultado else ''))
        print()
        print("  1. Ejecutar simulacion (minimizar energia)")
        print("  2. Agregar cargas al sistema")
        print("  3. Eliminar cargas del sistema")
        print("  4. Ver estado detallado del sistema (V y |E| por carga)")
        print("  5. Ver figura principal (6 subplots)")
        print("  6. Ver analisis avanzado (histogramas, carga libre, V+E)")
        print("  7. Abrir GIFs animados (posiciones / V / campo E)")
        print("  8. Reiniciar sistema")
        print("  9. Salir")
        print()

        opc = _leer_int("Opcion", lo=1, hi=9)

        if opc == 1:
            res_nuevo = _submenu_simular(sistema)
            if res_nuevo is not None:
                resultado = res_nuevo

        elif opc == 2:
            _submenu_agregar(sistema)

        elif opc == 3:
            _submenu_eliminar(sistema)

        elif opc == 4:
            sistema.info()

        elif opc == 5:
            if resultado is None:
                print("  Ejecute primero la simulacion (opcion 1).")
            else:
                visualizar_estado_completo(sistema, resultado)

        elif opc == 6:
            if resultado is None:
                print("  Ejecute primero la simulacion (opcion 1).")
            elif sistema.N < 2:
                print("  Se necesitan al menos 2 cargas.")
            else:
                visualizar_analisis_avanzado(sistema, resultado)

        elif opc == 7:
            abrir_gifs()

        elif opc == 8:
            if _leer_yn("Reiniciar sistema (se perdera el estado actual)",
                         default=False):
                sistema   = configurar_sistema_inicial()
                resultado = None

        elif opc == 9:
            print("\n  Simulacion finalizada.\n")
            break


if __name__ == '__main__':
    main()
