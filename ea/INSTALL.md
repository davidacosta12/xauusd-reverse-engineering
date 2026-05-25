# XAUUSD Asian Momentum v3 — Guía de instalación y despliegue

**Estrategia validada:** PF total 2.54 | PF matched 7.18 | WR 70.6% (OOS mayo 2026)

---

## 1. Copiar el archivo .mq5 a MT5

1. Abre MetaTrader 5 (MEX Atlantic demo)
2. Menú **Archivo → Abrir carpeta de datos** (o `Ctrl+Shift+D`)
3. Navega a `MQL5 / Experts /`
4. Copia `XAUUSD_AsianMomentum_v3.mq5` en esa carpeta

> La ruta completa suele ser:
> `C:\Users\<usuario>\AppData\Roaming\MetaQuotes\Terminal\<id>\MQL5\Experts\`

---

## 2. Compilar con MetaEditor

1. Desde MT5: **Herramientas → MetaEditor** (o `F4`)
2. En el árbol de archivos del MetaEditor, localiza `Experts / XAUUSD_AsianMomentum_v3.mq5`
3. Pulsa `F7` para compilar
4. El panel de mensajes inferior debe mostrar `0 errores, 0 advertencias`
5. Si hay error de compilación, revisa la sección de bugs conocidos al final de este documento

---

## 3. Adjuntar al gráfico

1. En MT5, abre un gráfico de **XAUUSD.. M1** (el símbolo con dos puntos es el de MEX Atlantic)
2. En el panel "Navegador" (izquierda), expande **Expert Advisors**
3. Doble clic o arrastra `XAUUSD_AsianMomentum_v3` al gráfico
4. En la ventana de configuración:
   - Pestaña **Común**: marcar "Permitir trading automatizado"
   - Pestaña **Parámetros de entrada**: configurar los inputs (ver sección 4)
5. Clic en **Aceptar**
6. Verificar que en la esquina superior derecha del gráfico aparece el nombre del EA con una carita feliz (no triste)

> **Importante:** El botón "Auto Trading" en la barra de MT5 debe estar activado (verde).

---

## 4. Configuración de inputs recomendada

### Fase 1 — Solo alertas (semanas 1-4)

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| OperationMode | MODE_ALERT_ONLY (0) | No abre órdenes |
| VolumeLots | 0.02 | Volumen que se usará en LIVE |
| UseMaxTradesPerDay | true | |
| MaxTradesPerDay | 2 | Validado empiricamente |
| CooldownMinutes | 90 | Validado empiricamente |
| SessionStartHourUTC | 22 | |
| SessionEndHourUTC | 3 | |
| MinDistFromAsianMidPips | 100.0 | |
| BB_PctB_Buy_Threshold | 0.55 | |
| BB_PctB_Sell_Threshold | 0.45 | |
| SLPipsBuy | 260.7 | Mediana operador real |
| TPPipsBuy | 272.1 | Mediana operador real |
| SLPipsSell | 342.8 | Mediana operador real |
| TPPipsSell | 289.8 | Mediana operador real |
| MaxSpreadPoints | 50 | |
| EnableSoundAlert | true | |

Durante esta fase verifica manualmente que las alertas se producen en horas razonables (22:00-02:59 UTC) y que los precios/direcciones tienen sentido.

### Fase 2 — Paper trading (semanas 5-8)

Cambiar solo:

| Parámetro | Valor |
|-----------|-------|
| OperationMode | MODE_PAPER (1) |

El EA guardará cada señal en `MQL5/Files/AsianMomentum_v3_paper_trades.csv`. Revisa el PnL simulado semanalmente y compáralo con el backtest de referencia (+$196 OOS).

### Fase 3 — Live (semana 9 en adelante)

Cambiar solo:

| Parámetro | Valor |
|-----------|-------|
| OperationMode | MODE_LIVE (2) |
| VolumeLots | 0.01 (empezar pequeño) |

Después de 20 trades reales sin anomalías, subir a 0.02.

---

## 5. Ver logs del EA

### Pestaña "Expertos" en MT5

- Menú **Ver → Terminal** (o `Ctrl+T`)
- Pestaña **Expertos**: muestra todos los Print() del EA en tiempo real
- Busca los mensajes de inicio: `=== XAUUSD Asian Momentum v3 INIT ===`
- Confirma que `PipsToPrice(1 pip) ≈ 0.10` (crítico para SL/TP correctos)

### Archivos CSV (logs persistentes)

Los archivos CSV se guardan en la carpeta **compartida** de MT5 (FILE_COMMON):

```
C:\Users\<usuario>\AppData\Roaming\MetaQuotes\Terminal\Common\Files\
```

Archivos generados:
- `AsianMomentum_v3_signals.csv` — todas las señales detectadas con precio, SL, TP y razón
- `AsianMomentum_v3_paper_trades.csv` — trades simulados (solo en MODE_PAPER)

Ábrelos con Excel o cualquier editor de texto (delimitador: `;`).

---

## 6. Verificación crítica post-instalación

Antes de pasar a MODE_LIVE, confirma estos puntos en la pestaña Expertos:

```
✓ _Digits=2 _Point=0.01 (XAUUSD MEX Atlantic estándar)
✓ PipsToPrice(1 pip) = 0.10000 (si fuera 0.01, hay que ajustar PipsToPrice en el código)
✓ Señales aparecen entre las 22:00 y 02:59 UTC solamente
✓ asian_mid cambia con el tiempo (no está fijo)
✓ SL/TP calculados correctamente: SL BUY ≈ entry - 26.07 pts, TP BUY ≈ entry + 27.21 pts
```

---

## 7. Bugs conocidos y limitaciones

Ver `BUGS.md` en el mismo directorio para el análisis completo. Resumen:

1. **Conversión de pips** — Si el broker usa `_Digits=3` (XAUUSD con 3 decimales), `PipsToPrice` devuelve el valor incorrecto. Verificar en OnInit con el Print de `PipsToPrice(1 pip)`.
2. **Timezone del broker** — El EA usa `TimeGMT()` para UTC. MEX Atlantic opera en GMT+3 pero `TimeGMT()` devuelve UTC puro independientemente. Correcto mientras el broker no override `TimeGMT()`.
3. **Asian range en primer bar** — La primera vela de la ventana (22:00 exactas) siempre retorna `asian_range_not_ready` porque no hay barras previas. Es el comportamiento correcto (point-in-time).
4. **Reset de estado tras reinicio** — Si el EA se reinicia durante la sesión asiática activa, `g_trades_today` se pone a 0. Podría abrir más de MaxTradesPerDay si ya había operado antes del reinicio.

---

## 8. Plan de despliegue gradual

```
Semanas 1-4:  MODE_ALERT_ONLY
              Objetivo: confirmar que señales coinciden con lo esperado
              Metrica: >= 1 señal/semana en horario correcto
              
Semanas 5-8:  MODE_PAPER
              Objetivo: verificar PnL simulado vs backtest OOS
              Referencia: backtest OOS = +$196 en 15 dias (17 trades)
              
Semana 9+:    MODE_LIVE con 0.01 lot
              Condicion de entrada: paper trading positivo
              Escalar a 0.02 lots despues de 20 trades reales sin anomalias
```

---

## 9. Contacto y mantenimiento

- Si las condiciones de mercado cambian significativamente, el recall puede caer.
- Re-evaluar la estrategia cada 3 meses con nuevos datos OOS.
- El EA NO se auto-actualiza — cualquier cambio de parámetros requiere recompilación.
