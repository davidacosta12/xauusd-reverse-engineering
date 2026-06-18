//+------------------------------------------------------------------+
//|                        XAUUSD_AsianMomentum_v3.mq5              |
//|           Estrategia Asian Range Breakout con momentum           |
//|              Validada in-sample + OOS (mayo 2026)                |
//|                                                                  |
//|  Parametros derivados empiricamente de 30 trades in-sample y    |
//|  validados en 12 trades OOS. PF matched OOS = 7.18.             |
//|                                                                  |
//|  ATENCIÓN: El broker MEX Atlantic usa GMT+3 en el servidor.      |
//|  Este EA opera en UTC (TimeGMT). Verificar que el broker no      |
//|  aplique DST automaticamente.                                    |
//+------------------------------------------------------------------+
#property copyright "Reverse-engineered XAUUSD strategy v3"
#property version   "1.00"
#property description "Asian Range Breakout con momentum — sesion 22:00-02:59 UTC"

//=== MODOS DE OPERACIÓN ============================================
enum ENUM_OPERATION_MODE
  {
   MODE_ALERT_ONLY = 0,  // Solo alertas — no abre ordenes
   MODE_PAPER      = 1,  // Simula trades en CSV — no abre ordenes reales
   MODE_LIVE       = 2,  // Abre ordenes reales (usar solo en demo primero)
  };

//=== INPUTS ========================================================
input group "=== Modo de operacion ==="
input ENUM_OPERATION_MODE OperationMode  = MODE_ALERT_ONLY;

input group "=== Risk Management ==="
input double VolumeLots          = 0.02;   // Lotes por operacion
input bool   UseMaxTradesPerDay  = true;   // Limitar max entradas/dia
input int    MaxTradesPerDay     = 2;      // Max entradas por session-day
input int    CooldownMinutes     = 90;     // Minutos minimos entre señales

input group "=== Session asiatica (UTC) ==="
input int    SessionStartHourUTC = 22;     // Inicio sesion core (22h UTC)
input int    SessionEndHourUTC   = 3;      // Fin sesion core — exclusivo (03h UTC)
input int    AsianRangeStartUTC  = 22;     // Inicio ventana de formacion del rango
input int    AsianRangeEndUTC    = 2;      // Fin ventana de formacion — barra >=2h usa rango cerrado

input group "=== Parametros de señal ==="
input double MinDistFromAsianMidPips = 100.0; // Distancia minima al asian_mid (pips)
input double BB_PctB_Buy_Threshold   = 0.55;  // BB %B umbral BUY
input double BB_PctB_Sell_Threshold  = 0.45;  // BB %B umbral SELL
input int    MinConsecBullForBuy     = 1;      // Velas consecutivas alcistas M5
input int    MinConsecBearForSell    = 1;      // Velas consecutivas bajistas M15

input group "=== SL/TP empiricos (medianas operador) ==="
input double SLPipsBuy   = 260.7;  // SL para BUY en pips
input double TPPipsBuy   = 272.1;  // TP para BUY en pips
input double SLPipsSell  = 342.8;  // SL para SELL en pips
input double TPPipsSell  = 289.8;  // TP para SELL en pips

input group "=== Bollinger Bands (M15) ==="
input int    BB_Period    = 20;
input double BB_Deviation = 2.0;

input group "=== Spread y ejecucion ==="
input int    MaxSpreadPoints  = 50;   // No abrir si spread supera este valor
input int    SlippagePoints   = 10;   // Desviacion maxima en puntos

input group "=== Alertas y logging ==="
input bool   EnableSoundAlert = true;
input bool   EnablePushAlert  = false;
input bool   EnableEmailAlert = false;
input string LogFilePrefix    = "AsianMomentum_v3";

//=== ENUMS LOCALES =================================================
enum ENUM_SIGNAL_TYPE { SIGNAL_NONE = 0, SIGNAL_BUY = 1, SIGNAL_SELL = 2 };

//=== ESTADO GLOBAL =================================================
int      g_bb_handle      = INVALID_HANDLE;
datetime g_date_anchor    = 0;     // session_date en curso
int      g_trades_today   = 0;     // señales aceptadas hoy
datetime g_last_signal_ts = 0;     // timestamp ultima señal aceptada

//+------------------------------------------------------------------+
//|  UTILIDADES DE TIEMPO                                            |
//+------------------------------------------------------------------+

// Comprueba si hora_utc esta dentro de la sesion core (cruza medianoche).
bool IsInCoreSession(int hour_utc)
  {
   if(SessionStartHourUTC < SessionEndHourUTC)
      return (hour_utc >= SessionStartHourUTC && hour_utc < SessionEndHourUTC);
   // Cruza medianoche: ej. 22 <= h OR h < 3
   return (hour_utc >= SessionStartHourUTC || hour_utc < SessionEndHourUTC);
  }

// Devuelve el "session_date" normalizado a medianoche UTC.
// Horas >= SessionStartHourUTC (22) pertenecen a la sesion del DIA SIGUIENTE.
datetime GetSessionDate(datetime ts_utc)
  {
   MqlDateTime dt;
   TimeToStruct(ts_utc, dt);
   if(dt.hour >= SessionStartHourUTC)
     {
      // Avanzar al dia siguiente y truncar a medianoche
      datetime next = ts_utc + 86400;
      MqlDateTime nd;
      TimeToStruct(next, nd);
      nd.hour = 0; nd.min = 0; nd.sec = 0;
      return StructToTime(nd);
     }
   MqlDateTime nd = dt;
   nd.hour = 0; nd.min = 0; nd.sec = 0;
   return StructToTime(nd);
  }

//+------------------------------------------------------------------+
//|  ASIAN RANGE ROLLING                                             |
//+------------------------------------------------------------------+
// Calcula el rango asiatico en modo rolling (punto-en-tiempo correcto):
//   - Si T esta dentro de la ventana (22:00-02:00 UTC):
//       high/low acumulado de barras M15 desde el inicio de la ventana hasta T-1
//   - Si T esta fuera (02:00-22:00 UTC):
//       high/low final del ultimo rango cerrado
//
// BUG CONOCIDO: iTime/iHigh/iLow iteran desde la barra mas reciente (i=0)
// hacia atras. El bucle asume barras en orden cronologico inverso;
// break cuando bar_time < window_start es correcto en ese sentido.
// Limite de 200 iteraciones M15 cubre mas de 50 horas — suficiente.
bool ComputeAsianRange(datetime current_ts, double &asian_high, double &asian_low, double &asian_mid)
  {
   MqlDateTime now_dt;
   TimeToStruct(current_ts, now_dt);
   int h = now_dt.hour;

   datetime window_start;
   bool     window_is_open;

   if(h >= AsianRangeStartUTC)
     {
      // 22h, 23h — ventana empieza HOY a las 22:00
      MqlDateTime ws = now_dt;
      ws.hour = AsianRangeStartUTC; ws.min = 0; ws.sec = 0;
      window_start   = StructToTime(ws);
      window_is_open = true;
     }
   else if(h < AsianRangeEndUTC)
     {
      // 00h, 01h — ventana empieza AYER a las 22:00
      datetime yesterday = current_ts - 86400;
      MqlDateTime yd;
      TimeToStruct(yesterday, yd);
      yd.hour = AsianRangeStartUTC; yd.min = 0; yd.sec = 0;
      window_start   = StructToTime(yd);
      window_is_open = true;
     }
   else
     {
      // 02h-21h — ventana ya cerrada; usar el rango completo de ayer 22 -> hoy 02
      datetime yesterday = current_ts - 86400;
      MqlDateTime yd;
      TimeToStruct(yesterday, yd);
      yd.hour = AsianRangeStartUTC; yd.min = 0; yd.sec = 0;
      window_start   = StructToTime(yd);
      window_is_open = false;
     }

   // Para el rolling: excluir la vela corriente (T-1); para el rango cerrado: incluir todo.
   // range_end es la primera barra que NO incluimos.
   datetime range_end;
   if(window_is_open)
     {
      // "hasta T-1 (barras cerradas antes de T)"
      // iTime retorna el open time de la barra. Una barra M15 con open=T cubre [T, T+15min).
      // Excluimos barras cuyo open >= current_ts para garantizar point-in-time.
      range_end = current_ts;
     }
   else
     {
      // Hasta las 02:00 UTC del dia actual
      MqlDateTime wd;
      TimeToStruct(current_ts, wd);
      wd.hour = AsianRangeEndUTC; wd.min = 0; wd.sec = 0;
      range_end = StructToTime(wd);
      // Si current_ts es antes de las 02:00, ajustar al dia siguiente (caso raro fuera de sesion)
      if(range_end <= window_start) range_end = window_start + (datetime)(AsianRangeEndUTC * 3600) + 86400;
     }

   double high_acc = -DBL_MAX;
   double low_acc  =  DBL_MAX;
   int    count    = 0;

   int total_m15 = Bars(_Symbol, PERIOD_M15);
   for(int i = 1; i < total_m15 && i < 200; i++)
     {
      datetime bar_time = iTime(_Symbol, PERIOD_M15, i);
      if(bar_time < window_start) break;       // ya estamos antes de la ventana
      if(bar_time >= range_end)   continue;    // barra posterior a range_end — ignorar
      double h_bar = iHigh(_Symbol, PERIOD_M15, i);
      double l_bar = iLow(_Symbol, PERIOD_M15, i);
      if(h_bar > high_acc) high_acc = h_bar;
      if(l_bar < low_acc)  low_acc  = l_bar;
      count++;
     }

   if(count == 0 || high_acc == -DBL_MAX)
      return false;

   asian_high = high_acc;
   asian_low  = low_acc;
   asian_mid  = (asian_high + asian_low) / 2.0;
   return true;
  }

//+------------------------------------------------------------------+
//|  INDICADORES                                                     |
//+------------------------------------------------------------------+

// Bollinger %B de la ultima vela M15 cerrada.
// Buffer 0=Middle, 1=Upper, 2=Lower (estandar MT5 iBands).
bool GetBollingerPctB(double &pct_b)
  {
   double upper[1], lower[1];
   if(CopyBuffer(g_bb_handle, 1, 1, 1, upper) != 1) return false;
   if(CopyBuffer(g_bb_handle, 2, 1, 1, lower) != 1) return false;
   double band_width = upper[0] - lower[0];
   if(band_width <= 0.0) return false;
   double close_m15 = iClose(_Symbol, PERIOD_M15, 1);
   pct_b = (close_m15 - lower[0]) / band_width;
   return true;
  }

// Cuenta velas M5 consecutivas alcistas (desde i=1 hacia atras).
int CountConsecBull_M5()
  {
   int count = 0;
   for(int i = 1; i < 15; i++)
     {
      if(iClose(_Symbol, PERIOD_M5, i) > iOpen(_Symbol, PERIOD_M5, i))
         count++;
      else
         break;
     }
   return count;
  }

// Cuenta velas M15 consecutivas bajistas (desde i=1 hacia atras).
int CountConsecBear_M15()
  {
   int count = 0;
   for(int i = 1; i < 15; i++)
     {
      if(iClose(_Symbol, PERIOD_M15, i) < iOpen(_Symbol, PERIOD_M15, i))
         count++;
      else
         break;
     }
   return count;
  }

//+------------------------------------------------------------------+
//|  CONVERSION DE PIPS A PRECIO                                     |
//+------------------------------------------------------------------+
// XAUUSD en MEX Atlantic: _Digits=2, _Point=0.01
// 1 pip = $0.10 = 10 puntos de precio (verificar con SymbolInfoDouble SYMBOL_POINT)
// Si el broker usa _Digits=3, ajustar a 1 pip = 100 _Point.
// ACCION REQUERIDA: confirmar con Print(_Digits, " ", _Point) en OnInit.
double PipsToPrice(double pips)
  {
   return pips * 10.0 * _Point;
  }

// Spread actual en "puntos" MT5 (SYMBOL_SPREAD).
int CurrentSpreadPts()
  {
   return (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
  }

//+------------------------------------------------------------------+
//|  EVALUACION DE SEÑAL                                             |
//+------------------------------------------------------------------+
ENUM_SIGNAL_TYPE EvaluateSignal(string &reason_out, double &asian_mid_out)
  {
   datetime now_utc = TimeGMT();
   MqlDateTime dt;
   TimeToStruct(now_utc, dt);

   // 1. Filtro de sesion
   if(!IsInCoreSession(dt.hour))
     {
      reason_out = "out_of_session";
      return SIGNAL_NONE;
     }

   // 2. Asian Range rolling
   double asian_high, asian_low, asian_mid;
   if(!ComputeAsianRange(now_utc, asian_high, asian_low, asian_mid))
     {
      reason_out = "asian_range_not_ready (sin barras previas en ventana)";
      return SIGNAL_NONE;
     }
   asian_mid_out = asian_mid;

   // 3. Distancia mid-precio
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double mid_price = (bid + ask) / 2.0;
   double dist_pips = MathAbs(mid_price - asian_mid) / (10.0 * _Point);

   if(dist_pips < MinDistFromAsianMidPips)
     {
      reason_out = StringFormat("too_close_to_mid (%.0f < %.0f pips)", dist_pips, MinDistFromAsianMidPips);
      return SIGNAL_NONE;
     }

   // 4. BB %B M15
   double pct_b;
   if(!GetBollingerPctB(pct_b))
     {
      reason_out = "bb_unavailable";
      return SIGNAL_NONE;
     }

   // 5. Velas consecutivas
   int cons_bull = CountConsecBull_M5();
   int cons_bear = CountConsecBear_M15();

   // 6. Decision direccional (MOMENTUM manda)
   bool is_buy  = (pct_b > BB_PctB_Buy_Threshold)  && (cons_bull >= MinConsecBullForBuy);
   bool is_sell = (pct_b < BB_PctB_Sell_Threshold) && (cons_bear >= MinConsecBearForSell);

   if(is_buy && is_sell)
     {
      reason_out = "conflict";
      return SIGNAL_NONE;
     }
   if(is_buy)
     {
      reason_out = StringFormat("BUY dist_mid=%.0f pips bb_pctb=%.2f cons_bull_m5=%d", dist_pips, pct_b, cons_bull);
      return SIGNAL_BUY;
     }
   if(is_sell)
     {
      reason_out = StringFormat("SELL dist_mid=%.0f pips bb_pctb=%.2f cons_bear_m15=%d", dist_pips, pct_b, cons_bear);
      return SIGNAL_SELL;
     }

   reason_out = StringFormat("no_signal bb_pctb=%.2f bull=%d bear=%d", pct_b, cons_bull, cons_bear);
   return SIGNAL_NONE;
  }

//+------------------------------------------------------------------+
//|  FILTROS OPERACIONALES                                           |
//+------------------------------------------------------------------+
bool CanExecuteSignal()
  {
   datetime now_utc = TimeGMT();
   datetime today   = GetSessionDate(now_utc);

   // Reset diario
   if(today != g_date_anchor)
     {
      g_date_anchor    = today;
      g_trades_today   = 0;
      g_last_signal_ts = 0;
     }

   // Max trades/dia
   if(UseMaxTradesPerDay && g_trades_today >= MaxTradesPerDay)
     {
      Print("Filtro: max trades/dia alcanzado (", g_trades_today, "/", MaxTradesPerDay, ")");
      return false;
     }

   // Cooldown
   if(g_last_signal_ts > 0)
     {
      int elapsed_min = (int)((now_utc - g_last_signal_ts) / 60);
      if(elapsed_min < CooldownMinutes)
        {
         Print("Filtro: cooldown (", elapsed_min, "/", CooldownMinutes, " min)");
         return false;
        }
     }

   // Spread maximo
   if(CurrentSpreadPts() > MaxSpreadPoints)
     {
      Print("Filtro: spread alto (", CurrentSpreadPts(), " > ", MaxSpreadPoints, " pts)");
      return false;
     }

   return true;
  }

//+------------------------------------------------------------------+
//|  EJECUCION DE ORDEN                                              |
//+------------------------------------------------------------------+
bool ExecuteOrder(ENUM_SIGNAL_TYPE sig, double price, double sl, double tp)
  {
   MqlTradeRequest req = {};
   MqlTradeResult  res = {};
   req.action   = TRADE_ACTION_DEAL;
   req.symbol   = _Symbol;
   req.volume   = VolumeLots;
   req.type     = (sig == SIGNAL_BUY) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   req.price    = price;
   req.sl       = NormalizeDouble(sl, _Digits);
   req.tp       = NormalizeDouble(tp, _Digits);
   req.deviation = SlippagePoints;
   req.magic    = 20260512;
   req.comment  = "AsianMomV3";
   if(!OrderSend(req, res))
     {
      PrintFormat("OrderSend FAILED retcode=%d: %s", res.retcode, res.comment);
      return false;
     }
   PrintFormat("Order OK ticket=%I64d price=%.2f sl=%.2f tp=%.2f", res.order, res.price, sl, tp);
   return true;
  }

//+------------------------------------------------------------------+
//|  LOGGING                                                         |
//+------------------------------------------------------------------+
string ModeStr()
  {
   switch(OperationMode)
     {
      case MODE_ALERT_ONLY: return "ALERT";
      case MODE_PAPER:      return "PAPER";
      case MODE_LIVE:       return "LIVE";
      default:              return "UNKNOWN";
     }
  }

void LogToCSV(string filename, string line)
  {
   int h = FileOpen(filename, FILE_WRITE | FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON, ';');
   if(h == INVALID_HANDLE)
     {
      Print("FileOpen FAILED: ", filename, " err=", GetLastError());
      return;
     }
   FileSeek(h, 0, SEEK_END);
   FileWriteString(h, line + "\n");
   FileClose(h);
  }

void LogSignal(datetime ts, string sig, double price, double sl, double tp, double asian_mid, string reason)
  {
   string line = StringFormat("%s;%s;%.2f;%.2f;%.2f;%.2f;%s;%s",
                              TimeToString(ts, TIME_DATE | TIME_MINUTES | TIME_SECONDS),
                              sig, price, sl, tp, asian_mid, ModeStr(), reason);
   LogToCSV(LogFilePrefix + "_signals.csv", line);
  }

void LogPaperTrade(datetime ts, string sig, double price, double sl, double tp)
  {
   string line = StringFormat("%s;%s;%.2f;%.2f;%.2f;%.2f",
                              TimeToString(ts, TIME_DATE | TIME_MINUTES | TIME_SECONDS),
                              sig, price, sl, tp, VolumeLots);
   LogToCSV(LogFilePrefix + "_paper_trades.csv", line);
  }

//+------------------------------------------------------------------+
//|  ACCION AL DETECTAR SEÑAL                                        |
//+------------------------------------------------------------------+
void OnSignalDetected(ENUM_SIGNAL_TYPE sig, string reason, double asian_mid)
  {
   string sig_str = (sig == SIGNAL_BUY) ? "BUY" : "SELL";
   double price   = (sig == SIGNAL_BUY)
                    ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                    : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl_pips = (sig == SIGNAL_BUY) ? SLPipsBuy  : SLPipsSell;
   double tp_pips = (sig == SIGNAL_BUY) ? TPPipsBuy  : TPPipsSell;
   double sl, tp;
   if(sig == SIGNAL_BUY)
     {
      sl = price - PipsToPrice(sl_pips);
      tp = price + PipsToPrice(tp_pips);
     }
   else
     {
      sl = price + PipsToPrice(sl_pips);
      tp = price - PipsToPrice(tp_pips);
     }

   datetime now_utc = TimeGMT();
   string msg = StringFormat("[%s] %s @ %.2f | SL %.2f | TP %.2f | asian_mid %.2f | %s",
                             ModeStr(), sig_str, price, sl, tp, asian_mid, reason);
   Print(msg);
   if(EnableSoundAlert) Alert(msg);
   if(EnablePushAlert)  SendNotification(msg);
   if(EnableEmailAlert) SendMail("[EA] Asian Momentum Signal", msg);

   LogSignal(now_utc, sig_str, price, sl, tp, asian_mid, reason);

   if(OperationMode == MODE_PAPER)
      LogPaperTrade(now_utc, sig_str, price, sl, tp);
   else if(OperationMode == MODE_LIVE)
      ExecuteOrder(sig, price, sl, tp);

   // Actualizar estado operacional
   g_last_signal_ts = now_utc;
   g_trades_today++;
  }

//+------------------------------------------------------------------+
//|  CICLO PRINCIPAL                                                 |
//+------------------------------------------------------------------+
int OnInit()
  {
   PrintFormat("=== XAUUSD Asian Momentum v3 INIT | Modo=%s | Vol=%.2f lots ===",
               ModeStr(), VolumeLots);
   PrintFormat("Sesion core %dh-%dh UTC | Max trades/dia=%d | Cooldown=%d min",
               SessionStartHourUTC, SessionEndHourUTC, MaxTradesPerDay, CooldownMinutes);
   PrintFormat("_Symbol=%s _Digits=%d _Point=%.5f", _Symbol, _Digits, _Point);
   PrintFormat("PipsToPrice(1 pip)=%.5f — verificar que esto es ~0.10 para XAUUSD",
               PipsToPrice(1.0));

   // Verificacion critica: PipsToPrice debe dar 0.10 si _Digits=2 (_Point=0.01)
   double test_pip = PipsToPrice(1.0);
   Print(StringFormat("[CHECK] PipsToPrice(1 pip) = %.5f  (esperado: 0.10000)", test_pip));
   Print(StringFormat("[CHECK] _Digits = %d  (esperado: 2)", _Digits));
   Print(StringFormat("[CHECK] _Point  = %.5f  (esperado: 0.01000)", _Point));
   if(MathAbs(test_pip - 0.10) > 0.0001)
     {
      Print("WARNING: PipsToPrice incorrecto. SL/TP seran erroneos. ABORTANDO.");
      return INIT_FAILED;
     }

   g_bb_handle = iBands(_Symbol, PERIOD_M15, BB_Period, 0, BB_Deviation, PRICE_CLOSE);
   if(g_bb_handle == INVALID_HANDLE)
     {
      Print("INIT FAILED: no se pudo crear handle de Bollinger Bands");
      return INIT_FAILED;
     }

   g_date_anchor    = 0;
   g_trades_today   = 0;
   g_last_signal_ts = 0;

   // Crear headers de CSV si no existen
   string sig_csv = LogFilePrefix + "_signals.csv";
   if(!FileIsExist(sig_csv, FILE_COMMON))
     {
      int h = FileOpen(sig_csv, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ';');
      if(h != INVALID_HANDLE)
        {
         FileWriteString(h, "timestamp_utc;signal;price;sl;tp;asian_mid;mode;reason\n");
         FileClose(h);
        }
     }
   string paper_csv = LogFilePrefix + "_paper_trades.csv";
   if(!FileIsExist(paper_csv, FILE_COMMON))
     {
      int h = FileOpen(paper_csv, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ';');
      if(h != INVALID_HANDLE)
        {
         FileWriteString(h, "timestamp_utc;signal;price;sl;tp;lots\n");
         FileClose(h);
        }
     }

   Print("Init OK — esperando señales en sesion asiatica UTC...");
   return INIT_SUCCEEDED;
  }

void OnTick()
  {
   // Ejecutar solo al inicio de cada vela M1 nueva (reducir carga)
   static datetime s_last_m1 = 0;
   datetime cur_m1 = iTime(_Symbol, PERIOD_M1, 0);
   if(cur_m1 == s_last_m1) return;
   s_last_m1 = cur_m1;

   string reason;
   double asian_mid = 0.0;
   ENUM_SIGNAL_TYPE sig = EvaluateSignal(reason, asian_mid);

   if(sig == SIGNAL_NONE) return;

   if(!CanExecuteSignal()) return;

   OnSignalDetected(sig, reason, asian_mid);
  }

void OnDeinit(const int reason)
  {
   if(g_bb_handle != INVALID_HANDLE)
      IndicatorRelease(g_bb_handle);
   PrintFormat("=== EA desactivado | razon=%d ===", reason);
  }
//+------------------------------------------------------------------+
