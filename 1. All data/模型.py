#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:19:39 2025

@author: panyuyan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== 轻量工具：列名宽松匹配 ======
FLOW_COLS  = ["Flow per hour","Flow/hour","Flow_per_hour","Flow","Observed_Flow","ObservedFlow",
              "Volume","Flow (veh/h)","flow per hour","flow","volume","FLOW"]
SPEED_COLS = ["Speed","speed","SPEED"]
DENS_COLS  = ["Density","density","DENSITY","K","k"]
TIME_COLS  = ["time","Time","TIME","timestamp","Timestamp"]
QUEUE_COLS = ["Queue","queue","QUEUE"]

def soft_pick(cols, candidates):
    m = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in m: return m[c.lower()]
    low = [c.lower() for c in cols]
    for i,n in enumerate(low):
        if "flow" in n or "volume" in n:
            return cols[i]
    return None

def load_excel(path):
    df = pd.read_excel(path)
    cols = df.columns.tolist()
    flow  = soft_pick(cols, FLOW_COLS)
    speed = soft_pick(cols, SPEED_COLS)
    dens  = soft_pick(cols, DENS_COLS)
    timec = soft_pick(cols, TIME_COLS)
    queue = soft_pick(cols, QUEUE_COLS)
    return df, {"flow":flow,"speed":speed,"dens":dens,"time":timec,"queue":queue}

# ====== 拥堵窗口检测（速度法） ======
def detect_t0_t3_by_speed(df, speed_col, drop_ratio=0.7, smooth_win=3):
    s = df[speed_col].astype(float).rolling(smooth_win, min_periods=1).mean().values
    v_free = np.nanpercentile(s, 85)
    thresh = v_free * drop_ratio
    below = s < thresh
    if not below.any():
        return 0, len(df) - 1
    idx = np.where(below)[0]
    return int(idx[0]), int(idx[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="PeMS Excel path")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--delta_min", type=float, default=5.0)
    ap.add_argument("--speed_drop_ratio", type=float, default=0.7)
    ap.add_argument("--smooth_win", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "figs"), exist_ok=True)

    df, cols = load_excel(args.excel)
    flow_col, speed_col = cols["flow"], cols["speed"]
    if flow_col is None or speed_col is None:
        raise ValueError("Flow/Speed 列未找到，请检查 Excel 表头。")

    df = df.dropna(subset=[flow_col, speed_col]).reset_index(drop=True)
    q = df[flow_col].astype(float).values  # veh/h
    dt_h = args.delta_min / 60.0
    A = np.cumsum(q * dt_h)     # 累计到达（单点示例作为 proxy）
    Dp = np.cumsum(q * dt_h)    # 累计离开（单点示例作为 proxy；若有下游点请替换）

    t0, t3 = detect_t0_t3_by_speed(df, speed_col, args.speed_drop_ratio, args.smooth_win)
    P = (t3 - t0 + 1) * dt_h                              # 拥堵持续时间（小时）
    mu = float(np.mean(q[t0:t3+1]) if t3 > t0 else np.mean(q))   # 拥堵期服务率 μ（veh/h）
    D  = float(np.sum(q[t0:t3+1] * dt_h))                 # 拥堵窗内需求 D（veh）

    meta = {
        "flow_col": flow_col, "speed_col": speed_col, "dens_col": cols["dens"],
        "time_col": cols["time"], "queue_col": cols["queue"],
        "delta_min": args.delta_min, "t0_idx": int(t0), "t3_idx": int(t3),
        "P_hour": float(P), "mu_veh_per_h": float(mu), "D_veh": float(D)
    }
    with open(os.path.join(args.outdir, "tables", "dc_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    plt.figure()
    plt.plot(A, label="Cumulative Arrivals (A) proxy")
    plt.plot(Dp, label="Cumulative Departures (Dp) proxy")
    plt.axvline(t0, linestyle="--", label="t0")
    plt.axvline(t3, linestyle="--", label="t3")
    plt.title("Cumulative curves & detected congestion window")
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "figs", "01_cum_curves.png"), dpi=150, bbox_inches="tight")

    print("Saved:", os.path.join(args.outdir, "tables", "dc_meta.json"))
    print("Saved:", os.path.join(args.outdir, "figs", "01_cum_curves.png"))

if __name__ == "__main__":
    main()
