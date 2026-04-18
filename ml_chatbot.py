
# ─────────────────────────────────────────────────────────────────────────────
#  AI Data Assistant  —  v13  (Obsidian Glass UI)
#  Refined dark theme · layered glass surfaces · typing animations · premium UX
#  Backend ML logic preserved — UI layer completely redesigned
# ─────────────────────────────────────────────────────────────────────────────
import datetime, json, re, threading, uuid, warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import seaborn as sns

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               AdaBoostClassifier, AdaBoostRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.tree  import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm   import SVC, SVR
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score,
                              mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


# ══════════════════════  OBSIDIAN GLASS DESIGN SYSTEM  ═══════════════════════
#
#  Surface depth model (simulated glass on #0B0F19):
#    Level 0: bg         #0B0F19   (deepest — main background)
#    Level 1: surface1   #111827   (sidebar, panels ~4% white)
#    Level 2: surface2   #1E2433   (cards, bubbles ~8% white)
#    Level 3: surface3   #252B3B   (elevated cards, hover ~12% white)
#
#  Glass border:  #FFFFFF with ~6% opacity = #1F2937 on dark base
#  Glass highlight: top-edge 1px line at #FFFFFF0A ≈ #1A2030
#
C = {
    # ── depth surfaces ──
    "bg":         "#0B0F19",
    "bg2":        "#0E1322",
    "surface1":   "#111827",
    "surface2":   "#1E2433",
    "surface3":   "#252B3B",
    # ── borders ──
    "border":     "#1F2937",
    "borderLt":   "#374151",
    "borderGlass":"#2A3347",
    # ── accent (single blue) ──
    "accent":     "#3B82F6",
    "accentHi":   "#60A5FA",
    "accentDk":   "#2563EB",
    "accentMute": "#1E3A5F",
    # ── semantic ──
    "success":    "#10B981",
    "warning":    "#F59E0B",
    "error":      "#EF4444",
    "info":       "#06B6D4",
    # ── text ──
    "text":       "#E2E8F0",
    "textSec":    "#94A3B8",
    "textMuted":  "#64748B",
    "white":      "#FFFFFF",
    # ── chat ──
    "userBubble": "#1A325A",
    "botBubble":  "#151C2C",
    # ── input ──
    "inputBg":    "#0E1322",
    "inputBorder":"#1F2937",
}

# chart palette — distinguishable on dark background
CP = ["#3B82F6", "#06B6D4", "#10B981", "#F59E0B",
      "#F43F5E", "#8B5CF6", "#34D399", "#FBBF24"]

# ── chart styling ─────────────────────────────────────────────────────────────
CH_BG   = "#111827"
CH_AX   = "#151C2C"
CH_TEXT = "#E2E8F0"
CH_GRID = "#1F2937"
CH_TICK = 10
CH_LBL  = 11
CH_TTL  = 12
CH_LW   = 2.2

def _ax_style(ax, title=""):
    ax.set_facecolor(CH_AX)
    ax.tick_params(colors=CH_TEXT, labelsize=CH_TICK, length=3, width=0.8)
    ax.xaxis.label.set_color(CH_TEXT); ax.xaxis.label.set_fontsize(CH_LBL)
    ax.yaxis.label.set_color(CH_TEXT); ax.yaxis.label.set_fontsize(CH_LBL)
    ax.set_title(title, color=CH_TEXT, fontsize=CH_TTL, pad=14, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor(CH_GRID); sp.set_linewidth(0.8)
    ax.grid(color=CH_GRID, linestyle="--", linewidth=0.6, alpha=0.6)

def _embed(fig, parent, pady=6):
    c = FigureCanvasTkAgg(fig, master=parent)
    c.draw()
    w = c.get_tk_widget()
    w.configure(bg=CH_BG, highlightthickness=0)
    w.pack(fill=tk.BOTH, expand=True, padx=12, pady=pady)
    return c


# ══════════════════════  ML MODELS  ══════════════════════════════════════════
MODELS = {
    "1":{"name":"Random Forest",     "clf":lambda:RandomForestClassifier(n_estimators=80,max_depth=18,min_samples_leaf=2,n_jobs=-1,random_state=42),"reg":lambda:RandomForestRegressor(n_estimators=80,max_depth=18,min_samples_leaf=2,n_jobs=-1,random_state=42),"desc":"Great all-rounder"},
    "2":{"name":"Decision Tree",     "clf":lambda:DecisionTreeClassifier(max_depth=12,random_state=42),"reg":lambda:DecisionTreeRegressor(max_depth=12,random_state=42),"desc":"Simple & fast"},
    "3":{"name":"Gradient Boosting", "clf":lambda:GradientBoostingClassifier(n_estimators=80,random_state=42),"reg":lambda:GradientBoostingRegressor(n_estimators=80,random_state=42),"desc":"High accuracy"},
    "4":{"name":"Extra Trees",       "clf":lambda:ExtraTreesClassifier(n_estimators=80,n_jobs=-1,random_state=42),"reg":lambda:ExtraTreesRegressor(n_estimators=80,n_jobs=-1,random_state=42),"desc":"Faster ensemble"},
    "5":{"name":"AdaBoost",          "clf":lambda:AdaBoostClassifier(n_estimators=60,random_state=42),"reg":lambda:AdaBoostRegressor(n_estimators=60,random_state=42),"desc":"Boosting model"},
    "6":{"name":"KNN",               "clf":lambda:KNeighborsClassifier(n_neighbors=5,n_jobs=-1),"reg":lambda:KNeighborsRegressor(n_neighbors=5,n_jobs=-1),"desc":"Similarity based"},
    "7":{"name":"Logistic / Ridge",  "clf":lambda:LogisticRegression(max_iter=500,n_jobs=-1,random_state=42),"reg":lambda:Ridge(random_state=42),"desc":"Fast linear model"},
    "8":{"name":"SVM",               "clf":lambda:SVC(probability=True,random_state=42),"reg":lambda:SVR(),"desc":"Small clean data"},
}
ALIASES = {"random forest":"1","rf":"1","forest":"1","decision tree":"2","tree":"2","gradient boosting":"3","gb":"3","boosting":"3","extra trees":"4","et":"4","adaboost":"5","ada":"5","knn":"6","logistic":"7","ridge":"7","svm":"8","support vector":"8"}
SKIPS   = {"skip","dont know","don't know","idk","not sure","na","n/a","none","unknown","?","pass","ignore","empty","no idea"}


# ══════════════════════  SESSION  ════════════════════════════════════════════
class Session:
    def __init__(self):
        self.id=str(uuid.uuid4()); self.dataset=None; self.profile={}
        self.llm_summary=""; self.ollama_model="qwen2.5"
        self.task_type=None; self.target_column=None
        self.selected_feats=[]; self.ask_feats=[]; self.dropped_feats=[]
        self.null_strategy="fill"; self.chosen_model="1"
        self.model_bundle={}; self.metrics={}; self.feat_importance=[]
        self.col_defaults={}; self.stage="idle"; self.awaiting_feat=None
        self.pred_inputs={}; self.pred_result={}
        self.candidate_tgts=[]; self.id_cols=[]; self.note=""; self.top_k=5


# ══════════════════════  ML ENGINE  ══════════════════════════════════════════
class MLEngine:
    MAX=12000
    def train(self,df,target,features,null_strat,task_type,model_key="1"):
        data=df[features+[target]].copy()
        if null_strat=="drop": data=data.dropna()
        if len(data)<10: raise ValueError("Not enough rows (need ≥ 10).")
        if len(data)>self.MAX: data=data.sample(self.MAX,random_state=42)
        X=data[features].copy(); y=data[target].copy()
        cat=[c for c in features if X[c].dtype=="object" or str(X[c].dtype).startswith("category")]
        num=[c for c in features if c not in cat]
        for c in cat:
            X[c]=X[c].astype(str); keep=set(X[c].value_counts().head(30).index)
            X[c]=X[c].apply(lambda v:v if v in keep else "__OTHER__")
        pre=ColumnTransformer([
            ("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),num),
            ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                             ("enc",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]),cat)
        ])
        m=MODELS[model_key]; le=None
        if task_type=="classification":
            if y.dtype=="object" or str(y.dtype).startswith("category"):
                le=LabelEncoder(); y=le.fit_transform(y.astype(str))
            clf=m["clf"]()
        else:
            y=pd.to_numeric(y,errors="coerce"); mask=~pd.isna(y)
            X,y=X.loc[mask],y.loc[mask]; clf=m["reg"]()
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
        pipe=Pipeline([("pre",pre),("clf",clf)]); pipe.fit(Xtr,ytr); preds=pipe.predict(Xte)
        if task_type=="classification":
            metrics={"Accuracy":round(float(accuracy_score(yte,preds)),4),
                     "F1":round(float(f1_score(yte,preds,average="weighted")),4)}
        else:
            metrics={"MAE":round(float(mean_absolute_error(yte,preds)),4),
                     "RMSE":round(float(np.sqrt(mean_squared_error(yte,preds))),4),
                     "R2":round(float(r2_score(yte,preds)),4)}
        names=pipe.named_steps["pre"].get_feature_names_out()
        imps=getattr(pipe.named_steps["clf"],"feature_importances_",np.ones(len(names))/len(names))
        # ── map sklearn prefixed names back to ORIGINAL column names ──────────
        orig_imp = {}
        for sk_name, imp_val in zip(names, imps):
            raw = sk_name.split("__", 1)[1] if "__" in sk_name else sk_name
            raw_key = re.sub(r"[^a-z0-9]", "", raw.lower())
            matched = None
            for c in features:
                if c == raw:
                    matched = c; break
            if not matched:
                for c in features:
                    if c.lower() == raw.lower():
                        matched = c; break
            if not matched:
                for c in features:
                    if re.sub(r"[^a-z0-9]", "", c.lower()) == raw_key:
                        matched = c; break
            if not matched:
                matched = raw
            orig_imp[matched] = orig_imp.get(matched, 0.0) + float(imp_val)
        fi = sorted([{"feature": k, "importance": v} for k, v in orig_imp.items()],
                    key=lambda x: x["importance"], reverse=True)[:10]
        return {"pipeline":pipe,"metrics":metrics,"fi":fi,"le":le,"task":task_type,
                "features":features,"target":target,"model_name":m["name"]}
    def predict(self,bundle,inputs):
        X=pd.DataFrame([inputs],columns=bundle["features"])
        pred=bundle["pipeline"].predict(X)[0]; prob=None
        if bundle["task"]=="classification":
            if hasattr(bundle["pipeline"].named_steps["clf"],"predict_proba"):
                prob=[round(float(v),4) for v in bundle["pipeline"].predict_proba(X)[0]]
            if bundle["le"] is not None: pred=bundle["le"].inverse_transform([int(pred)])[0]
        return {"prediction":pred,"probabilities":prob}


# ══════════════════════  OLLAMA  ═════════════════════════════════════════════
class Ollama:
    URL="http://localhost:11434"
    def ok(self):
        try: return requests.get(f"{self.URL}/api/tags",timeout=3).ok
        except: return False
    def ask(self,model,prompt):
        r=requests.post(f"{self.URL}/api/generate",json={"model":model,"prompt":prompt,"stream":False},timeout=120)
        r.raise_for_status(); return r.json().get("response","").strip()
    def summarize(self,model,profile):
        snap={"rows":profile["rows"],"columns":profile["columns"],"col_names":profile["col_names"][:20],
              "targets":profile["tgt_cols"][:5],"samples":{k:v for k,v in list(profile["samples"].items())[:8]}}
        return self.ask(model,"Summarize this dataset in 2-3 plain sentences. What is it about? Best column to predict?\n\n"+json.dumps(snap)[:2000])
    def explain(self,model,ctx):
        p=(f"Friendly ML explanation (≤220 words, no jargon):\n"
           f"Target:{ctx['target']} Predicted:{ctx['prediction']} Model:{ctx['model_name']}\n"
           f"Quality:{ctx['metrics']} TopFeatures:{ctx['top_features']}\n"
           f"UserGave:{ctx['user_inputs']} AutoFilled:{ctx['auto_filled']}\n"
           f"Explain:1)What predicted 2)Why(mention features) 3)How reliable 4)Auto-fill impact 5)Limits.")
        return self.ask(model,p[:3000])


# ══════════════════════  CUSTOM UI WIDGETS  ══════════════════════════════════

class TypingDots(tk.Frame):
    """Animated 3-dot typing indicator shown while AI is thinking."""
    def __init__(self, parent):
        super().__init__(parent, bg=C["botBubble"], padx=0, pady=0)
        self._phase = 0
        self._active = True
        self.dots = []
        container = tk.Frame(self, bg=C["botBubble"], padx=16, pady=10)
        container.pack()
        for i in range(3):
            dot = tk.Label(container, text="●", bg=C["botBubble"],
                          fg=C["textMuted"], font=("Segoe UI", 9))
            dot.pack(side=tk.LEFT, padx=2)
            self.dots.append(dot)
        self._animate()

    def _animate(self):
        if not self._active:
            return
        for i, dot in enumerate(self.dots):
            dot.config(fg=C["accent"] if i == self._phase else C["textMuted"])
        self._phase = (self._phase + 1) % 3
        self._anim_id = self.after(350, self._animate)

    def stop(self):
        self._active = False
        if hasattr(self, '_anim_id'):
            self.after_cancel(self._anim_id)


class GlassCard(tk.Frame):
    """Card container with glass-like surface styling."""
    def __init__(self, parent, title=None, title_icon=None, **kwargs):
        super().__init__(parent, bg=C["surface2"],
                        highlightbackground=C["borderGlass"],
                        highlightthickness=1, **kwargs)
        if title:
            hdr = tk.Frame(self, bg=C["surface2"])
            hdr.pack(fill=tk.X, padx=16, pady=(12, 4))
            display = f"{title_icon}  {title}" if title_icon else title
            tk.Label(hdr, text=display, bg=C["surface2"], fg=C["text"],
                    font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
            # subtle separator line
            tk.Frame(self, bg=C["border"], height=1).pack(fill=tk.X, padx=16, pady=(4, 0))


class SessionCard(tk.Frame):
    """Compact card for session info display."""
    def __init__(self, parent, title, icon, **kwargs):
        super().__init__(parent, bg=C["surface2"],
                        highlightbackground=C["borderGlass"],
                        highlightthickness=1, **kwargs)
        hdr = tk.Frame(self, bg=C["surface2"])
        hdr.pack(fill=tk.X, padx=12, pady=(10, 2))
        tk.Label(hdr, text=f"{icon}  {title}", bg=C["surface2"],
                fg=C["textMuted"], font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
        self.value_frame = tk.Frame(self, bg=C["surface2"])
        self.value_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

    def set_value(self, text, fg=None):
        for w in self.value_frame.winfo_children():
            w.destroy()
        tk.Label(self.value_frame, text=text, bg=C["surface2"],
                fg=fg or C["text"], font=("Segoe UI", 11, "bold"),
                anchor="w").pack(side=tk.LEFT)


class StatusPill(tk.Frame):
    """Small pill showing status with colored dot."""
    def __init__(self, parent, text="Ready", color=None):
        super().__init__(parent, bg=C["surface1"], padx=10, pady=4,
                        highlightbackground=C["border"], highlightthickness=1)
        self.dot = tk.Label(self, text="●", bg=C["surface1"],
                           fg=color or C["success"], font=("Segoe UI", 7))
        self.dot.pack(side=tk.LEFT, padx=(0, 6))
        self.label = tk.Label(self, text=text, bg=C["surface1"],
                             fg=C["textSec"], font=("Segoe UI", 8))
        self.label.pack(side=tk.LEFT)

    def update_status(self, text, color=None):
        self.label.config(text=text)
        if color:
            self.dot.config(fg=color)


# ══════════════════════  GRAPHS PANEL (redesigned)  ══════════════════════════
class GraphsPanel(tk.Frame):
    """
    Redesigned graph panel — charts wrapped in glass cards with section labels.
    Scrollable. Charts render after prediction completes.
    """
    def __init__(self, master):
        super().__init__(master, bg=C["bg"])
        self.graph_data = None
        self._canvases = []
        self._build_shell()

    def _build_shell(self):
        # ── header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C["surface1"], height=52)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="📊", bg=C["surface1"], fg=C["accent"],
                font=("Segoe UI", 16)).pack(side=tk.LEFT, padx=(20, 8), pady=10)
        tk.Label(hdr, text="Analytics Dashboard",
                bg=C["surface1"], fg=C["text"],
                font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, pady=10)
        self.hdr_sub = tk.Label(hdr, text="Complete a prediction to view charts",
                                bg=C["surface1"], fg=C["textMuted"],
                                font=("Segoe UI", 9))
        self.hdr_sub.pack(side=tk.LEFT, padx=16)
        # glass border at bottom of header
        tk.Frame(self, bg=C["borderGlass"], height=1).pack(fill=tk.X)

        # ── scrollable area ───────────────────────────────────────────────────
        outer = tk.Frame(self, bg=C["bg"])
        outer.pack(fill=tk.BOTH, expand=True)
        self.scroll_canvas = tk.Canvas(outer, bg=C["bg"], highlightthickness=0)
        sb = tk.Scrollbar(outer, orient=tk.VERTICAL,
                          command=self.scroll_canvas.yview,
                          bg=C["surface1"], troughcolor=C["bg"],
                          activebackground=C["accent"], width=8)
        self.scroll_canvas.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = tk.Frame(self.scroll_canvas, bg=C["bg"])
        self._win = self.scroll_canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.scroll_canvas.bind("<Configure>", self._on_canvas_configure)
        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # ── empty state placeholder ───────────────────────────────────────────
        self.placeholder = tk.Frame(self.inner, bg=C["bg"])
        self.placeholder.pack(expand=True, pady=100)
        tk.Label(self.placeholder, text="📊", bg=C["bg"], fg=C["textMuted"],
                font=("Segoe UI", 48)).pack(pady=(0, 16))
        tk.Label(self.placeholder, text="No Data Yet",
                bg=C["bg"], fg=C["text"], font=("Segoe UI", 16, "bold")).pack()
        tk.Label(self.placeholder,
                text="Complete a prediction session, then switch\nto this tab to view all charts and statistics.",
                bg=C["bg"], fg=C["textMuted"], font=("Segoe UI", 11),
                justify="center").pack(pady=(8, 0))

    def _on_inner_configure(self, e):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self.scroll_canvas.itemconfig(self._win, width=e.width)

    def _on_mousewheel(self, e):
        self.scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def load(self, df, target, fi, task, metrics, model_name, pred_result):
        """Called after prediction — stores data and renders charts."""
        self.graph_data = dict(df=df, target=target, fi=fi, task=task,
                               metrics=metrics, model_name=model_name,
                               pred_result=pred_result)
        pred = pred_result.get("prediction", "—")
        try:
            disp = f"{float(pred):,.2f}" if task == "regression" else str(pred)
        except:
            disp = str(pred)
        self.hdr_sub.config(
            text=f"Target: {target}   ·   Predicted: {disp}   ·   Model: {model_name}",
            fg=C["textSec"])
        self._draw()

    def _draw(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self._canvases.clear()
        d = self.graph_data
        self._section_fi(d["fi"])
        self._section_dist(d["df"], d["target"], d["task"])
        self._section_trends(d["df"], d["target"])
        self._section_stats(d["df"])

    # ── section label ─────────────────────────────────────────────────────────
    def _sec_label(self, icon, title, subtitle=""):
        row = tk.Frame(self.inner, bg=C["bg"])
        row.pack(fill=tk.X, padx=16, pady=(24, 8))
        tk.Label(row, text=icon, bg=C["bg"], fg=C["accent"],
                font=("Segoe UI", 14)).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text=title, bg=C["bg"], fg=C["text"],
                font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        if subtitle:
            tk.Label(row, text=f"  —  {subtitle}", bg=C["bg"], fg=C["textMuted"],
                    font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)
        tk.Frame(self.inner, bg=C["border"], height=1).pack(fill=tk.X, padx=16)

    def _card(self, h=None):
        f = tk.Frame(self.inner, bg=C["surface1"], bd=0,
                     highlightbackground=C["borderGlass"], highlightthickness=1)
        f.pack(fill=tk.X, padx=16, pady=8)
        if h:
            f.configure(height=h)
        return f

    # ── Chart 1: Feature Importance (seaborn barplot) ─────────────────────────
    def _section_fi(self, fi):
        self._sec_label("📈", "Feature Importance", "columns driving the prediction")
        card = self._card()
        if not fi:
            tk.Label(card, text="No feature importance data available.",
                     bg=C["surface1"], fg=C["textMuted"],
                     font=("Segoe UI", 10)).pack(pady=24)
            return
        items = fi[:8]
        fi_df = pd.DataFrame({
            "Feature": [x["feature"].split("__")[-1][:26] for x in items][::-1],
            "Importance": [x["importance"] for x in items][::-1]
        })
        palette = [CP[i % len(CP)] for i in range(len(fi_df))]
        fig = Figure(figsize=(13, max(3.6, len(fi_df) * 0.55 + 1)), facecolor=CH_BG)
        ax = fig.add_subplot(111)
        _ax_style(ax, "Feature Importance Score")
        sns.barplot(data=fi_df, y="Feature", x="Importance", ax=ax,
                    palette=palette, orient="h", edgecolor="none", width=0.6)
        for i, (val, name) in enumerate(zip(fi_df["Importance"], fi_df["Feature"])):
            ax.text(val + 0.004, i, f"{val:.4f}", va="center", ha="left",
                    color=CH_TEXT, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=CH_LBL)
        ax.set_ylabel("", fontsize=CH_LBL)
        ax.set_xlim(0, fi_df["Importance"].max() * 1.32)
        fig.tight_layout(pad=1.8)
        c = _embed(fig, card)
        self._canvases.append(c)

    # ── Chart 2: Distributions (seaborn histplot / countplot + kdeplot) ────────
    def _section_dist(self, df, target, task):
        self._sec_label("📊", "Distributions", "target spread & numeric feature ranges")
        card = self._card()
        col      = df[target].dropna()
        num_cols = [c for c in df.columns
                    if c != target and str(df[c].dtype) in ("int64", "float64")][:5]
        fig = Figure(figsize=(13, 5.2), facecolor=CH_BG)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        if task == "classification" or col.nunique() <= 20:
            count_df = col.astype(str).value_counts().head(12).reset_index()
            count_df.columns = [target, "count"]
            sns.barplot(data=count_df, x=target, y="count", ax=ax1,
                        color=CP[0], edgecolor="none", width=0.65)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=35, ha="right", fontsize=9)
            ax1.set_ylabel("Count", fontsize=CH_LBL)
            ax1.set_xlabel("", fontsize=CH_LBL)
        else:
            sns.histplot(col.astype(float), bins=30, ax=ax1, color=CP[0],
                         kde=True, alpha=0.8, edgecolor=CH_BG, linewidth=0.4,
                         line_kws={"linewidth": CH_LW, "color": CP[3]})
            ax1.set_xlabel(target, fontsize=CH_LBL)
            ax1.set_ylabel("Frequency", fontsize=CH_LBL)
        _ax_style(ax1, f"Target  '{target}'  Distribution")
        if num_cols:
            for i, c in enumerate(num_cols):
                d = df[c].dropna().astype(float)
                if d.std() > 0:
                    sns.kdeplot(d, ax=ax2, label=c, color=CP[i % len(CP)],
                                linewidth=CH_LW, fill=True, alpha=0.15)
            ax2.set_xlabel("Value", fontsize=CH_LBL)
            ax2.set_ylabel("Density", fontsize=CH_LBL)
            leg = ax2.legend(fontsize=9, facecolor=C["surface2"],
                             edgecolor=C["border"], labelcolor=CH_TEXT, framealpha=0.9)
        else:
            ax2.set_visible(False)
        _ax_style(ax2, "Numeric Features  —  Density Curves")
        fig.tight_layout(pad=1.8)
        c = _embed(fig, card)
        self._canvases.append(c)

    # ── Chart 3: Trends (seaborn regplot + lineplot) ──────────────────────────
    def _section_trends(self, df, target):
        self._sec_label("📉", "Trends", "relationships between top numeric features")
        card = self._card()
        num_cols = [c for c in df.columns
                    if c != target and str(df[c].dtype) in ("int64", "float64")][:4]
        tgt_num  = str(df[target].dtype) in ("int64", "float64")
        fig = Figure(figsize=(13, 5.2), facecolor=CH_BG)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        if num_cols and tgt_num:
            best = num_cols[0]
            sdf  = df[[best, target]].dropna().sample(min(500, len(df)), random_state=42)
            try:
                sns.regplot(data=sdf, x=best, y=target, ax=ax1,
                            scatter_kws={"alpha": 0.45, "color": CP[1],
                                         "edgecolors": "none", "s": 24},
                            line_kws={"color": CP[3], "linewidth": CH_LW + 0.4,
                                      "linestyle": "--"},
                            ci=95, color=CP[1])
            except Exception:
                ax1.scatter(sdf[best].astype(float), sdf[target].astype(float),
                            alpha=0.45, color=CP[1], edgecolors="none", s=24)
            ax1.set_xlabel(best, fontsize=CH_LBL)
            ax1.set_ylabel(target, fontsize=CH_LBL)
            _ax_style(ax1, f"Regression:  {best}  vs  {target}")
        else:
            ax1.set_visible(False)
        if len(num_cols) >= 2:
            c1, c2 = num_cols[0], num_cols[1]
            sdf2 = df[[c1, c2]].dropna().sort_values(c1).head(300)
            sns.lineplot(data=sdf2, x=c1, y=c2, ax=ax2,
                         color=CP[2], linewidth=CH_LW)
            ax2.fill_between(sdf2[c1].values, sdf2[c2].values,
                             alpha=0.12, color=CP[2])
            ax2.set_xlabel(c1, fontsize=CH_LBL)
            ax2.set_ylabel(c2, fontsize=CH_LBL)
            _ax_style(ax2, f"Line Trend:  {c1}  →  {c2}")
        elif num_cols:
            c1 = num_cols[0]
            d = df[c1].dropna().astype(float).head(300).reset_index(drop=True)
            line_df = pd.DataFrame({"Index": np.arange(len(d)), c1: d.values})
            sns.lineplot(data=line_df, x="Index", y=c1, ax=ax2,
                         color=CP[2], linewidth=CH_LW)
            ax2.fill_between(line_df["Index"].values, line_df[c1].values,
                             alpha=0.12, color=CP[2])
            ax2.set_xlabel("Row index", fontsize=CH_LBL)
            ax2.set_ylabel(c1, fontsize=CH_LBL)
            _ax_style(ax2, f"Line Trend:  {c1}")
        else:
            ax2.set_visible(False)
        fig.tight_layout(pad=1.8)
        c = _embed(fig, card)
        self._canvases.append(c)

    # ── Section 4: Stats table ────────────────────────────────────────────────
    def _section_stats(self, df):
        self._sec_label("🗂️", "Statistical Summary", "pandas describe()")
        card = self._card(h=280)
        try:
            desc = df.describe(include="all").round(4).fillna("—")
        except:
            desc = df.describe().round(4)
        cols = list(desc.columns)

        frame = tk.Frame(card, bg=C["surface1"])
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        sy = tk.Scrollbar(frame, orient=tk.VERTICAL,   bg=C["surface1"],
                          troughcolor=C["surface1"], activebackground=C["accent"], width=7)
        sx = tk.Scrollbar(frame, orient=tk.HORIZONTAL, bg=C["surface1"],
                          troughcolor=C["surface1"], activebackground=C["accent"], width=7)
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")

        sty = ttk.Style()
        sty.configure("OG.Treeview",
                      background=C["surface1"], foreground=C["text"],
                      fieldbackground=C["surface1"], font=("Consolas", 9), rowheight=26)
        sty.configure("OG.Treeview.Heading",
                      background=C["surface2"], foreground=C["accent"],
                      font=("Segoe UI", 9, "bold"), relief="flat")
        sty.map("OG.Treeview",
                background=[("selected", C["accentDk"])],
                foreground=[("selected", C["white"])])

        tv = ttk.Treeview(frame, style="OG.Treeview",
                          columns=["Stat"] + cols, show="headings",
                          yscrollcommand=sy.set, xscrollcommand=sx.set)
        tv.grid(row=0, column=0, sticky="nsew")
        sy.config(command=tv.yview)
        sx.config(command=tv.xview)

        tv.heading("Stat", text="Statistic")
        tv.column("Stat", width=90, anchor="w", stretch=False)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=110, anchor="center", stretch=False)

        tv.tag_configure("even", background=C["surface1"])
        tv.tag_configure("odd",  background=C["surface2"])
        for i, (idx, row) in enumerate(desc.iterrows()):
            vals = [str(idx)] + [str(v) for v in row.values]
            tv.insert("", "end", values=vals, tags=("even" if i % 2 == 0 else "odd",))


# ══════════════════════  CONTROLLER  ═════════════════════════════════════════
class Controller:
    def __init__(self):
        self.s = Session(); self.ml = MLEngine(); self.llm = Ollama()
        self._done_cb = None

    def reset(self): self.s = Session()

    def _col(self, text):
        clean = re.sub(r"[^a-z0-9 ]","", text.strip().lower())
        if self.s.dataset is None: return None
        for c in self.s.dataset.columns:
            if re.sub(r"[^a-z0-9 ]","", c.strip().lower()) == clean: return c
        for c in self.s.dataset.columns:
            if clean and clean in re.sub(r"[^a-z0-9 ]","", c.strip().lower()): return c
        return None

    def _cols_in(self, text):
        found=[]; low=re.sub(r"[^a-z0-9 ]","",text.strip().lower())
        if self.s.dataset is None: return found
        for c in self.s.dataset.columns:
            k=re.sub(r"[^a-z0-9 ]","",c.strip().lower())
            if k and k in low: found.append(c)
        return found

    def _match_val(self, feat, text):
        col=self.s.dataset[feat]
        if str(col.dtype)=="object":
            raw=re.sub(r"[^a-z0-9 ]","",str(text).strip().lower()); lmap={}
            for v in col.dropna().astype(str).unique():
                k=re.sub(r"[^a-z0-9 ]","",v.strip().lower()); lmap.setdefault(k,v)
            if raw in lmap: return lmap[raw]
            for k,v in lmap.items():
                if raw and (raw in k or k in raw): return v
            return None
        try: return float(text)
        except: return None

    def _default(self, f):
        col=self.s.dataset[f].dropna()
        if str(col.dtype)=="object": vc=col.value_counts(); return vc.index[0] if len(vc)>0 else "unknown"
        return float(col.median()) if len(col)>0 else 0.0

    def _build_defaults(self):
        self.s.col_defaults = {f:self._default(f) for f in self.s.selected_feats}

    def _profile(self, df):
        return {"rows":int(df.shape[0]),"columns":int(df.shape[1]),
                "col_names":list(df.columns),
                "dtypes":{c:str(df[c].dtype) for c in df.columns},
                "nulls":{c:int(df[c].isna().sum()) for c in df.columns},
                "unique":{c:int(df[c].nunique(dropna=True)) for c in df.columns},
                "samples":{c:df[c].dropna().astype(str).head(3).tolist() for c in df.columns},
                "id_cols":self._det_ids(df),"tgt_cols":self._det_tgts(df)}

    def _det_ids(self, df):
        ids=[]; n=len(df)
        for c in df.columns:
            nm=c.strip().lower()
            if nm=="id" or nm.endswith("_id") or nm.endswith("id") or df[c].nunique()>=max(1,int(0.98*n)):
                ids.append(c)
        return ids

    def _det_tgts(self, df):
        prio=[]; rest=[]
        for c in df.columns:
            nm=re.sub(r"[^a-z]","",c.lower()); u=df[c].nunique(dropna=True)
            if nm in {"price","sellingprice","target","label","class","output","result"}: prio.insert(0,c)
            elif 2<=u<=min(30,max(2,len(df)//3)): rest.append(c)
        out=[]; seen=set()
        for c in prio+rest:
            if c not in seen: out.append(c); seen.add(c)
        return out or list(df.columns[-3:])

    def _guess_tgt(self, df):
        for c in df.columns:
            if re.sub(r"[^a-z]","",c.lower()) in {"price","sellingprice","target","label","class"}: return c
        t=self._det_tgts(df); return t[0] if t else df.columns[-1]

    def _infer_task(self, col):
        s=self.s.dataset[col]
        return "regression" if (str(s.dtype) not in {"object","category"} and s.nunique()>20) else "classification"

    def _train(self, note=""):
        feats=[c for c in self.s.dataset.columns if c!=self.s.target_column and c not in self.s.dropped_feats]
        if not feats: raise ValueError("No feature columns remain.")
        self.s.selected_feats=feats
        b=self.ml.train(self.s.dataset,self.s.target_column,feats,self.s.null_strategy,self.s.task_type,self.s.chosen_model)
        self.s.model_bundle=b; self.s.metrics=b["metrics"]; self.s.feat_importance=b["fi"]
        self.s.note=note; self._build_defaults(); self._upd_ask()

    def _topk(self):
        total=len(self.s.selected_feats)
        if total<=3: return total
        task=self.s.task_type or "regression"; fi=self.s.feat_importance
        n_cls=int(self.s.dataset[self.s.target_column].nunique(dropna=True))
        if task=="regression":   min_k,max_k=5,9
        elif n_cls==2:           min_k,max_k=3,5
        elif n_cls<=5:           min_k,max_k=4,7
        else:                    min_k,max_k=5,8
        max_k=min(max_k,total); min_k=min(min_k,total)
        if fi:
            tot=sum(x["importance"] for x in fi)
            if tot>0:
                cum=0.0; k=0
                for x in fi:
                    cum+=x["importance"]; k+=1
                    if cum/tot>=0.80: break
                return max(min_k,min(k,max_k))
        return min(max_k,max(min_k,5))

    def _upd_ask(self):
        tk_ = self._topk()
        feat_set = set(self.s.selected_feats)
        top = [x["feature"] for x in self.s.feat_importance
               if x["feature"] in feat_set]
        seen = set()
        ask  = []
        for c in top:
            if c not in seen:
                ask.append(c); seen.add(c)
            if len(ask) >= tk_: break
        for c in self.s.selected_feats:
            if len(ask) >= tk_: break
            if c not in seen:
                ask.append(c); seen.add(c)
        self.s.ask_feats = ask
        self.s.top_k     = tk_

    def load(self, path, model):
        self.reset(); self.s.ollama_model=model
        df=pd.read_csv(path)
        if df.empty or len(df.columns)<2: raise ValueError("CSV too small.")
        prof=self._profile(df); self.s.dataset=df; self.s.profile=prof
        self.s.id_cols=prof["id_cols"]; self.s.candidate_tgts=prof["tgt_cols"]
        self.s.target_column=self._guess_tgt(df); self.s.task_type=self._infer_task(self.s.target_column)
        self.s.dropped_feats=[]; self.s.stage="confirm_target"
        if self.llm.ok():
            try: self.s.llm_summary=self.llm.summarize(model,prof)
            except: self.s.llm_summary=self._fallback()
        else: self.s.llm_summary=self._fallback()
        self._train()
        q=self._ask()
        cols="  |  ".join(df.columns.tolist()[:15])+(" …" if len(df.columns)>15 else "")
        return (f"✅  Loaded  {prof['rows']:,} rows × {prof['columns']} columns\n\n"
                f"📋  Columns:  {cols}\n\n🤖  {self.s.llm_summary}\n\n"
                f"─────────────────────────────────\n\n{q}")

    def _fallback(self):
        return f"Dataset ready. Possible targets: {', '.join(self.s.candidate_tgts[:4]) or self.s.target_column}."

    def _mm(self):
        return "\n".join(f"  {k}.  {v['name']}  —  {v['desc']}" for k,v in MODELS.items())

    def _ask(self):
        s=self.s
        if s.stage=="confirm_target":
            opts="  |  ".join(s.candidate_tgts[:6]) if s.candidate_tgts else s.target_column
            return (f"🎯  What do you want to predict?\n\n"
                    f"Best guess:  **{s.target_column}**\n"
                    f"Options:  {opts}\n\n"
                    f"👉  Type a column name or reply  yes.")
        if s.stage=="confirm_ids":
            if s.id_cols:
                return (f"🆔  These look like IDs:\n   {'  |  '.join(s.id_cols[:6])}\n\n"
                        f"👉  yes  to ignore them,  no  to keep.")
            return "✅  No obvious ID columns.\n\n👉  Any column to ignore? Type a name or  no."
        if s.stage=="confirm_nulls":
            nulls=[k for k,v in s.profile.get("nulls",{}).items() if v>0]
            if nulls:
                return (f"🕳️  Columns with missing values:\n   {'  |  '.join(nulls[:6])}\n\n"
                        f"👉  fill  (use averages)   or   drop  (remove those rows)")
            return "✅  No missing values.\n\n👉  Reply  fill  or  drop."
        if s.stage=="confirm_features":
            top="  |  ".join(x["feature"] for x in s.feat_importance[:5]) or "—"
            return (f"🧩  Model trained!\n\nStrongest columns:\n   {top}\n\n"
                    f"👉  keep  to use all,  or  drop col1, col2  to remove some.")
        if s.stage=="choose_model":
            return f"🤖  Choose a ML model:\n\n{self._mm()}\n\n👉  Type number 1–8 or model name."
        if s.stage=="prediction": return self._next_q()
        return "What would you like to do next?"

    def reply(self, msg):
        if self.s.dataset is None: return "Please upload a CSV file first."
        low=msg.strip().lower(); st=self.s.stage
        if st=="confirm_target":
            if low not in {"yes","ok","y","sure","correct","yep"}:
                col=self._col(msg)
                if col: self.s.target_column=col; self.s.task_type=self._infer_task(col)
                else:
                    return ("❓  Could not find  '"+msg+"'.\n\nColumns:\n"
                            +"  |  ".join(self.s.dataset.columns.tolist())+"\n\nType one.")
            self._train("✅  Target confirmed. Task:  "+self.s.task_type)
            self.s.stage="confirm_ids"; return self._note()+self._ask()
        if st=="confirm_ids":
            if low in {"yes","ignore","y"}:
                self.s.dropped_feats=sorted(set(self.s.dropped_feats+self.s.id_cols))
            elif low not in {"no","keep","n"}:
                cols=self._cols_in(msg)
                if cols: self.s.dropped_feats=sorted(set(self.s.dropped_feats+cols))
            self._train("✅  Columns updated."); self.s.stage="confirm_nulls"
            return self._note()+self._ask()
        if st=="confirm_nulls":
            if any(w in low for w in ["fill","impute","average","mean"]): self.s.null_strategy="fill"
            elif any(w in low for w in ["drop","remove","delete"]): self.s.null_strategy="drop"
            self._train("✅  Missing-value strategy set."); self.s.stage="confirm_features"
            return self._note()+self._ask()
        if st=="confirm_features":
            if low not in {"keep","all","yes","ok","y"}:
                if any(w in low for w in ["drop","remove","ignore"]):
                    cols=self._cols_in(msg)
                    if cols: self.s.dropped_feats=sorted(set(self.s.dropped_feats+cols)); self._train("✅  Features updated.")
                    else: return "❓  Could not find those columns."
            self.s.stage="choose_model"; return self._note()+self._ask()
        if st=="choose_model":
            key=self._res_model(low)
            if not key: return f"❓  Did not recognise that.\n\n{self._mm()}"
            self.s.chosen_model=key; self._train(f"✅  Using  **{MODELS[key]['name']}**.")
            self.s.pred_inputs={}; self.s.stage="prediction"
            return (self._note()+
                    f"🚀  Ready!  Asking  **{self.s.top_k} key questions**  for  **{self.s.target_column}**.\n"
                    "   Type  skip  to auto-fill any question.\n\n"+self._ask())
        if st=="prediction": return self._handle_pred(msg)
        if st=="done": return "🏁  Done.  Click  📊 Graphs  in the sidebar to view all charts.\n\nPress  Reset  to start a new session."
        return "Something went wrong. Please reset."

    def _res_model(self, low):
        clean=re.sub(r"[^a-z0-9 ]","",low.strip())
        if clean in MODELS: return clean
        if clean in ALIASES: return ALIASES[clean]
        for alias,key in ALIASES.items():
            if alias in clean: return key
        return None

    def _note(self):
        n=self.s.note; self.s.note=""; return (n+"\n\n") if n else ""

    def _next_q(self):
        rem=[f for f in self.s.ask_feats if f not in self.s.pred_inputs]
        if not rem: return self._run_pred()
        feat=rem[0]; self.s.awaiting_feat=feat
        col=self.s.dataset[feat]; samp=col.dropna().astype(str).unique()[:4].tolist()
        total=len(self.s.ask_feats); done=len(self.s.pred_inputs)
        d=self.s.col_defaults.get(feat,"—"); ds=f"{d:,.2f}" if isinstance(d,float) else str(d)
        prog=f"[{done+1}/{total}]"
        if str(col.dtype)=="object":
            return (f"📝  {prog}  Value for  **{feat}**?\n\n"
                    f"Known values:  {' | '.join(samp)}\n"
                    f"💡  Type  skip  →  auto-fill  '{ds}'")
        return (f"🔢  {prog}  Value for  **{feat}**?\n\n"
                f"(Number like:  {samp[0] if samp else '0'})\n"
                f"💡  Type  skip  →  auto-fill  {ds}")

    def _handle_pred(self, msg):
        feat=self.s.awaiting_feat
        if not feat: return self._ask()
        low=re.sub(r"[^a-z0-9 ]","",msg.strip().lower())
        if low in SKIPS:
            d=self.s.col_defaults.get(feat); self.s.pred_inputs[feat]=d
            ds=f"{d:,.2f}" if isinstance(d,float) else str(d)
            return f"✅  Skipped  **{feat}**  →  using  **{ds}**\n\n"+self._next_q()
        val=self._match_val(feat,msg)
        if val is None:
            col=self.s.dataset[feat]
            if str(col.dtype)=="object":
                known=sorted(set(col.dropna().astype(str).tolist()))[:8]
                return f"❓  Did not recognise  '{msg}'  for  **{feat}**.\nChoose from:  {' | '.join(known)}\nOr  skip."
            return f"❓  **{feat}**  needs a number. Or  skip."
        self.s.pred_inputs[feat]=val; return self._next_q()

    def _run_pred(self):
        full={}; auto={}
        for f in self.s.selected_feats:
            if f in self.s.pred_inputs: full[f]=self.s.pred_inputs[f]
            else: d=self.s.col_defaults.get(f,0); full[f]=d; auto[f]=d
        result=self.ml.predict(self.s.model_bundle,full)
        self.s.pred_result=result; self.s.stage="done"
        pred=result["prediction"]; mname=self.s.model_bundle.get("model_name","ML")
        m=self.s.metrics; top5=self.s.feat_importance[:5]
        task=self.s.task_type; tgt=self.s.target_column
        top_str="  |  ".join(f"{x['feature']} ({x['importance']:.3f})" for x in top5) or "—"
        auto_str=", ".join(f"{k}={v}" for k,v in list(auto.items())[:5]) if auto else "none"
        if task=="classification":
            m_txt=f"Accuracy: {m.get('Accuracy')}   F1: {m.get('F1')}"
            det=(f"🎉  Prediction complete!\n\n"
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                 f"🏷️   **{tgt}**  →  **{pred}**\n"
                 f"🤖  Model:  {mname}\n📈  {m_txt}\n"
                 f"🔑  Top features:  {top_str}\n🔧  Auto-filled:  {auto_str}\n"
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            try: pd_=f"{float(pred):,.2f}"
            except: pd_=str(pred)
            m_txt=f"MAE: {m.get('MAE')}   RMSE: {m.get('RMSE')}   R²: {m.get('R2')}"
            det=(f"🎉  Prediction complete!\n\n"
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                 f"💰  **{tgt}**  →  **{pd_}**\n"
                 f"🤖  Model:  {mname}\n📈  {m_txt}\n"
                 f"🔑  Top features:  {top_str}\n🔧  Auto-filled:  {auto_str}\n"
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        llm_block=""
        if self.llm.ok():
            try:
                exp=self.llm.explain(self.s.ollama_model,{"target":tgt,"prediction":pred,"model_name":mname,"task_type":task,"metrics":m,"top_features":top5,"user_inputs":self.s.pred_inputs,"auto_filled":auto})
                llm_block=f"\n\n💬  In simple words:\n\n─────────────────────────\n{exp}\n─────────────────────────"
            except Exception as e: llm_block=f"\n\n⚠️  LLM explanation failed: {e}"
        else: llm_block="\n\n⚠️  Ollama not running."
        llm_block+="\n\n📊  Click  📊 Graphs  in the sidebar to see all charts & stats!"
        if self._done_cb:
            self._done_cb(self.s.dataset,tgt,self.s.feat_importance,task,m,mname,result)
        return det+llm_block

    def summary_json(self):
        s=self.s
        return json.dumps({
            "stage":s.stage,"target":s.target_column,"task":s.task_type,
            "model":MODELS.get(s.chosen_model,{}).get("name","—"),
            "features":len(s.selected_feats),"questions":s.top_k,
            "dropped":s.dropped_feats,"nulls":s.null_strategy,
            "metrics":s.metrics,"top5":s.feat_importance[:5],
            "inputs":s.pred_inputs,"result":s.pred_result
        },indent=2,default=str)


# ══════════════════════  MAIN APP  (OBSIDIAN GLASS UI)  ══════════════════════
class App:
    # ── spacing tokens (8px grid) ─────────────────────────────────────────────
    SP_XS  = 4
    SP_SM  = 8
    SP_MD  = 16
    SP_LG  = 24
    SP_XL  = 32

    def __init__(self, root):
        self.root = root
        self.ctrl = Controller()
        root.title("AI Data Assistant")
        root.geometry("1560x960")
        root.configure(bg=C["bg"])
        root.minsize(1200, 760)
        self.ctrl._done_cb = self._on_prediction_done
        self._nav_state = "chat"
        self._typing_widget = None
        self._typewriter_id = None
        self._build()
        self._welcome()

    # ── outer layout ──────────────────────────────────────────────────────────
    def _build(self):
        self._topbar()
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=5)
        body.columnconfigure(2, weight=0)
        body.rowconfigure(0, weight=1)
        self._sidebar(body)
        # content area — chat and graphs share the same grid cell
        content = tk.Frame(body, bg=C["bg"])
        content.grid(row=0, column=1, sticky="nsew")
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        self._build_chat(content)
        self.graphs_panel = GraphsPanel(content)
        self.graphs_panel.grid(row=0, column=0, sticky="nsew")
        self._show_chat()
        self._right_panel(body)

    # ══════════════════════════════════════════════════════════════════════════
    #  TOP BAR — sticky header with glass surface
    # ══════════════════════════════════════════════════════════════════════════
    def _topbar(self):
        bar = tk.Frame(self.root, bg=C["surface1"], height=56)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        # glass bottom edge
        tk.Frame(self.root, bg=C["borderGlass"], height=1).pack(fill=tk.X)

        # ── left: logo ────────────────────────────────────────────────────────
        left = tk.Frame(bar, bg=C["surface1"])
        left.pack(side=tk.LEFT, padx=(20, 0))
        tk.Label(left, text="◆", bg=C["surface1"], fg=C["accent"],
                font=("Segoe UI", 18, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(left, text="AI Data Assistant", bg=C["surface1"], fg=C["white"],
                font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)

        # ── center: workflow breadcrumb ───────────────────────────────────────
        tk.Frame(bar, bg=C["border"], width=1).pack(side=tk.LEFT, fill=tk.Y,
                                                     padx=20, pady=14)
        steps = ["Upload CSV", "→", "Chat", "→", "Predict", "→", "Graphs"]
        breadcrumb = tk.Frame(bar, bg=C["surface1"])
        breadcrumb.pack(side=tk.LEFT)
        for s in steps:
            fg = C["textMuted"] if s == "→" else C["textSec"]
            tk.Label(breadcrumb, text=s, bg=C["surface1"], fg=fg,
                    font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=2)

        # ── right: controls ───────────────────────────────────────────────────
        right = tk.Frame(bar, bg=C["surface1"])
        right.pack(side=tk.RIGHT, padx=20)

        # status pill
        self.status_pill = StatusPill(right, "Ready", C["success"])
        self.status_pill.pack(side=tk.LEFT, padx=(0, 16))

        # model input
        model_frame = tk.Frame(right, bg=C["surface2"],
                               highlightbackground=C["border"], highlightthickness=1)
        model_frame.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(model_frame, text="Model:", bg=C["surface2"], fg=C["textMuted"],
                font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(8, 4), pady=4)
        self.model_var = tk.StringVar(value="qwen2.5")
        model_entry = tk.Entry(model_frame, textvariable=self.model_var,
                              bg=C["surface2"], fg=C["text"],
                              insertbackground=C["accent"], relief=tk.FLAT,
                              font=("Segoe UI", 10), width=12, bd=0)
        model_entry.pack(side=tk.LEFT, ipady=4, ipadx=4, padx=(0, 8), pady=4)

        # upload button
        self._make_btn(right, "📂  Upload", self.upload, C["accent"], C["white"]).pack(
            side=tk.LEFT, padx=4)
        # reset button
        self._make_btn(right, "↻  Reset", self.reset, C["surface2"], C["textSec"]).pack(
            side=tk.LEFT, padx=4)

    def _make_btn(self, parent, text, cmd, bg, fg):
        """Create a styled button with hover effects."""
        btn = tk.Label(parent, text=text, bg=bg, fg=fg,
                      font=("Segoe UI", 9, "bold"), padx=16, pady=7,
                      cursor="hand2",
                      highlightbackground=C["borderGlass"], highlightthickness=1)
        btn.bind("<Button-1>", lambda e: cmd())
        if bg == C["accent"]:
            btn.bind("<Enter>", lambda e: btn.config(bg=C["accentHi"]))
            btn.bind("<Leave>", lambda e: btn.config(bg=C["accent"]))
        else:
            btn.bind("<Enter>", lambda e: btn.config(bg=C["surface3"]))
            btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    # ══════════════════════════════════════════════════════════════════════════
    #  SIDEBAR — clean vertical nav with active indicator
    # ══════════════════════════════════════════════════════════════════════════
    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=C["surface1"], width=220)
        sb.grid(row=0, column=0, sticky="ns")
        sb.pack_propagate(False)

        # glass right edge
        tk.Frame(sb, bg=C["borderGlass"], width=1).pack(side=tk.RIGHT, fill=tk.Y)

        # ── navigation section ────────────────────────────────────────────────
        nav_label = tk.Frame(sb, bg=C["surface1"])
        nav_label.pack(fill=tk.X, padx=16, pady=(20, 8))
        tk.Label(nav_label, text="NAVIGATION", bg=C["surface1"], fg=C["textMuted"],
                font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT)

        self.nav_chat   = self._nav_item(sb, "💬", "Chat",   lambda: self._show_chat())
        self.nav_graphs = self._nav_item(sb, "📊", "Analytics", lambda: self._show_graphs())

        # ── session section ───────────────────────────────────────────────────
        tk.Frame(sb, bg=C["border"], height=1).pack(fill=tk.X, padx=16, pady=(20, 12))
        session_label = tk.Frame(sb, bg=C["surface1"])
        session_label.pack(fill=tk.X, padx=16, pady=(0, 8))
        tk.Label(session_label, text="SESSION", bg=C["surface1"], fg=C["textMuted"],
                font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT)

        self.lbl_stage  = self._sb_info(sb, "Stage",  "idle",  C["warning"])
        self.lbl_target = self._sb_info(sb, "Target", "—",     C["textSec"])
        self.lbl_model  = self._sb_info(sb, "Model",  "—",     C["textSec"])
        self.lbl_task   = self._sb_info(sb, "Task",   "—",     C["textSec"])
        self.lbl_rows   = self._sb_info(sb, "Rows",   "—",     C["textSec"])

    def _sb_info(self, parent, label, value, fg):
        """Compact sidebar info row."""
        row = tk.Frame(parent, bg=C["surface1"])
        row.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(row, text=f"{label}:", bg=C["surface1"], fg=C["textMuted"],
                font=("Segoe UI", 9)).pack(side=tk.LEFT)
        val = tk.Label(row, text=value, bg=C["surface1"], fg=fg,
                      font=("Segoe UI", 9, "bold"))
        val.pack(side=tk.LEFT, padx=(6, 0))
        return val

    def _nav_item(self, parent, icon, label, cmd):
        """Sidebar nav item with left active indicator."""
        outer = tk.Frame(parent, bg=C["surface1"])
        outer.pack(fill=tk.X, padx=8, pady=2)

        # left indicator bar
        indicator = tk.Frame(outer, bg=C["surface1"], width=3)
        indicator.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 0))

        row = tk.Frame(outer, bg=C["surface1"], cursor="hand2", padx=12, pady=10)
        row.pack(fill=tk.X)

        ico = tk.Label(row, text=icon, bg=C["surface1"], fg=C["text"],
                      font=("Segoe UI", 14))
        ico.pack(side=tk.LEFT, padx=(0, 10))

        lbl = tk.Label(row, text=label, bg=C["surface1"], fg=C["textSec"],
                      font=("Segoe UI", 10, "bold"))
        lbl.pack(side=tk.LEFT)

        # bind click and hover to all sub-widgets
        for w in [outer, row, ico, lbl]:
            w.bind("<Button-1>", lambda e, c=cmd: c())
        for w in [row, ico, lbl]:
            w.bind("<Enter>", lambda e, r=row, i=ico, l=lbl: (
                r.config(bg=C["surface2"]),
                i.config(bg=C["surface2"]),
                l.config(bg=C["surface2"])
            ))
            w.bind("<Leave>", lambda e, r=row, i=ico, l=lbl, nav_data=None: (
                r.config(bg=C["surface1"]),
                i.config(bg=C["surface1"]),
                l.config(bg=C["surface1"])
            ))

        return {"outer": outer, "indicator": indicator, "row": row,
                "ico": ico, "lbl": lbl}

    def _set_nav_active(self, key):
        for nav_key, item in [("chat", self.nav_chat), ("graphs", self.nav_graphs)]:
            is_active = nav_key == key
            bg = C["surface2"] if is_active else C["surface1"]
            fg = C["white"]    if is_active else C["textSec"]
            ind_bg = C["accent"] if is_active else C["surface1"]
            item["indicator"].config(bg=ind_bg)
            for w in [item["row"], item["ico"], item["lbl"]]:
                w.config(bg=bg)
            item["lbl"].config(fg=fg)
            item["ico"].config(fg=C["white"] if is_active else C["text"])
            # rebind hover for inactive items only
            if not is_active:
                for w in [item["row"], item["ico"], item["lbl"]]:
                    w.bind("<Enter>", lambda e, r=item["row"], i=item["ico"], l=item["lbl"]: (
                        r.config(bg=C["surface2"]),
                        i.config(bg=C["surface2"]),
                        l.config(bg=C["surface2"])
                    ))
                    w.bind("<Leave>", lambda e, r=item["row"], i=item["ico"], l=item["lbl"]: (
                        r.config(bg=C["surface1"]),
                        i.config(bg=C["surface1"]),
                        l.config(bg=C["surface1"])
                    ))
            else:
                for w in [item["row"], item["ico"], item["lbl"]]:
                    w.bind("<Enter>", lambda e: None)
                    w.bind("<Leave>", lambda e: None)
        self._nav_state = key

    def _show_chat(self):
        self.graphs_panel.grid_remove()
        self.chat_outer.grid(row=0, column=0, sticky="nsew")
        self._set_nav_active("chat")

    def _show_graphs(self):
        self.chat_outer.grid_remove()
        self.graphs_panel.grid(row=0, column=0, sticky="nsew")
        self._set_nav_active("graphs")

    # ══════════════════════════════════════════════════════════════════════════
    #  CHAT PANEL — bubble-style messages with typing animation
    # ══════════════════════════════════════════════════════════════════════════
    def _build_chat(self, parent):
        self.chat_outer = tk.Frame(parent, bg=C["bg"])
        self.chat_outer.grid(row=0, column=0, sticky="nsew")
        self.chat_outer.rowconfigure(1, weight=1)
        self.chat_outer.columnconfigure(0, weight=1)

        # ── header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.chat_outer, bg=C["surface1"], height=52)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="💬", bg=C["surface1"], fg=C["accent"],
                font=("Segoe UI", 16)).pack(side=tk.LEFT, padx=(20, 8), pady=10)
        tk.Label(hdr, text="Chat", bg=C["surface1"], fg=C["text"],
                font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, pady=10)
        self.chat_sub = tk.Label(hdr, text="Upload a CSV to begin",
                                bg=C["surface1"], fg=C["textMuted"],
                                font=("Segoe UI", 9))
        self.chat_sub.pack(side=tk.LEFT, padx=16)
        # glass border
        tk.Frame(self.chat_outer, bg=C["borderGlass"], height=1).grid(
            row=0, column=0, sticky="sew")

        # ── scrollable message area ───────────────────────────────────────────
        msg_container = tk.Frame(self.chat_outer, bg=C["bg"])
        msg_container.grid(row=1, column=0, sticky="nsew")
        msg_container.rowconfigure(0, weight=1)
        msg_container.columnconfigure(0, weight=1)

        self.msg_canvas = tk.Canvas(msg_container, bg=C["bg"], highlightthickness=0)
        sc = tk.Scrollbar(msg_container, command=self.msg_canvas.yview,
                         bg=C["surface1"], troughcolor=C["bg"],
                         activebackground=C["accent"], width=7)
        self.msg_canvas.config(yscrollcommand=sc.set)
        sc.grid(row=0, column=1, sticky="ns")
        self.msg_canvas.grid(row=0, column=0, sticky="nsew")

        self.msg_frame = tk.Frame(self.msg_canvas, bg=C["bg"])
        self._msg_win = self.msg_canvas.create_window(
            (0, 0), window=self.msg_frame, anchor="nw")

        self.msg_frame.bind("<Configure>",
            lambda e: self.msg_canvas.configure(scrollregion=self.msg_canvas.bbox("all")))
        self.msg_canvas.bind("<Configure>",
            lambda e: self.msg_canvas.itemconfig(self._msg_win, width=e.width))
        self.msg_canvas.bind_all("<MouseWheel>",
            lambda e: self.msg_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # ── input bar ─────────────────────────────────────────────────────────
        inp_outer = tk.Frame(self.chat_outer, bg=C["surface1"])
        inp_outer.grid(row=2, column=0, sticky="ew")
        # glass top border
        tk.Frame(inp_outer, bg=C["borderGlass"], height=1).pack(fill=tk.X)

        inp = tk.Frame(inp_outer, bg=C["surface1"], padx=16, pady=12)
        inp.pack(fill=tk.X)

        # input field container
        input_frame = tk.Frame(inp, bg=C["inputBg"],
                              highlightbackground=C["border"], highlightthickness=1)
        input_frame.pack(fill=tk.X)
        input_frame.columnconfigure(0, weight=1)

        self.input_var = tk.StringVar()
        self.entry = tk.Entry(input_frame, textvariable=self.input_var,
                             bg=C["inputBg"], fg=C["text"],
                             insertbackground=C["accent"], relief=tk.FLAT,
                             font=("Segoe UI", 11), bd=0)
        self.entry.grid(row=0, column=0, sticky="ew", ipady=10, ipadx=16)
        self.entry.bind("<Return>", lambda e: self.send())

        # send button
        send_btn = tk.Label(input_frame, text="  →  ", bg=C["accent"], fg=C["white"],
                           font=("Segoe UI", 13, "bold"), padx=14, pady=8,
                           cursor="hand2")
        send_btn.grid(row=0, column=1, padx=4, pady=4)
        send_btn.bind("<Button-1>", lambda e: self.send())
        send_btn.bind("<Enter>", lambda e: send_btn.config(bg=C["accentHi"]))
        send_btn.bind("<Leave>", lambda e: send_btn.config(bg=C["accent"]))

    # ══════════════════════════════════════════════════════════════════════════
    #  RIGHT PANEL — structured session state cards
    # ══════════════════════════════════════════════════════════════════════════
    def _right_panel(self, parent):
        frame = tk.Frame(parent, bg=C["surface1"], width=300)
        frame.grid(row=0, column=2, sticky="nsew")
        frame.pack_propagate(False)

        # glass left edge
        tk.Frame(frame, bg=C["borderGlass"], width=1).pack(side=tk.LEFT, fill=tk.Y)

        inner = tk.Frame(frame, bg=C["surface1"])
        inner.pack(fill=tk.BOTH, expand=True)

        # ── header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(inner, bg=C["surface1"], height=52)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🧠", bg=C["surface1"], fg=C["accent"],
                font=("Segoe UI", 16)).pack(side=tk.LEFT, padx=(16, 8), pady=10)
        tk.Label(hdr, text="Session State", bg=C["surface1"], fg=C["text"],
                font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, pady=10)
        tk.Frame(inner, bg=C["borderGlass"], height=1).pack(fill=tk.X)

        # ── session cards ─────────────────────────────────────────────────────
        cards_area = tk.Frame(inner, bg=C["surface1"])
        cards_area.pack(fill=tk.X, padx=12, pady=12)

        self.card_model = SessionCard(cards_area, "MODEL", "🤖")
        self.card_model.pack(fill=tk.X, pady=4)
        self.card_model.set_value("—")

        self.card_target = SessionCard(cards_area, "TARGET", "🎯")
        self.card_target.pack(fill=tk.X, pady=4)
        self.card_target.set_value("—")

        self.card_task = SessionCard(cards_area, "TASK TYPE", "⚙️")
        self.card_task.pack(fill=tk.X, pady=4)
        self.card_task.set_value("—")

        self.card_metrics = SessionCard(cards_area, "METRICS", "📈")
        self.card_metrics.pack(fill=tk.X, pady=4)
        self.card_metrics.set_value("—")

        # ── raw JSON view ─────────────────────────────────────────────────────
        tk.Frame(inner, bg=C["border"], height=1).pack(fill=tk.X, padx=12, pady=(8, 4))
        json_label = tk.Frame(inner, bg=C["surface1"])
        json_label.pack(fill=tk.X, padx=16, pady=(4, 4))
        tk.Label(json_label, text="RAW STATE", bg=C["surface1"], fg=C["textMuted"],
                font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT)

        self.summary = tk.Text(inner, wrap=tk.WORD, state=tk.DISABLED,
                              font=("Consolas", 8), bg=C["bg2"], fg=C["textMuted"],
                              relief=tk.FLAT, padx=12, pady=8,
                              highlightbackground=C["border"], highlightthickness=1)
        ss = tk.Scrollbar(inner, command=self.summary.yview,
                         bg=C["surface1"], troughcolor=C["surface1"],
                         activebackground=C["accent"], width=6)
        self.summary.config(yscrollcommand=ss.set)
        ss.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12))
        self.summary.pack(fill=tk.BOTH, expand=True, padx=(12, 0), pady=(0, 12))
        self._ref_summary()

    # ══════════════════════════════════════════════════════════════════════════
    #  MESSAGE HELPERS — chat bubbles, typing animation
    # ══════════════════════════════════════════════════════════════════════════
    def _add_bubble(self, text, sender="bot", animate=False):
        """Add a styled message bubble to the chat."""
        now = datetime.datetime.now().strftime("%H:%M")
        is_user = sender == "user"
        is_sys  = sender == "sys"

        # ── container row ─────────────────────────────────────────────────────
        row = tk.Frame(self.msg_frame, bg=C["bg"])
        row.pack(fill=tk.X, padx=20, pady=(6, 6))

        # ── alignment wrapper ─────────────────────────────────────────────────
        align = tk.Frame(row, bg=C["bg"])

        if is_sys:
            align.pack(anchor="w")
            # system message: simple label
            sys_lbl = tk.Label(align, text=f"   {text}", bg=C["bg"],
                              fg=C["success"], font=("Segoe UI", 10),
                              wraplength=600, justify=tk.LEFT, anchor="w")
            sys_lbl.pack(anchor="w")
            self._scroll_to_bottom()
            return

        if is_user:
            align.pack(anchor="e")
        else:
            align.pack(anchor="w")

        # ── avatar + bubble container ─────────────────────────────────────────
        msg_row = tk.Frame(align, bg=C["bg"])
        msg_row.pack()

        # avatar
        avatar_text = "👤" if is_user else "🤖"
        avatar_fg   = C["accentHi"] if is_user else C["accent"]

        if not is_user:
            av = tk.Label(msg_row, text=avatar_text, bg=C["bg"], fg=avatar_fg,
                         font=("Segoe UI", 13))
            av.pack(side=tk.LEFT, anchor="n", padx=(0, 8), pady=(4, 0))

        # bubble
        bubble_bg = C["userBubble"] if is_user else C["botBubble"]
        bubble = tk.Frame(msg_row, bg=bubble_bg, padx=16, pady=12,
                         highlightbackground=C["borderGlass"] if not is_user else C["accentMute"],
                         highlightthickness=1)
        bubble.pack(side=tk.LEFT)

        # message text
        msg_label = tk.Label(bubble, text=text, bg=bubble_bg,
                            fg=C["white"] if is_user else C["text"],
                            font=("Segoe UI", 11), wraplength=520,
                            justify=tk.LEFT, anchor="w")
        msg_label.pack(anchor="w")

        # timestamp
        time_frame = tk.Frame(bubble, bg=bubble_bg)
        time_frame.pack(fill=tk.X, pady=(6, 0))
        time_lbl = tk.Label(time_frame, text=now, bg=bubble_bg,
                           fg=C["textMuted"], font=("Segoe UI", 7))
        time_lbl.pack(side=tk.RIGHT if is_user else tk.LEFT)

        if is_user:
            av = tk.Label(msg_row, text=avatar_text, bg=C["bg"], fg=avatar_fg,
                         font=("Segoe UI", 13))
            av.pack(side=tk.LEFT, anchor="n", padx=(8, 0), pady=(4, 0))

        # ── typewriter animation for bot messages ─────────────────────────────
        if animate and not is_user:
            msg_label.config(text="")
            self._typewriter_reveal(msg_label, text, 0)
        else:
            self._scroll_to_bottom()

    def _typewriter_reveal(self, label, full_text, index):
        """Reveal text word by word with a typewriter effect."""
        if index >= len(full_text):
            self._scroll_to_bottom()
            return
        # reveal 3 characters at a time for speed
        chunk = 4
        end = min(index + chunk, len(full_text))
        label.config(text=full_text[:end])
        self._scroll_to_bottom()
        self._typewriter_id = self.root.after(8, self._typewriter_reveal,
                                               label, full_text, end)

    def _show_typing(self):
        """Show animated typing indicator in chat."""
        if self._typing_widget:
            self._hide_typing()
        row = tk.Frame(self.msg_frame, bg=C["bg"])
        row.pack(fill=tk.X, padx=20, pady=(4, 4))
        align = tk.Frame(row, bg=C["bg"])
        align.pack(anchor="w")
        msg_row = tk.Frame(align, bg=C["bg"])
        msg_row.pack()
        av = tk.Label(msg_row, text="🤖", bg=C["bg"], fg=C["accent"],
                     font=("Segoe UI", 13))
        av.pack(side=tk.LEFT, anchor="n", padx=(0, 8), pady=(4, 0))
        dots = TypingDots(msg_row)
        dots.pack(side=tk.LEFT)
        self._typing_widget = row
        self._typing_dots = dots
        self._scroll_to_bottom()

    def _hide_typing(self):
        """Remove the typing indicator."""
        if self._typing_widget:
            if hasattr(self, '_typing_dots'):
                self._typing_dots.stop()
            self._typing_widget.destroy()
            self._typing_widget = None

    def _scroll_to_bottom(self):
        """Scroll chat to the latest message."""
        self.msg_canvas.update_idletasks()
        self.msg_canvas.yview_moveto(1.0)

    # ── legacy append adapter (maps old tag system to new bubbles) ────────────
    def _append(self, tag, text):
        sender_map = {"bot": "bot", "user": "user", "sys": "sys"}
        sender = sender_map.get(tag, "bot")
        animate = tag == "bot"
        self._add_bubble(text, sender=sender, animate=animate)

    # ── refresh session state display ─────────────────────────────────────────
    def _ref_summary(self):
        txt = self.ctrl.summary_json()
        self.summary.config(state=tk.NORMAL)
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, txt)
        self.summary.config(state=tk.DISABLED)

        s = self.ctrl.s
        # sidebar labels
        self.lbl_stage.config(text=s.stage)
        self.lbl_target.config(text=s.target_column or "—")
        self.lbl_model.config(text=MODELS.get(s.chosen_model, {}).get("name", "—"))
        self.lbl_task.config(text=s.task_type or "—")
        rows = f"{len(s.dataset):,}" if s.dataset is not None else "—"
        self.lbl_rows.config(text=rows)

        # right panel cards
        self.card_model.set_value(
            MODELS.get(s.chosen_model, {}).get("name", "—"), C["text"])
        self.card_target.set_value(
            s.target_column or "—", C["accent"] if s.target_column else C["textMuted"])
        self.card_task.set_value(
            (s.task_type or "—").capitalize(), C["info"] if s.task_type else C["textMuted"])

        # metrics card
        if s.metrics:
            m_parts = [f"{k}: {v}" for k, v in s.metrics.items()]
            self.card_metrics.set_value("  ·  ".join(m_parts), C["success"])
        else:
            self.card_metrics.set_value("—", C["textMuted"])

    # ── welcome message ───────────────────────────────────────────────────────
    def _welcome(self):
        self._add_bubble(
            "Welcome! 👋  Upload a CSV file to start a prediction session.\n\n"
            "What I'll do:\n"
            "  1️⃣   Choose what to predict\n"
            "  2️⃣   Ignore ID / irrelevant columns\n"
            "  3️⃣   Handle missing values\n"
            "  4️⃣   Pick a ML model  (8 choices)\n"
            "  5️⃣   Ask only the most important questions\n"
            "  6️⃣   Give prediction + explanation\n"
            "  7️⃣   Click  📊 Analytics  to see charts & stats",
            sender="bot", animate=False)

    # ══════════════════════════════════════════════════════════════════════════
    #  ACTIONS — upload, reset, send
    # ══════════════════════════════════════════════════════════════════════════
    def upload(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self.status_pill.update_status("Analyzing…", C["warning"])
        fname = path.split("/")[-1].split("\\")[-1]
        self.chat_sub.config(text=fname)
        self._add_bubble(f"Loading:  {fname}", sender="sys")
        self._show_chat()
        self._show_typing()

        def task():
            try:
                resp = self.ctrl.load(path, self.model_var.get().strip() or "qwen2.5")
                self.root.after(0, self._hide_typing)
                self.root.after(50, lambda: self._add_bubble(resp, "bot", animate=True))
                self.root.after(0, self._ref_summary)
                self.root.after(0, lambda: self.status_pill.update_status(
                    "Waiting for reply", C["accent"]))
            except Exception as ex:
                self.root.after(0, self._hide_typing)
                self.root.after(0, lambda: messagebox.showerror("Load Error", str(ex)))
                self.root.after(0, lambda: self.status_pill.update_status("Error", C["error"]))
        threading.Thread(target=task, daemon=True).start()

    def reset(self):
        self.ctrl.reset()
        # clear all messages
        for w in self.msg_frame.winfo_children():
            w.destroy()
        self._welcome()
        self._ref_summary()
        self.status_pill.update_status("Ready", C["success"])
        self.chat_sub.config(text="Upload a CSV to begin")
        self._show_chat()

    def send(self):
        msg = self.input_var.get().strip()
        if not msg:
            return
        self.input_var.set("")
        self._add_bubble(msg, sender="user")
        self.status_pill.update_status("Thinking…", C["warning"])
        self._show_typing()

        def task():
            try:
                resp = self.ctrl.reply(msg)
                self.root.after(0, self._hide_typing)
                self.root.after(50, lambda: self._add_bubble(resp, "bot", animate=True))
                self.root.after(0, self._ref_summary)
                self.root.after(0, lambda: self.status_pill.update_status(
                    "Waiting for reply", C["accent"]))
            except Exception as ex:
                self.root.after(0, self._hide_typing)
                self.root.after(0, lambda: self._add_bubble(
                    f"Something went wrong:\n{ex}", "bot"))
                self.root.after(0, lambda: self.status_pill.update_status("Error", C["error"]))
        threading.Thread(target=task, daemon=True).start()

    def _on_prediction_done(self, df, target, fi, task, metrics, model_name, pred_result):
        """Called from background thread — load graph data on main thread."""
        def load_graphs():
            self.graphs_panel.load(df, target, fi, task, metrics, model_name, pred_result)
            # flash Graphs nav to signal data ready
            item = self.nav_graphs
            original_bg = C["surface1"]
            for i in range(4):
                self.root.after(250 * i,
                    lambda: item["indicator"].config(bg=C["accent"]))
                self.root.after(250 * i + 125,
                    lambda: item["indicator"].config(
                        bg=C["accent"] if self._nav_state == "graphs" else original_bg))
        self.root.after(400, load_graphs)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
