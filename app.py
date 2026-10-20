import os, re, csv, json, glob, time, html
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from mu_rag import MURAG, REFUSE_MSG, refusal_route

st.set_page_config(page_title="MU Student Support Chatbot", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")
OUT_DIR = os.environ.get("MU_OUT_DIR", "outputs")
CFG = json.load(open(os.path.join(OUT_DIR, "config.json"), encoding="utf-8"))
RULE = CFG["rule"]
FEEDBACK = os.path.join(OUT_DIR, "feedback_log.csv")
LANG_LABEL = {"en": "English", "bn": "বাংলা", "banglish": "Banglish"}
LANG_FLAG = {"en": "🇬🇧", "bn": "🇧🇩", "banglish": "🔤"}
# (icon, short title, question) for the welcome cards and the sidebar
SAMPLES = [("📅", "Class routine", "CSE 58 A er monday class ki?"),
           ("📝", "Supplementary exam", "সাপ্লি পরীক্ষার রেজিস্ট্রেশন কোথায় করব?"),
           ("💳", "Fees", "retake fee koto?"),
           ("🎓", "Credits", "How many credits are needed for BSc in CSE?"),
           ("🚌", "Bus schedule", "rikabibazar theke bus koytay?"),
           ("🛡️", "Out of scope", "Who won the 2022 FIFA World Cup?")]
LANG_RULE = {"en": "Reply only in English.",
             "bn": "Reply only in Bangla (Bengali script). Keep course codes, names, room numbers, times and emails exactly as written.",
             "banglish": "Reply only in Banglish (Bangla written with English letters). Keep course codes, names, room numbers, times and emails exactly as written."}
RAG_SYSTEM = ("You are the official student-support assistant of Metropolitan University (MU), Sylhet. "
              "Rewrite the verified answer as a short, natural reply to the student's question. "
              "Use ONLY the verified answer. Keep EVERY number, date, time, amount, course code, name, room, phone number "
              "and email exactly as written, and do not add anything new. ")


# =============================== pipeline (unchanged) ===============================
@st.cache_resource(show_spinner="Loading models (first time only, 2-4 minutes)...")
def load_pipeline():
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,
                              BitsAndBytesConfig)
    n_gpu = torch.cuda.device_count()
    dev_llm = "cuda:0" if n_gpu else "cpu"
    dev_ret = "cuda:1" if n_gpu > 1 else dev_llm
    kb = pd.read_csv(CFG["kb_csv"], keep_default_na=False)
    facts = pd.read_csv(CFG["facts_csv"], keep_default_na=False)
    emb = SentenceTransformer(CFG["embed_model"], device=dev_ret)
    if n_gpu: emb.half()
    def embed(texts):
        return emb.encode(list(texts), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    rtok = AutoTokenizer.from_pretrained(CFG["rerank_model"])
    rmod = AutoModelForSequenceClassification.from_pretrained(
        CFG["rerank_model"], torch_dtype=torch.float16 if n_gpu else torch.float32).to(dev_ret).eval()
    @torch.no_grad()
    def rerank(q, passages):
        enc = rtok([[q, p] for p in passages], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev_ret)
        return torch.sigmoid(rmod(**enc).logits.view(-1).float()).cpu().numpy()
    rag = MURAG(kb, facts, embed=embed, rerank=rerank, n_rerank=CFG.get("n_rerank", 20))
    llm = tok = None
    if CFG.get("use_llm_rewrite") and n_gpu:
        try:
            tok = AutoTokenizer.from_pretrained(CFG["llm"])
            if "7B" in CFG["llm"]:
                llm = AutoModelForCausalLM.from_pretrained(CFG["llm"], device_map={"": dev_llm}, quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
            else:
                llm = AutoModelForCausalLM.from_pretrained(CFG["llm"], torch_dtype=torch.float16, device_map={"": dev_llm}).eval()
        except Exception:          # the app still works without the LLM: it shows the verified answers
            llm = tok = None
    return rag, llm, tok


def generate(llm, tok, msgs):
    import torch
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**enc, max_new_tokens=200, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def respond(q):
    rag, llm, tok = load_pipeline()
    t0 = time.time()
    r = rag.retrieve(q, CFG.get("final_system", "final"))
    lang = r["lang"]
    info = dict(lang=lang, conf=r["conf"], margin=r["margin"], fact="", source="", mode="", status=r["status"])
    if not MURAG.decide(r, RULE):
        route = refusal_route(q, r["status"])
        info["mode"] = f"refused ({route})"
        return REFUSE_MSG[route][lang], info, time.time() - t0
    fid = r["ranking"][0][0]
    verified = rag.verified_answer(fid, lang)
    info.update(fact=fid, source=str(rag.facts.at[fid, "source"]))
    ans, info["mode"] = verified, "verified answer"
    if llm is not None and rag.kind[fid] == "answer" and len(verified) <= CFG.get("rewrite_max_chars", 350):
        gen = generate(llm, tok, [{"role": "system", "content": RAG_SYSTEM + LANG_RULE[lang]},
                                  {"role": "user", "content": f"Verified answer:\n{verified}\n\nStudent question: {q}"}])
        if not rag.guard(gen, verified, lang):
            ans, info["mode"] = gen, "LLM answer (checked against the verified fact)"
        else:
            info["mode"] = "verified answer (LLM rewrite rejected by the guard)"
    return ans, info, time.time() - t0


def log_feedback(q, a, info, rating):
    new = not os.path.exists(FEEDBACK)
    with open(FEEDBACK, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["time", "question", "answer", "language", "fact", "confidence", "mode", "rating"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), q, a, info.get("lang"), info.get("fact"),
                    round(float(info.get("conf") or 0), 3), info.get("mode"), rating])


# =============================== look & feel ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600&family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stChatMessage, button, input, textarea {
  font-family: 'Poppins', 'Hind Siliguri', sans-serif !important; }
.stApp { background: linear-gradient(180deg, #eef3fb 0%, #f7f9fc 260px, #f7f9fc 100%); }
.block-container { padding-top: 1.4rem; padding-bottom: 6rem; max-width: 1000px; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stMain"] .stButton button p { white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; text-align: left; }

/* hero */
.hero { position: relative; overflow: hidden; border-radius: 22px; padding: 26px 30px 22px;
  background: linear-gradient(125deg, #0a2a66 0%, #123f93 45%, #1d6fd6 100%); color: #fff;
  box-shadow: 0 14px 34px rgba(13, 47, 110, .28); margin-bottom: 18px; }
.hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 240px; height: 240px;
  border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,.18), rgba(255,255,255,0) 70%); }
.hero .row { display: flex; align-items: center; gap: 16px; }
.hero .logo { width: 58px; height: 58px; border-radius: 16px; background: rgba(255,255,255,.16);
  display: flex; align-items: center; justify-content: center; font-size: 30px; border: 1px solid rgba(255,255,255,.3); }
.hero h1 { color: #fff; font-size: 1.65rem; font-weight: 700; margin: 0; line-height: 1.25; }
.hero p { margin: 4px 0 0; opacity: .9; font-size: .95rem; }
.chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.chip { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.3); padding: 4px 12px;
  border-radius: 999px; font-size: .78rem; }
.live { display: inline-flex; align-items: center; gap: 6px; }
.live:before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: #3ddc84;
  box-shadow: 0 0 0 0 rgba(61,220,132,.7); animation: pulse 1.8s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(61,220,132,.7);} 70% { box-shadow: 0 0 0 8px rgba(61,220,132,0);} 100% { box-shadow: 0 0 0 0 rgba(61,220,132,0);} }

/* welcome */
.welcome { background: #fff; border-radius: 18px; padding: 18px 22px; border: 1px solid #e3e9f3;
  box-shadow: 0 4px 14px rgba(20, 50, 100, .06); margin-bottom: 12px; }
.welcome h3 { margin: 0 0 4px; font-size: 1.12rem; color: #0a2a66; }
.welcome p { margin: 0; color: #4b5563; font-size: .92rem; }

/* sample cards (buttons) */
div[data-testid="stHorizontalBlock"] .stButton button.sample, .sample-wrap .stButton button {
  text-align: left; }
[data-testid="stMain"] .stButton button { border-radius: 14px; border: 1px solid #dbe4f3; background: #fff; color: #0f2a5c;
  padding: 12px 14px; min-height: 64px; font-size: .88rem; transition: all .15s ease; white-space: normal;
  box-shadow: 0 2px 6px rgba(20,50,100,.05); }
[data-testid="stMain"] .stButton button:hover { border-color: #1d6fd6; color: #1d6fd6; transform: translateY(-2px);
  box-shadow: 0 8px 18px rgba(29,111,214,.15); }

/* chat bubbles */
div[data-testid="stChatMessage"] { background: #fff; border: 1px solid #e3e9f3; border-radius: 18px;
  padding: 14px 16px; box-shadow: 0 3px 12px rgba(20, 50, 100, .06); margin-bottom: 10px; }
div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]),
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg, #e8f0fe, #dbe7fd); border-color: #c9daf8; }
.stChatMessage p { font-size: .97rem; line-height: 1.6; }

/* answer meta */
.meta { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 8px; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 10px; border-radius: 999px;
  font-size: .74rem; font-weight: 500; border: 1px solid transparent; }
.b-lang { background: #eef3fb; color: #123f93; border-color: #d6e2f6; }
.b-ok { background: #e7f7ee; color: #12753e; border-color: #c4ead4; }
.b-llm { background: #f1ecff; color: #5b34c7; border-color: #ddd2fb; }
.b-ref { background: #fff4e5; color: #a95b00; border-color: #fbdcb0; }
.b-time { background: #f3f4f6; color: #4b5563; }
.conf { display: inline-flex; align-items: center; gap: 6px; font-size: .74rem; color: #4b5563; }
.conf .bar { width: 70px; height: 6px; border-radius: 6px; background: #e5e7eb; overflow: hidden; }
.conf .bar > span { display: block; height: 100%; border-radius: 6px; }
.src { margin-top: 8px; font-size: .78rem; color: #6b7280; background: #f8fafc; border-left: 3px solid #1d6fd6;
  padding: 6px 10px; border-radius: 6px; }

/* sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a2a66 0%, #0f3780 100%); }
section[data-testid="stSidebar"] * { color: #e8eefc !important; }
section[data-testid="stSidebar"] .stButton button { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18);
  border-radius: 12px; text-align: left; justify-content: flex-start; font-size: .83rem; padding: 8px 12px; }
section[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,.18); border-color: rgba(255,255,255,.4); }
section[data-testid="stSidebar"] .stDownloadButton button { background: rgba(255,255,255,.08); border-radius: 12px;
  border: 1px solid rgba(255,255,255,.18); }
.side-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.side-brand .lg { font-size: 26px; }
.side-brand b { font-size: 1.02rem; }
.stat { display: flex; gap: 8px; margin: 8px 0 4px; }
.stat div { flex: 1; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.15); border-radius: 12px;
  padding: 8px; text-align: center; }
.stat b { display: block; font-size: 1.15rem; }
.stat span { font-size: .7rem; opacity: .85; }
.foot { text-align: center; color: #94a3b8; font-size: .75rem; margin-top: 18px; }
</style>
""", unsafe_allow_html=True)


def conf_color(c):
    return "#16a34a" if c >= 0.8 else ("#f59e0b" if c >= 0.5 else "#ef4444")


def meta_html(inf, secs):
    lang = inf.get("lang", "en")
    mode = inf.get("mode", "")
    if mode.startswith("refused"):
        route = mode[mode.find("(") + 1:mode.rfind(")")]
        kind = {"exam": '<span class="badge b-ref">🛡️ Not sure — referred to the Exam Office</span>',
                "mu": '<span class="badge b-ref">🛡️ Not sure — referred to the One Stop service</span>',
                "other": '<span class="badge b-ref">🏫 Other university — outside my scope</span>'}.get(
                    route, '<span class="badge b-ref">🛡️ Outside MU topics — not answered</span>')
    elif mode.startswith("LLM"):
        kind = '<span class="badge b-llm">✨ AI-phrased · checked against verified fact</span>'
    else:
        kind = '<span class="badge b-ok">✅ Verified MU information</span>'
    c = float(inf.get("conf") or 0)
    pct = max(3, min(100, int(c * 100)))
    conf = (f'<span class="conf">confidence <span class="bar"><span style="width:{pct}%;background:{conf_color(c)}">'
            f'</span></span> {c:.2f}</span>')
    out = (f'<div class="meta"><span class="badge b-lang">{LANG_FLAG.get(lang, "")} {LANG_LABEL.get(lang, lang)}</span>'
           f'{kind}{conf}<span class="badge b-time">⏱ {secs:.1f}s</span></div>')
    if inf.get("source"):
        out += f'<div class="src">📄 Source: {html.escape(inf["source"])} &nbsp;·&nbsp; fact {html.escape(str(inf.get("fact", "")))}</div>'
    return out


# =============================== state ===============================
if "chat" not in st.session_state:
    st.session_state["chat"] = []
chat = st.session_state["chat"]
n_q = sum(m["role"] == "user" for m in chat)
n_up = sum(m.get("rated") == "up" for m in chat)

# =============================== sidebar ===============================
with st.sidebar:
    st.markdown('<div class="side-brand"><span class="lg">🎓</span><b>MU Student Support</b></div>'
                '<div style="font-size:.8rem;opacity:.85">Metropolitan University, Sylhet</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat"><div><b>{n_q}</b><span>questions</span></div>'
                f'<div><b>{n_up}</b><span>👍 helpful</span></div><div><b>3</b><span>languages</span></div></div>',
                unsafe_allow_html=True)
    st.markdown("##### 💡 Try a question")
    for ic, title, s in SAMPLES:
        if st.button(f"{ic}  {s}", key=f"side_{s}", use_container_width=True):
            st.session_state["pending"] = s
    st.divider()
    show = st.toggle("Show details (confidence, source)", value=True)
    if st.button("🗑️  Clear chat", use_container_width=True):
        st.session_state["chat"] = []
        st.rerun()
    if os.path.exists(FEEDBACK):
        st.download_button("⬇️  Download feedback log", open(FEEDBACK, "rb").read(), "feedback_log.csv",
                           use_container_width=True)
    st.divider()
    st.markdown('<div style="font-size:.75rem;opacity:.8">Hybrid RAG · bge-m3 + BM25 + reranker · Qwen2.5<br>'
                'Answers come only from verified MU data.</div>', unsafe_allow_html=True)

# =============================== header ===============================
st.markdown("""
<div class="hero">
  <div class="row">
    <div class="logo">🎓</div>
    <div>
      <h1>MU Student Support Chatbot</h1>
      <p>Metropolitan University, Sylhet — routines, courses, fees, exams, buses and offices, any time.</p>
    </div>
  </div>
  <div class="chips">
    <span class="chip live">Online 24/7</span>
    <span class="chip">🇬🇧 English</span><span class="chip">🇧🇩 বাংলা</span><span class="chip">🔤 Banglish</span>
    <span class="chip">✅ Verified answers</span><span class="chip">🛡️ Says "I don't know" instead of guessing</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =============================== welcome screen ===============================
if not chat:
    st.markdown('<div class="welcome"><h3>👋 আসসালামু আলাইকুম! How can I help you today?</h3>'
                '<p>Ask anything about MU in English, বাংলা or Banglish — or tap a question below to start.</p></div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (ic, title, s) in enumerate(SAMPLES):
        if cols[i % 2].button(f"{ic}  **{title}**\n\n{s}", key=f"card_{i}", use_container_width=True):
            st.session_state["pending"] = s
            st.rerun()

# =============================== chat history ===============================
for i, m in enumerate(chat):
    avatar = "🧑‍🎓" if m["role"] == "user" else "🤖"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["text"])
        if m["role"] == "assistant":
            if show and m.get("info"):
                st.markdown(meta_html(m["info"], m["secs"]), unsafe_allow_html=True)
            if not m.get("rated"):
                c1, c2, _ = st.columns([1, 1, 10])
                if c1.button("👍", key=f"up{i}", help="Helpful"):
                    log_feedback(m["q"], m["text"], m["info"], "up"); m["rated"] = "up"; st.rerun()
                if c2.button("👎", key=f"down{i}", help="Not helpful"):
                    log_feedback(m["q"], m["text"], m["info"], "down"); m["rated"] = "down"; st.rerun()
            else:
                st.caption("Thanks for your feedback! 💙" if m["rated"] == "up" else "Thanks — we'll use this to improve. 🙏")

st.markdown('<div class="foot">Developed by Jakia Habiba &amp; Mst. Samia Begum · Supervised by Samia Rahman Rima · '
            'Dept. of CSE, Metropolitan University, Sylhet</div>', unsafe_allow_html=True)

# =============================== new question ===============================
q = st.chat_input("Type your question… (English / বাংলা / Banglish)") or st.session_state.pop("pending", None)
if q:
    chat.append(dict(role="user", text=q))
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(q)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching verified MU information…"):
            a, info, secs = respond(q)
    chat.append(dict(role="assistant", text=a, info=info, secs=secs, q=q))
    st.rerun()
