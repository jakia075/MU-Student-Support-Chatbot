"""mu_rag.py - retrieval, refusal decision and grounded answering for the Metropolitan University chatbot.

Used by the evaluation notebook and by the Streamlit app, so both run exactly the same pipeline.

Pipeline (final system):
  query -> language detection (en / bn / banglish)
        -> entity check (batch "58 A", course code "CSE 0613 1307", bus "bus 03"):
             unknown entity  -> refuse (never guess a neighbouring batch / code)
             known entity    -> only facts that contain that exact entity can be retrieved
        -> hybrid retrieval: bge-m3 dense (questions + answers) + BM25 (questions + answers) + char n-grams, RRF fusion
           (one score per FACT, so the 3 language copies of the same answer never fill the top-k)
        -> cross-encoder reranker (bge-reranker-v2-m3) on the top candidates
        -> decision: answer only if reranker score >= tau AND margin to the 2nd fact >= mu (both tuned on dev)
        -> answer = verified dataset answer in the user's language; optionally rewritten by the LLM,
           but the rewrite is used only if it passes the grounding guard (else the verified answer is returned)
        -> refusals are routed: exam topics -> Exam Office, other MU topics -> MU office, everything else -> out of domain
"""
import re
import math
from collections import Counter, defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

BN_RE = re.compile(r"[ঀ-৿]")
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
LANG_COL = {"en": "answer_en", "bn": "answer_bn", "banglish": "answer_banglish"}


# ----------------------------------------------------------------------------- text helpers
def _stem(w):
    # light English plural stripping ("students" -> "student"), used for BM25 on both the questions and the query
    return w[:-1] if len(w) > 3 and w.isalpha() and w.isascii() and w.endswith("s") and not w.endswith("ss") else w


def tokens(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+|[ঀ-৿]+", str(text).lower().translate(BN_DIGITS))]


def _norm_num(s):
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.lstrip("0") or "0"


def numbers(text):
    t = str(text).translate(BN_DIGITS)
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    return {_norm_num(x) for x in re.findall(r"\d+(?:\.\d+)?", t)}


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def emails(text):
    return {e.lower().rstrip(".") for e in EMAIL_RE.findall(str(text))}


def bn_ratio(text):
    letters = re.findall(r"[A-Za-z\u0980-\u09FF]", str(text))
    if not letters:
        return 0.0
    return sum(bool(BN_RE.match(c)) for c in letters) / len(letters)


def script_ok(text, lang, ref=None):
    """Bangla replies must contain real Bengali script (unless the reference itself is almost all Latin:
    codes, names, emails); English / Banglish replies must be in Latin script."""
    r = bn_ratio(text)
    if lang == "bn":
        if ref is not None and bn_ratio(ref) < 0.08:
            return True
        return r >= 0.08
    return r < 0.08


REFUSAL_RE = re.compile(
    r"(i don't know|i do not know|i don't have|i do not have|don't have (that|this|any|the)|do not have (that|this|any|the)|"
    r"not sure|no information|not available|unable to|cannot (provide|find|answer|access|help)|can't (provide|find|answer|access|help)|"
    r"i'm sorry|i am sorry|sorry|"
    r"জানি না|জানা নেই|তথ্য নেই|আমার কাছে নেই|দুঃখিত|নিশ্চিত নই|"
    r"jani na|jana nei|tottho nei|amar kache nei|dukkhito|sure na)", re.I)


def looks_like_refusal(text):
    return bool(REFUSAL_RE.search(str(text)))


NEG_RE = re.compile(r"\b(no|not|don't|doesn't|does not|isn't|aren't|cannot|can't|never|nei|nai|na)\b|নেই|না\b|নাই", re.I)
YES_START = re.compile(r"^\s*(yes|হ্যাঁ|hae|ha|hya)\b", re.I)
NO_START = re.compile(r"^\s*(no|না|na)\b[.,!।]", re.I)


def polarity(ref):
    if YES_START.search(ref):
        return "yes"
    if NO_START.search(ref):
        return "no"
    return None


CONSENSUS_MAX_CONF = 0.5     # below this reranker score, dense + hybrid agreement may choose the fact (see retrieve)

# ----------------------------------------------------------------------------- query normalisation
CANON = {
    "kto": "koto", "kot": "koto", "ase": "ache", "achhe": "ache", "asey": "ache", "acha": "ache", "ase?": "ache",
    "kothai": "kothay", "kothae": "kothay", "koi": "kothay", "kibhabe": "kivabe", "kemne": "kivabe", "kmne": "kivabe",
    "koita": "koyta", "koiti": "koyta", "koyti": "koyta", "kunta": "konta", "kun": "kon",
    "fi": "fee", "fis": "fee", "fees": "fee", "feez": "fee", "rezalt": "result", "rejalt": "result",
    "librery": "library", "libary": "library", "lybrary": "library", "scolarship": "scholarship",
    "scholership": "scholarship", "pasword": "password", "passwrd": "password", "pass": "password",
    "wai": "wifi", "wi": "wifi", "bas": "bus", "varsity": "university", "versity": "university",
    "xm": "exam", "exm": "exam", "stuff": "staff", "stuf": "staff", "studen": "student", "supplies": "supply", "supplie": "supply", "supplys": "supply", "supples": "supply", "supple": "supply", "eksam": "exam", "rejistration": "registration", "registation": "registration",
    "sir": "", "mam": "", "madam": "", "plz": "", "pls": "", "please": "", "vai": "", "apu": "", "bhai": "",
}
# extra tokens added (not replacing) so both spellings can match the knowledge base
EXPAND = {
    "supply": ["supplementary"], "supli": ["supply", "supplementary"], "sapli": ["supply", "supplementary"],
    "suply": ["supply", "supplementary"], "shapli": ["supply", "supplementary"], "supplimentary": ["supplementary", "supply"],
    "retek": ["retake"], "ritek": ["retake"], "rtk": ["retake"],
    "porikkha": ["exam"], "porikkhar": ["exam"], "porikkhay": ["exam"], "exam": ["porikkha"],
    "vorti": ["admission", "bhorti"], "bhorti": ["admission"], "admission": ["bhorti"],
    "teacher": ["faculty"], "faculty": ["teacher"], "mu": ["metropolitan", "university"], "sathe": ["eksathe"], "ek": ["eksathe"],
}
# Banglish -> rough English gloss, used only as an EXTRA query variant for the dense model and the reranker
GLOSS = {"ki": "what", "kivabe": "how", "kibhabe": "how", "kemne": "how", "koto": "how much", "kobe": "when",
         "kokhon": "when", "kothay": "where", "ke": "who", "kara": "who", "koyta": "how many", "koyjon": "how many",
         "kotojon": "how many", "ache": "is there", "nai": "no", "nei": "no", "er": "of", "theke": "from", "dite": "pay",
         "korte": "do", "korbo": "do", "lagbe": "need", "lage": "need", "jonno": "for", "kon": "which", "konta": "which",
         "porikkha": "exam", "porikkhar": "exam", "vorti": "admission", "bhorti": "admission", "deya": "give",
         "jay": "can", "parbo": "can", "pabo": "get", "shuru": "start", "shesh": "end", "chuti": "holiday", "hoy": ""}
PHRASES = [(r"\bek\s*s?h?athe\b", "eksathe"), (r"\bek\s*(time|somoy|bar)\s*(e|a|ey)?\b", "eksathe"),
           (r"\bekbare\b", "eksathe"), (r"\bat a time\b|\bat once\b|\btogether\b", "at the same time"), (r"\bkoto\s*(gula|ta|ti)\b|\bkotogula\b", "koyta"),
           (r"\b(dewa|deoa|dea)\b", "deya"), (r"\b(newa|neoa)\b", "newa")]

# ----------------------------------------------------------------------------- Bangla loanwords
# Students write English words in Bangla script ("সাপ্লি পরীক্ষার ফি"), while many dataset questions keep them in
# Latin script ("Supplementary পরীক্ষার fee"). The English word is added for BM25 and an English-loanword version of
# the question is used as an extra variant for the dense model and the reranker.
BN_LOAN = {
    "সাপ্লি": "supplementary", "সাপ্লিমেন্টারি": "supplementary", "সাপ্লাই": "supplementary", "রিটেক": "retake",
    "ফি": "fee", "পরীক্ষা": "exam", "রেজাল্ট": "result", "ফলাফল": "result", "রুটিন": "routine", "ক্লাস": "class",
    "কোর্স": "course", "ক্রেডিট": "credit", "ভর্তি": "admission", "ওয়েভার": "waiver", "স্কলারশিপ": "scholarship",
    "বৃত্তি": "scholarship", "সেমিস্টার": "semester", "টার্ম": "term", "রেজিস্ট্রেশন": "registration",
    "নিবন্ধন": "registration", "অ্যাডমিট": "admit", "এডমিট": "admit", "কার্ড": "card", "ল্যাব": "lab", "বাস": "bus",
    "ডিপার্টমেন্ট": "department", "বিভাগ": "department", "শিক্ষক": "teacher", "লাইব্রেরি": "library",
    "হোস্টেল": "hostel", "ক্যান্টিন": "canteen", "ওয়াইফাই": "wifi", "পাসওয়ার্ড": "password",
    "সার্টিফিকেট": "certificate", "ট্রান্সক্রিপ্ট": "transcript", "ফাইনাল": "final", "গ্রেড": "grade",
    "জিপিএ": "gpa", "সিজিপিএ": "cgpa", "উপস্থিতি": "attendance", "অ্যাটেনডেন্স": "attendance", "নোটিশ": "notice",
    "অফিস": "office", "ড্রপ": "drop", "কিস্তি": "installment", "ব্যাচ": "batch", "সেকশন": "section", "রুম": "room",
    "ক্যাম্পাস": "campus", "ইমেইল": "email", "জরিমানা": "fine", "ফাইন": "fine", "থিসিস": "thesis",
    "প্রজেক্ট": "project", "ক্যালেন্ডার": "calendar", "ছুটি": "holiday", "ভাইভা": "viva", "ক্লাসরুম": "classroom",
}
_BN_STEMS = sorted(BN_LOAN, key=len, reverse=True)
_BN_SUF = re.compile("(?:ের|এর|র|ে|তে|কে|টি|টা|গুলো|গুলি|য়|ও)?")


def bn_loan(t):
    """English word for a Bangla-script loanword (with or without a case ending), else None"""
    for st in _BN_STEMS:
        if t.startswith(st) and _BN_SUF.fullmatch(t[len(st):]):
            return BN_LOAN[st]
    return None


def bn_loan_variant(q):
    """'সাপ্লি পরীক্ষার ফি কত?' -> 'supplementary exam fee কত?' (None when nothing changes)"""
    out = re.sub("[\u0980-\u09ff]+", lambda m: bn_loan(m.group(0)) or m.group(0), str(q))
    return out if out != str(q) else None


# ----------------------------------------------------------------------------- teacher short names
# short names / Bangla names of CSE routine teachers -> full name (from the routine's faculty initial list).
# A question that uses only a short name ("rima mam ki class nen?") also gets the full name, so it can match
# the teacher's facts. Only names that belong to exactly one teacher are listed.
TEACHER_ALIAS = {
    'archi': 'Archi Arani Basak',
    'আর্চি': 'Archi Arani Basak',
    'rafsan': 'Ahmed Afif Rafsan',
    'afif': 'Ahmed Afif Rafsan',
    'রাফসান': 'Ahmed Afif Rafsan',
    'alaul': 'Md. Alaul Haque',
    'আলাউল': 'Md. Alaul Haque',
    'aisha': 'Aisha Hayder Chowdhury',
    'আয়েশা': 'Aisha Hayder Chowdhury',
    'istiakur': 'Ahmed Istiakur Rahman',
    'ইস্তিয়াকুর': 'Ahmed Istiakur Rahman',
    'abu jafar': 'Abu Jafar Md Jakaria',
    'আবু জাফর': 'Abu Jafar Md Jakaria',
    'abdullah al masud': 'Abdullah Al Masud',
    'আব্দুল্লাহ আল মাসুদ': 'Abdullah Al Masud',
    'ashikur': 'Ashikur Rahman',
    'আশিকুর': 'Ashikur Rahman',
    'anika': 'Anika Tabassum',
    'অনিকা': 'Anika Tabassum',
    'wadud': 'Abdul Wadud Shakib',
    'ওয়াদুদ': 'Abdul Wadud Shakib',
    'musa': 'Abu Yousuf Musa',
    'মুসা': 'Abu Yousuf Musa',
    'bushra': 'Bushra Azmat Hussain',
    'বুশরা': 'Bushra Azmat Hussain',
    'mukammel': 'Professor Choudhury M. Mukammel Wahid',
    'মুকাম্মেল': 'Professor Choudhury M. Mukammel Wahid',
    'fardin': 'Md. Fardin Ahasan Maraz',
    'maraz': 'Md. Fardin Ahasan Maraz',
    'ফারদিন': 'Md. Fardin Ahasan Maraz',
    'farhana akter': 'Farhana Akter',
    'ফারহানা আক্তার': 'Farhana Akter',
    'fahmidur': 'Md. Fahmidur Rahman Sakib',
    'fahmid': 'Md. Fahmidur Rahman Sakib',
    'ফাহমিদুর': 'Md. Fahmidur Rahman Sakib',
    'naeem': 'Golam Mostofa Naeem',
    'naim': 'Golam Mostofa Naeem',
    'নাঈম': 'Golam Mostofa Naeem',
    'mahdi': 'Md. Imam Mahdi',
    'মাহদী': 'Md. Imam Mahdi',
    'ishrar': 'Ishrar Nazah Chowdhury',
    'ইশরার': 'Ishrar Nazah Chowdhury',
    'mayami': 'Mayami Das Purkayastha Purba',
    'purba': 'Mayami Das Purkayastha Purba',
    'পূর্বা': 'Mayami Das Purkayastha Purba',
    'mahfujul': 'Md. Mahfujul Hasan',
    'mahfuj': 'Md. Mahfujul Hasan',
    'মাহফুজ': 'Md. Mahfujul Hasan',
    'muzammil': 'Muhammad Muzammil',
    'মুজাম্মিল': 'Muhammad Muzammil',
    'limon': 'Md. Shamihul Islam Khan Limon',
    'shamihul': 'Md. Shamihul Islam Khan Limon',
    'লিমন': 'Md. Shamihul Islam Khan Limon',
    'rafee': 'Md. Mushtaq Shahriyar Rafee',
    'rafi': 'Md. Mushtaq Shahriyar Rafee',
    'mushtaq': 'Md. Mushtaq Shahriyar Rafee',
    'রাফি': 'Md. Mushtaq Shahriyar Rafee',
    'shadman': 'Md. Shadman Shakib',
    'শাদমান': 'Md. Shadman Shakib',
    'nabila': 'Nabila Jannat Rifa',
    'rifa': 'Nabila Jannat Rifa',
    'নাবিলা': 'Nabila Jannat Rifa',
    'ruhul': 'Ruhul Amin',
    'রুহুল': 'Ruhul Amin',
    'raisa': 'Raisa Fairooz',
    'fairooz': 'Raisa Fairooz',
    'রাইসা': 'Raisa Fairooz',
    'sadman': 'Sadman Sakib',
    'সাদমান': 'Sadman Sakib',
    'shrabanti': 'Shrabanti Chowdhury',
    'srabonti': 'Shrabanti Chowdhury',
    'shrabonti': 'Shrabanti Chowdhury',
    'srabanti': 'Shrabanti Chowdhury',
    'শ্রাবন্তী': 'Shrabanti Chowdhury',
    'rima': 'Samia Rahman Rima',
    'samia': 'Samia Rahman Rima',
    'রিমা': 'Samia Rahman Rima',
    'tofayel': 'Professor Dr. Tofayel Ahmed',
    'তোফায়েল': 'Professor Dr. Tofayel Ahmed',
    'tajbin': 'Tajbin Jahan',
    'তাজবিন': 'Tajbin Jahan',
}
_ALIAS_LAT = sorted((a for a in TEACHER_ALIAS if a.isascii()), key=len, reverse=True)
_ALIAS_BN = [a for a in TEACHER_ALIAS if not a.isascii()]


def with_teacher_name(q):
    """append the full teacher name when only a short name is used (returns q unchanged otherwise)"""
    s = str(q); low = s.lower(); add = []
    for a in _ALIAS_LAT:
        if re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", low):
            add.append(TEACHER_ALIAS[a])
    for a in _ALIAS_BN:                      # whole word only (+ possessive), so e.g. "পূর্বাভাস" does not match "পূর্বা"
        if re.search(r"(?<![\u0980-\u09ff])" + re.escape(a) + r"(?:ের|র|কে)?(?![\u0980-\u09ff])", s):
            add.append(TEACHER_ALIAS[a])
    add = [n for n in dict.fromkeys(add) if n.lower() not in low]
    return s if not add else s.rstrip(" ?।") + " (" + ", ".join(add) + ")"

# ----------------------------------------------------------------------------- refusal routing
EXAM_TOPIC = re.compile(
    r"(exam|suppl|supply|supli|sapli|suply|supplementary|retake|retek|result|admit|grade|gpa|cgpa|marks?\b|porikkha|semester drop|"
    r"পরীক্ষা|সাপ্লি|সাপ্লিমেন্টারি|রিটেক|ফলাফল|রেজাল্ট|অ্যাডমিট|এডমিট|গ্রেড|সিজিপিএ)", re.I)
MU_TOPIC = re.compile(
    r"(\bmu\b|metropolitan|university|varsity|campus|department|dept|teacher|faculty|\bsir\b|madam|fee|waiver|scholarship|"
    r"admission|vorti|bhorti|class|routine|course|credit|\bbus\b|library|hostel|office|semester|\bterm\b|cse|eee|bba|swe|llb|"
    r"মেট্রোপলিটন|ভার্সিটি|বিশ্ববিদ্যালয়|ইউনিভার্সিটি|ক্যাম্পাস|বিভাগ|ডিপার্টমেন্ট|শিক্ষক|স্যার|ফি|ভর্তি|ক্লাস|রুটিন|কোর্স|ক্রেডিট|বাস|লাইব্রেরি|অফিস|সেমিস্টার)", re.I)

REFUSE_MSG = {
    "exam": {
        "en": "I'm sorry, I don't have that information. Please contact the Exam Office (Room 108, 1st floor; 01757-537754 or +8802996642698; exam@metrouni.edu.bd).",
        "bn": "দুঃখিত, এই তথ্য আমার কাছে নেই। Exam Office-এ যোগাযোগ করো (Room 108, 1st floor; 01757-537754 বা +8802996642698; exam@metrouni.edu.bd)।",
        "banglish": "Dukkhito, ei tottho amar kache nei. Exam Office e jogajog koro (Room 108, 1st floor; 01757-537754 ba +8802996642698; exam@metrouni.edu.bd).",
    },
    "mu": {
        "en": "I'm sorry, I don't know that. Please contact the Metropolitan University One Stop service (01757-537754, 1stop@metrouni.edu.bd) or info@metrouni.edu.bd.",
        "bn": "দুঃখিত, এটা আমি জানি না। Metropolitan University-র One Stop service (01757-537754, 1stop@metrouni.edu.bd) অথবা info@metrouni.edu.bd-তে যোগাযোগ করো।",
        "banglish": "Dukkhito, eta ami jani na. Metropolitan University r One Stop service (01757-537754, 1stop@metrouni.edu.bd) othoba info@metrouni.edu.bd e jogajog koro.",
    },
    "other": {
        "en": "Sorry, I only have information about Metropolitan University, Sylhet. Please check that university's own website or office.",
        "bn": "দুঃখিত, আমার কাছে শুধু Metropolitan University, Sylhet-এর তথ্য আছে। ওই university-র নিজস্ব website বা office-এ খোঁজ নাও।",
        "banglish": "Dukkhito, amar kache shudhu Metropolitan University, Sylhet er tottho ache. Oi university r nijer website ba office e khoj nao.",
    },
    "ood": {
        "en": "Sorry, I can only answer questions about Metropolitan University, Sylhet (courses, routines, exams, fees, faculty, bus and campus).",
        "bn": "দুঃখিত, আমি শুধু Metropolitan University, Sylhet সম্পর্কিত প্রশ্নের উত্তর দিতে পারি (course, routine, পরীক্ষা, fee, faculty, bus ও campus)।",
        "banglish": "Dukkhito, ami shudhu Metropolitan University, Sylhet niye proshner uttor dite pari (course, routine, exam, fee, faculty, bus ar campus).",
    },
}


OTHER_UNI = re.compile(
    r"\b(brac|nsu|north\s*south|east\s*west|ewu|aiub|american international|iub|independent university|uiu|united international|"
    r"daffodil|diu|buet|dhaka university|university of dhaka|rajshahi university|chittagong university|jahangirnagar|"
    r"sust|shahjalal|sau|sylhet agricultural|leading university|north\s*east university|neub|sylhet international|siu|iut|"
    r"kuet|ruet|cuet|ulab|aust|ahsanullah|bup|manarat|stamford|green university|southeast university|presidency university|"
    r"harvard|oxford|cambridge|\bmit\b|stanford)\b|ব্র্যাক|নর্থ সাউথ|শাবিপ্রবি|শাহজালাল|ঢাকা বিশ্ববিদ্যালয়|বুয়েট|লিডিং ইউনিভার্সিটি|"
    r"নর্থ ইস্ট|রাজশাহী বিশ্ববিদ্যালয়|চট্টগ্রাম বিশ্ববিদ্যালয়|জাহাঙ্গীরনগর|ড্যাফোডিল|ইস্ট ওয়েস্ট", re.I)


def mentions_other_university(q):
    return bool(OTHER_UNI.search(str(q))) and not re.search(r"metropolitan|\bmu\b|মেট্রোপলিটন", str(q), re.I)


def refusal_route(q, status="ok"):
    if status == "other_university":
        return "other"
    if EXAM_TOPIC.search(str(q)):
        return "exam"
    if status == "unknown_entity" or MU_TOPIC.search(str(q)):
        return "mu"
    return "ood"


# ----------------------------------------------------------------------------- entities
BATCH_RE = re.compile(r"(?<![\d.:/])\b([5-7]\d)\s*-?\s*([a-i](?:\s*\+\s*[a-i])*)\b(?![.:]\w)", re.I)
CODE_RE = re.compile(r"\b([a-z]{2,4})\s*-?\s*(\d{3,4})(?:\s*-?\s*(\d{4}))?\b", re.I)
CODE2_RE = re.compile(r"\b([a-z]{2,4})-(\d{1,2})\b", re.I)
BUS_RE = re.compile(r"\bbus\s*(?:no\.?|number|nombor|nong|num)?\s*#?\s*(\d{1,2})\b(?![:.]\d)", re.I)
BUS_BN_RE = re.compile(r"(\d{1,2})\s*(?:নম্বর|নং)\s*বাস|বাস\s*(?:নম্বর|নং)?\s*(\d{1,2})\b(?![:.]\d)")


RT_RE = re.compile(r"(?<![A-Za-z0-9&+])[A-Z][A-Z&]{1,5}(?![A-Za-z0-9&])")


def _batches(text):
    out = set()
    for m in BATCH_RE.finditer(str(text).translate(BN_DIGITS)):
        for letter in re.split(r"\s*\+\s*", m.group(2)):
            out.add(f"{m.group(1)} {letter.upper()}")
    return out


def _codes(text, prefixes):
    t = str(text).translate(BN_DIGITS)
    out = set()
    for m in CODE_RE.finditer(t):
        p = m.group(1).upper()
        if p in prefixes:
            digits = m.group(2) + (m.group(3) or "")
            out.add((p, digits, m.group(3) or m.group(2)))
    for m in CODE2_RE.finditer(t):
        p = m.group(1).upper()
        if p in prefixes:
            out.add((p, m.group(2), m.group(2)))
    return out


def _buses(text):
    t = str(text).translate(BN_DIGITS)
    out = {str(int(m.group(1))) for m in BUS_RE.finditer(t)}
    for m in BUS_BN_RE.finditer(t):
        out.add(str(int(m.group(1) or m.group(2))))
    return out


def _year_like(code):
    p, digits, last = code
    return len(digits) == 4 and 2019 <= int(digits) <= 2031


# ----------------------------------------------------------------------------- main class
class MURAG:
    """embed(texts) -> L2-normalised np.ndarray ; rerank(query, passages) -> np.ndarray of scores in [0, 1]."""

    W = {"dense_q": 1.0, "dense_a": 0.5, "bm25_q": 0.7, "bm25_a": 0.3, "char_q": 0.4}

    def __init__(self, kb, facts, embed=None, rerank=None, n_rerank=20, rrf_k=60):
        self.kb = kb.reset_index(drop=True).copy()
        self.facts = facts.set_index("fact_id", drop=False)
        self.fids = list(self.facts.index)
        self.fpos = {f: i for i, f in enumerate(self.fids)}
        self.nf = len(self.fids)
        self.row_f = np.array([self.fpos[f] for f in self.kb["fact_id"]])
        self.kind = self.facts["kind"].to_dict()
        self.embed, self.rerank_fn, self.n_rerank, self.rrf_k = embed, rerank, n_rerank, rrf_k
        self.ans_docs = [(self.fpos[f], lang, str(self.facts.at[f, col])) for f in self.fids for lang, col in LANG_COL.items()]
        self.ans_f = np.array([d[0] for d in self.ans_docs])
        self._q_cache, self._rr_cache, self._tok_cache, self._ch_cache = {}, {}, {}, {}
        self._build_lexical()
        self._build_entities()
        self._build_language_model()
        self._build_rare_tokens()
        if embed is not None:
            self.q_emb = np.asarray(embed(list(self.kb["question"])), dtype="float32")
            self.a_emb = np.asarray(embed([d[2][:1200] for d in self.ans_docs]), dtype="float32")

    # ------------------------------------------------------------------ building
    def _build_lexical(self):
        qs = list(self.kb["question"])
        self.vocab = Counter(t for q in qs for t in tokens(q))
        self.vocab_lat = [w for w in self.vocab if w.isascii() and w.isalpha() and len(w) >= 3]
        self.vocab_bn = [w for w in self.vocab if not w.isascii() and len(w) >= 3]
        self.bm25_q = BM25Okapi([tokens(q) for q in qs])
        self.bm25_a = BM25Okapi([tokens(d[2]) for d in self.ans_docs])
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(qs)
        self.char_q = self.char.transform(qs)
        # representative question of each fact per language (first dataset row) for the reranker passage
        self.rep_q = defaultdict(dict)
        for r in self.kb.itertuples():
            self.rep_q[self.fpos[r.fact_id]].setdefault(r.language, r.question)

    def _build_entities(self):
        cats = self.facts["category"].to_dict()
        code_rows = self.kb[self.kb["fact_id"].map(cats).isin(["curriculum", "supplementary routine"])]
        self.prefixes = {m.group(1).upper() for q in code_rows["question"] for m in CODE_RE.finditer(q)}
        self.prefixes |= {m.group(1).upper() for q in code_rows["question"] for m in CODE2_RE.finditer(q)}
        self.fact_ent = defaultdict(lambda: {"batch": set(), "code": set(), "bus": set()})
        for r in self.kb.itertuples():
            fe = self.fact_ent[self.fpos[r.fact_id]]
            fe["batch"] |= _batches(r.question)
            fe["code"] |= _codes(r.question, self.prefixes)
            fe["bus"] |= _buses(r.question)
        self.ans_ent = {"batch": set(), "code": set(), "bus": set()}
        for _, _, a in self.ans_docs:
            self.ans_ent["batch"] |= _batches(a)
            self.ans_ent["code"] |= _codes(a, self.prefixes)
        self.ans_ent["bus"] = {b for fe in self.fact_ent.values() for b in fe["bus"]}
        # routine abbreviations / faculty initials ("MP&I", "DBMS", "SAS"): case-sensitive, soft filter only
        rt_rows = self.kb[self.kb["fact_id"].map(cats) == "class routine"]
        other = Counter(t for q in self.kb[self.kb["fact_id"].map(cats) != "class routine"]["question"] for t in set(RT_RE.findall(q)))
        cand = Counter(t for q in rt_rows["question"] for t in set(RT_RE.findall(q)))
        self.rt_vocab = {t for t in cand if not other[t] and t not in self.prefixes and (len(t) >= 3 or "&" in t)
                         and t not in {"CSE", "AUTUMN"}}
        self.rt_facts = defaultdict(set)
        for r in rt_rows.itertuples():
            for t in set(RT_RE.findall(r.question)) & self.rt_vocab:
                self.rt_facts[t].add(self.fpos[r.fact_id])

    def _build_language_model(self):
        cnt = {"en": Counter(), "banglish": Counter()}
        for r in self.kb.itertuples():
            if r.language in cnt:
                cnt[r.language].update(t for t in tokens(r.question) if t.isalpha())
        tot = {k: sum(v.values()) for k, v in cnt.items()}
        V = len(set(cnt["en"]) | set(cnt["banglish"])) + 1
        self.lang_llr = {}
        for w in set(cnt["en"]) | set(cnt["banglish"]):
            pb = (cnt["banglish"][w] + 1) / (tot["banglish"] + V)
            pe = (cnt["en"][w] + 1) / (tot["en"] + V)
            self.lang_llr[w] = math.log(pb / pe)

    def _build_rare_tokens(self):
        self.df = {}
        for lang, col in LANG_COL.items():
            c = Counter()
            for a in self.facts[col]:
                c.update(set(self._content_tokens(a)))
            self.df[lang] = c
        self.rare_cut = max(3, int(0.01 * self.nf))

    @staticmethod
    def _content_tokens(text):
        return [t for t in tokens(text) if not t.isdigit() and (len(t) >= 3 if t.isascii() else len(t) >= 2)]

    # ------------------------------------------------------------------ language
    def detect_language(self, q):
        if BN_RE.search(str(q)):
            return "bn"
        s = sum(self.lang_llr.get(t, 0.0) for t in tokens(q) if t.isalpha())
        return "banglish" if s > 0.5 else "en"

    # ------------------------------------------------------------------ normalisation
    def _correct(self, t):
        if t in self._tok_cache:
            return self._tok_cache[t]
        out = [t]
        if t not in self.vocab and len(t) >= 4 and not t.isdigit():
            pool, cut = (self.vocab_lat, 82) if t.isascii() else (self.vocab_bn, 85)
            m = process.extractOne(t, pool, scorer=fuzz.ratio, score_cutoff=cut)
            if m and abs(len(m[0]) - len(t)) <= 2:
                out.append(m[0])
        self._tok_cache[t] = out
        return out

    def normalize(self, q):
        q = str(q).lower().translate(BN_DIGITS)
        for pat, rep in PHRASES:
            q = re.sub(pat, rep, q)
        out = []
        for t in tokens(q):
            t = CANON.get(t, t)
            if not t:
                continue
            for c in self._correct(t):
                out.append(c)
                out.extend(EXPAND.get(c, []))
            if not t.isascii():
                e = bn_loan(t)
                if e:
                    out.append(e); out.extend(EXPAND.get(e, []))
        return out

    def clean_query(self, q):
        """spelling-corrected version of the query (same words, typos fixed against the dataset vocabulary)"""
        t = str(q).lower().translate(BN_DIGITS)
        for pat, rep_ in PHRASES:
            t = re.sub(pat, rep_, t)
        out = []
        for w in tokens(t):
            w = CANON.get(w, w)
            if not w:
                continue
            c = self._correct(w)
            out.append(c[1] if len(c) > 1 else w)
        return " ".join(out)

    def variants(self, q, lang=None):
        """the raw query + its spelling-corrected form (+ an English gloss for Banglish); dense and reranker
        scores are the maximum over these variants, which helps Banglish and misspelled questions"""
        lang = lang or self.detect_language(q)
        v = [str(q).strip()]
        c = self.clean_query(q)
        if c and c != v[0].lower():
            v.append(c)
        if lang == "banglish":
            g = " ".join(x for x in (GLOSS.get(w, w) for w in (c or str(q).lower()).split()) if x)
            if g and g not in v:
                v.append(g)
        if lang == "bn":
            b = bn_loan_variant(q)                  # "সাপ্লি পরীক্ষার ফি কত?" -> "supplementary exam fee কত?"
            if b:
                v.append(b)
        t = with_teacher_name(q)                    # "rima mam ki class nen?" -> "... (Samia Rahman Rima)"
        if t != str(q):
            v.append(t)
        return list(dict.fromkeys(v))

    # ------------------------------------------------------------------ entity filter
    def entity_filter(self, q):
        """returns (status, allowed fact positions or None, detail, hard) - hard = an exact batch/code/bus matched"""
        q_ent = {"batch": _batches(q), "code": {c for c in _codes(q, self.prefixes)}, "bus": _buses(q)}
        allowed = None
        for typ, ents in q_ent.items():
            for e in ents:
                if typ == "code":
                    match = {f for f, fe in self.fact_ent.items()
                             if any(c[0] == e[0] and e[1] in (c[1], c[2]) for c in fe["code"])}
                    in_answers = any(c[0] == e[0] and e[1] in (c[1], c[2]) for c in self.ans_ent["code"])
                    if not match and not in_answers and _year_like(e):
                        continue
                else:
                    match = {f for f, fe in self.fact_ent.items() if e in fe[typ]}
                    in_answers = e in self.ans_ent[typ]
                if match:
                    allowed = match if allowed is None else (allowed & match or allowed | match)
                elif not in_answers:
                    return "unknown_entity", None, f"{typ}:{e if typ != 'code' else e[0] + ' ' + e[1]}", False
        hard = allowed is not None
        if allowed is None:                     # soft: only narrows, never refuses
            rts = [self.rt_facts[t] for t in set(RT_RE.findall(str(q))) if t in self.rt_vocab]
            if rts:
                allowed = set.intersection(*rts) or set.union(*rts)
        return "ok", allowed, "", hard

    # ------------------------------------------------------------------ scoring channels (one score per fact)
    def _fact_max(self, row_scores, owner):
        out = np.full(self.nf, -1e9)
        np.maximum.at(out, owner, row_scores)
        return out

    def _qvec(self, q):
        if q not in self._q_cache:
            self._q_cache[q] = np.asarray(self.embed([q])[0], dtype="float32")
        return self._q_cache[q]

    def channels(self, q):
        if q in self._ch_cache:
            return self._ch_cache[q]
        ntoks = self.normalize(q)
        nq = " ".join(ntoks) or str(q).lower()
        ch = {}
        if self.embed is not None:
            v = self._qvec(q)
            ch["dense_q"] = self._fact_max(self.q_emb @ v, self.row_f)
            ch["dense_a"] = self._fact_max(self.a_emb @ v, self.ans_f)
            raw_q, raw_a = ch["dense_q"], ch["dense_a"]
            for extra in self.variants(q)[1:]:
                ve = self._qvec(extra)
                raw_q = np.maximum(raw_q, self._fact_max(self.q_emb @ ve, self.row_f))
                raw_a = np.maximum(raw_a, self._fact_max(self.a_emb @ ve, self.ans_f))
            ch["dense_q_var"], ch["dense_a_var"] = raw_q, raw_a
        ch["bm25_q"] = self._fact_max(self.bm25_q.get_scores(ntoks), self.row_f)
        ch["bm25_a"] = self._fact_max(self.bm25_a.get_scores(ntoks), self.ans_f)
        ch["char_q"] = self._fact_max((self.char_q @ self.char.transform([nq]).T).toarray().ravel(), self.row_f)
        if len(self._ch_cache) < 20000:
            self._ch_cache[q] = ch
        return ch

    def rrf(self, ch, use=None, allowed=None):
        """weighted Reciprocal Rank Fusion. With an entity filter the ranks are computed among the allowed facts only,
        so a fact that exactly matches the batch / code is not penalised by facts that were filtered out."""
        total = np.zeros(self.nf)
        idx = np.arange(self.nf) if allowed is None else np.fromiter(sorted(allowed), dtype=int)
        for name, s in ch.items():
            if name not in self.W or (use is not None and name not in use):
                continue
            if name in ("dense_q", "dense_a") and name + "_var" in ch:
                s = ch[name + "_var"]          # hybrid systems use the best of the query variants
            sub = s[idx]
            pos = np.empty(len(idx))
            pos[np.argsort(-sub, kind="stable")] = np.arange(len(idx))
            total[idx] += self.W[name] / (self.rrf_k + pos + 1)
        return total

    # ------------------------------------------------------------------ rankers
    def _ranking(self, scores, allowed=None, k=10):
        if allowed is not None:
            mask = np.full(self.nf, -np.inf)
            idx = np.fromiter(allowed, dtype=int)
            mask[idx] = scores[idx]
            scores = mask
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.fids[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def passage(self, f, lang):
        qrep = self.rep_q[f].get(lang) or self.rep_q[f].get("en")
        acol = "answer_bn" if lang == "bn" else "answer_en"
        return f"{qrep}\n{str(self.facts.iat[f, self.facts.columns.get_loc(acol)])[:700]}"

    def rerank(self, q, cand_pos, lang):
        key = (q, tuple(cand_pos))
        if key not in self._rr_cache:
            passages = [self.passage(f, lang) for f in cand_pos]
            s = None
            for v in self.variants(q, lang):        # best score over raw / corrected / glossed query
                sv = np.asarray(self.rerank_fn(v, passages), dtype="float64")
                s = sv if s is None else np.maximum(s, sv)
            self._rr_cache[key] = s
        return self._rr_cache[key]

    def retrieve(self, q, system="final", k=10):
        """system: bm25 | dense | hybrid | hybrid_entity | final
        returns dict(ranking=[(fact_id, score)], status, conf, margin, sim, lang)
          conf   = the system's own score of the top fact (cosine / fused RRF / reranker probability)
          margin = conf(top-1) - conf(top-2)
          sim    = absolute evidence for the top fact: its dense cosine (char-n-gram cosine without a dense model),
                   1.0 when an exact batch / course code / bus number matched. RRF scores are rank-based, so a
                   refusal rule needs this absolute signal as well."""
        lang = self.detect_language(q)
        status, allowed, detail, hard = "ok", None, "", False
        if system in ("hybrid_entity", "final"):
            if mentions_other_university(q):        # never answer MU facts for a question about another university
                return dict(ranking=[], status="other_university", detail="", conf=0.0, margin=0.0, sim=0.0, lang=lang)
            status, allowed, detail, hard = self.entity_filter(q)
            if status == "unknown_entity":
                return dict(ranking=[], status=status, detail=detail, conf=0.0, margin=0.0, sim=0.0, lang=lang)
        ch = self.channels(q)
        if system == "dense":                      # old RAG baseline: raw query only
            absolute = ch["dense_q"] if "dense_q" in ch else ch["char_q"]
        else:
            absolute = ch.get("dense_q_var", ch.get("dense_q", ch["char_q"]))
        if system == "bm25":
            scores = ch["bm25_q"]
        elif system == "dense":
            scores = absolute
        else:
            scores = self.rrf(ch, allowed=allowed)
        hybrid_top = self._ranking(scores, allowed, k=1)
        consensus = False
        if system == "final" and self.rerank_fn is not None:
            # rerank the union of the hybrid top-N and the dense top-N (inside the entity filter)
            cand = [self.fpos[f] for f, _ in self._ranking(scores, allowed, k=self.n_rerank)]
            cand += [self.fpos[f] for f, _ in self._ranking(absolute, allowed, k=self.n_rerank // 2)]
            cand = list(dict.fromkeys(cand))
            rr = self.rerank(q, cand, lang)
            order = np.argsort(-rr, kind="stable")
            ranking = [(self.fids[cand[i]], float(rr[i])) for i in order]
            # the cross-encoder is weak on code-mixed Banglish: when it is unsure (top score < 0.5) but the dense
            # retriever and the hybrid (lexical + dense) retriever independently rank the SAME fact first, that fact
            # is put first. It can then only be answered through the "retrievers agree and similarity >= alpha"
            # path of the rule, whose alpha is tuned on dev with the wrong-answer penalty.
            dense_top = self._ranking(absolute, allowed, k=1)
            if (ranking and ranking[0][1] < CONSENSUS_MAX_CONF and hybrid_top and dense_top
                    and hybrid_top[0][0] == dense_top[0][0] != ranking[0][0]):
                pick = [x for x in ranking if x[0] == hybrid_top[0][0]]
                if pick:
                    ranking = pick + [x for x in ranking if x[0] != pick[0][0]]
                    consensus = True
        else:
            ranking = self._ranking(scores, allowed, k=max(k, 2))
        conf = ranking[0][1] if ranking else 0.0
        margin = conf - ranking[1][1] if len(ranking) > 1 else conf
        dense = float(absolute[self.fpos[ranking[0][0]]]) if ranking else 0.0
        sim = 1.0 if hard else dense
        agree = bool(ranking) and bool(hybrid_top) and hybrid_top[0][0] == ranking[0][0]
        return dict(ranking=ranking[:k], status=status, detail=detail, conf=conf, margin=margin, sim=sim, dense=dense,
                    agree=agree, lang=lang, consensus=consensus)

    # ------------------------------------------------------------------ decision
    @staticmethod
    def decide(r, rule):
        """Answer (True) or say "I don't know" (False). rule = {"direct": cos or None, "by_lang": {lang: sub-rule}}
        with sub-rule = {"tau": reranker score, "sigma": similarity floor, "alpha": floor when retrievers agree}.
        Answer if: the question is almost identical to a dataset question (dense >= direct), OR
                   score >= tau and similarity >= sigma, OR
                   the reranker and the hybrid retriever pick the same fact and similarity >= alpha."""
        if r["status"] != "ok" or not r["ranking"]:
            return False
        if rule.get("direct") is not None and r.get("dense", 0.0) >= rule["direct"]:
            return True
        R = rule.get("by_lang", {}).get(r.get("lang"), rule)
        if r["conf"] >= R["tau"] and r.get("sim", 1.0) >= R.get("sigma", -1.0):
            return True
        a = R.get("alpha")
        return a is not None and bool(r.get("agree")) and r.get("sim", 0.0) >= a

    @staticmethod
    def accept(r, tau, mu=0.0, sigma=-1.0, direct=None):
        """answer if (score >= tau and similarity >= sigma), or if the question is almost identical to a dataset
        question (dense cosine >= direct). Unknown entities / other universities are never answered."""
        if r["status"] != "ok" or not r["ranking"]:
            return False
        if direct is not None and r.get("dense", 0.0) >= direct:
            return True
        return r["conf"] >= tau and r["margin"] >= mu and r.get("sim", 1.0) >= sigma

    def verified_answer(self, fid, lang):
        return str(self.facts.at[fid, LANG_COL[lang]])

    # ------------------------------------------------------------------ grounding guard / key-fact coverage
    def key_items(self, ref, lang):
        rare = []
        df = self.df[lang if lang in self.df else "en"]
        seen = set()
        for t in self._content_tokens(ref):
            if t in seen:
                continue
            seen.add(t)
            d = df.get(t, 0)
            if d <= self.rare_cut:
                rare.append((d, -len(t), t))
        rare = [t for _, _, t in sorted(rare)[:8]]
        return dict(numbers=numbers(ref), emails=emails(ref), rare=rare, polarity=polarity(ref))

    def covers(self, answer, ref, lang):
        """True if `answer` states the key facts of `ref`: every number and email, at least half of the rare
        content words, and the same yes/no polarity."""
        k = self.key_items(ref, lang)
        a_low = str(answer).lower()
        a_nums, a_mail = numbers(answer), emails(answer)
        if not k["numbers"] <= a_nums or not k["emails"] <= a_mail:
            return False
        if k["rare"]:
            hit = sum(t in a_low for t in k["rare"])
            if hit < math.ceil(0.5 * len(k["rare"])):
                return False
        if k["polarity"] is not None:
            neg = bool(NEG_RE.search(str(answer)[:160]))
            if (k["polarity"] == "yes") == neg and not YES_START.search(str(answer)):
                return False
            if k["polarity"] == "no" and not neg:
                return False
        return True

    def guard(self, gen, verified, lang):
        probs = []
        g = str(gen or "").strip()
        if len(g) < 3:
            return ["empty"]
        if not script_ok(g, lang, verified):
            probs.append("wrong_language")
        extra = numbers(g) - numbers(verified)
        if extra:
            probs.append("unsupported_numbers:" + ",".join(sorted(extra)[:5]))
        if emails(g) - emails(verified):
            probs.append("unsupported_email")
        if looks_like_refusal(g) and not looks_like_refusal(verified):
            probs.append("needless_refusal")
        if not self.covers(g, verified, lang):
            probs.append("missing_key_facts")
        return probs