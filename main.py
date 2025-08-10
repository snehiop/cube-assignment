import os, sys, math, re, json, pathlib, time
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from datetime import datetime
from pathlib import Path

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import yaml
import pandas as pd
import numpy as np

# Optional Google Ads
try:
    from google.ads.googleads.client import GoogleAdsClient
    from google.ads.googleads.errors import GoogleAdsException
except Exception:
    GoogleAdsClient = None
    class GoogleAdsException(Exception): ...
    pass

import requests
from bs4 import BeautifulSoup
from pytrends.request import TrendReq


# ======================== Utils ========================

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("brand", {})
    cfg.setdefault("competitors", [])
    cfg.setdefault("service_locations", [])
    cfg.setdefault("budgets", {})
    cfg.setdefault("fallback", {})
    cfg.setdefault("mode", {"source_priority": ["gkp", "fallback"]})
    cfg.setdefault("domain_keywords", [])
    cfg.setdefault("pmax_theme_rules", {})
    cfg["service_locations"] = [str(x).strip() for x in cfg.get("service_locations", [])]
    cfg["domain_keywords"] = sorted({w.lower().strip() for w in cfg.get("domain_keywords", []) if w})
    return cfg


def build_ads_client_from_env(cfg: Dict[str, Any]) -> Optional[GoogleAdsClient]:
    if GoogleAdsClient is None:
        return None
    try:
        developer_token   = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN", "")
        login_customer_id = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID", "")
        customer_id       = os.getenv("GOOGLE_ADS_CUSTOMER_ID", "")
        client_id         = os.getenv("GOOGLE_ADS_OAUTH_CLIENT_ID", "")
        client_secret     = os.getenv("GOOGLE_ADS_OAUTH_CLIENT_SECRET", "")
        refresh_token     = os.getenv("GOOGLE_ADS_REFRESH_TOKEN", "")

        if not all([developer_token, login_customer_id, client_id, client_secret, refresh_token]):
            return None

        ads_cfg = {
            "developer_token": developer_token,
            "login_customer_id": login_customer_id,
            "use_proto_plus": True,
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }
        client = GoogleAdsClient.load_from_dict(ads_cfg)
        client._extra_customer_id = customer_id or login_customer_id
        client._lang_constant = cfg.get("google_ads", {}).get("language_constant", "languageConstants/1000")
        client._geo_constants = cfg.get("google_ads", {}).get("geo_target_constants", ["geoTargetConstants/2356"])
        return client
    except Exception:
        return None


def safe_to_csv(df: pd.DataFrame, path: Path, attempts: int = 5, sleep_s: float = 0.6) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for _ in range(attempts):
        try:
            df.to_csv(path, index=False)
            return str(path)
        except PermissionError as e:
            last_err = e
            time.sleep(sleep_s)
    alt = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
    df.to_csv(alt, index=False)
    print(f"⚠️ '{path.name}' locked; wrote to '{alt.name}' instead. Last error: {last_err}")
    return str(alt)


# ======================== Google Ads path ========================

def fetch_keyword_ideas(client: GoogleAdsClient, cfg: Dict[str, Any], seeds: Dict[str, Any]) -> pd.DataFrame:
    svc = client.get_service("KeywordPlanIdeaService")
    req = client.get_type("GenerateKeywordIdeasRequest")

    req.customer_id = getattr(client, "_extra_customer_id", "")
    req.language = getattr(client, "_lang_constant", "languageConstants/1000")
    req.geo_target_constants.extend(getattr(client, "_geo_constants", ["geoTargetConstants/2356"]))
    req.keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS

    if seeds.get("keyword_seed"):
        ks = client.get_type("KeywordSeed")
        ks.keywords.extend(seeds["keyword_seed"])
        req.keyword_seed = ks
    if seeds.get("url_seed"):
        us = client.get_type("UrlSeed")
        us.url = seeds["url_seed"]
        req.url_seed = us
    if seeds.get("site_seed"):
        ss = client.get_type("SiteSeed")
        ss.site = seeds["site_seed"]
        req.site_seed = ss

    rows = []
    for idea in svc.generate_keyword_ideas(request=req):
        kw = str(idea.text)
        m = idea.keyword_idea_metrics
        vol  = int(m.avg_monthly_searches or 0)
        low  = (m.low_top_of_page_bid_micros or 0) / 1_000_000
        high = (m.high_top_of_page_bid_micros or 0) / 1_000_000
        if hasattr(m, "competition_index") and m.competition_index is not None:
            comp = float(m.competition_index) / 100.0
        else:
            em = client.enums.KeywordPlanCompetitionLevelEnum.KeywordPlanCompetitionLevel
            enum_map = {em.LOW: 0.25, em.MEDIUM: 0.5, em.HIGH: 0.85}
            comp = enum_map.get(m.competition, 0.5)
        rows.append([kw, vol, low, high, comp])

    df = pd.DataFrame(rows, columns=["keyword","avg_monthly_searches","top_of_page_low","top_of_page_high","competition"])
    return df.drop_duplicates(subset=["keyword"]).reset_index(drop=True)


def safe_fetch_keyword_ideas(client: Optional[GoogleAdsClient], cfg: Dict[str, Any], seeds: Dict[str, Any]) -> pd.DataFrame:
    priority = (cfg.get("mode", {}).get("source_priority") or ["gkp", "fallback"])
    if "gkp" in priority and client is not None:
        try:
            print("Collecting keyword ideas via Google Ads API …")
            return fetch_keyword_ideas(client, cfg, seeds)
        except Exception as e:
            print("⚠️ Google Ads path failed; falling back. Reason:", str(e))

    if "fallback" in priority:
        print("Collecting keyword ideas via FALLBACK …")
        return fallback_keyword_ideas(cfg, seeds)

    raise RuntimeError("No active data sources (gkp disabled and fallback disabled).")


# ======================== Fallback (hardened + failsafe) ========================

STOPWORDS = {
    "a","an","the","of","for","and","or","to","in","with","on","by","from","is","are","be",
    "this","that","your","you","me","my","our","we","at","as","it","its","into","over","all",
    "can","how","what","why","who","where","when","near","best","top"
}

LOCATION_SYNONYMS = {
    "bengaluru": {"bengaluru","bangalore","blr"},
    "mumbai": {"mumbai","bombay","bom"},
    "delhi": {"delhi","new delhi","ncr","delhi ncr"},
    "pune": {"pune"},
    "hyderabad": {"hyderabad","hyd"},
    "chennai": {"chennai","madras"},
    "kolkata": {"kolkata","calcutta"},
}

def _fetch_html(url: str, timeout=15) -> str:
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=timeout)
        if r.ok: return r.text
    except Exception:
        pass
    return ""

def _extract_text_blobs(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    texts = []
    for tag in ["title","h1","h2","h3","p","li","a","strong","em"]:
        for t in soup.find_all(tag):
            txt = (t.get_text(" ", strip=True) or "").strip()
            if txt: texts.append(txt)
    for m in soup.find_all("meta"):
        for attr in ("name","property"):
            name = m.get(attr, "").lower()
            if any(k in name for k in ["keywords","description","og:title","og:description"]):
                content = (m.get("content") or "").strip()
                if content: texts.append(content)
    for s in soup.find_all("script", {"type":"application/ld+json"}):
        try:
            data = json.loads(s.get_text() or "{}")
            texts.append(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass
    return texts

def _normalize_tokens(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    toks = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return toks

def _ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _extract_site_terms(url: str, top_k=40) -> List[str]:
    html = _fetch_html(url)
    if not html: return []
    blobs = _extract_text_blobs(html)
    tokens = []
    for b in blobs:
        tokens.extend(_normalize_tokens(b))
    grams = []
    for n in (2,3,4):
        grams += _ngrams(tokens, n)
    cleaned = []
    for g in grams:
        parts = g.split()
        if not parts: continue
        if parts[0] in STOPWORDS or parts[-1] in STOPWORDS: continue
        if any(p.isdigit() for p in parts): continue
        cleaned.append(g)
    return [w for w,_ in Counter(cleaned).most_common(top_k)]

def _google_autocomplete(seed: str, hl="en") -> List[str]:
    try:
        url = "https://suggestqueries.google.com/complete/search"
        params = {"client":"firefox","q":seed,"hl":hl}
        r = requests.get(url, params=params, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if r.ok:
            data = r.json()
            return [s for s in data[1] if isinstance(s, str)]
    except Exception:
        pass
    return []

def _expand_seeds(cfg: Dict[str, Any], seeds: Dict[str, Any]) -> List[str]:
    pool = set()
    kw_seed = seeds.get("keyword_seed") or []
    if kw_seed:
        pool.update([k.lower() for k in kw_seed if k])
    if seeds.get("url_seed"):
        pool.update(_extract_site_terms(seeds["url_seed"], top_k=40))
    if seeds.get("site_seed"):
        pool.update(_extract_site_terms(seeds["site_seed"], top_k=40))
    for comp in (cfg.get("competitors") or []):
        if comp.get("website"):
            pool.update(_extract_site_terms(comp["website"], top_k=30))
    pool = {p for p in pool if 2 <= len(p.split()) <= 6}

    domain = set(cfg.get("domain_keywords") or [])
    if domain:
        filtered = {p for p in pool if any(d in p for d in domain)}
        if filtered: pool = filtered

    variants = int(cfg.get("fallback",{}).get("autocomplete_variants", 3))
    extra = set()
    for kw in list(pool)[:60]:
        suggestions = _google_autocomplete(kw)
        for s in suggestions:
            s = s.lower()
            if 2 <= len(s.split()) <= 6 and (not domain or any(d in s for d in domain)):
                extra.add(s)
        if len(extra) >= variants * 60:
            break
    pool.update(extra)
    return list(pool)

def _trends_volumes(keywords: List[str], geo="IN") -> Dict[str, int]:
    py = TrendReq(hl="en-US", tz=330)
    out = {}
    batch = 5
    for i in range(0, len(keywords), batch):
        chunk = keywords[i:i+batch]
        try:
            py.build_payload(chunk, timeframe="today 12-m", geo=geo)
            df = py.interest_over_time()
            if df is None or df.empty:
                continue
            for col in chunk:
                if col in df.columns:
                    out[col] = int(df[col].max())
        except Exception:
            time.sleep(0.8)
            continue
    return out

def _estimate_cpc_band(kw: str) -> Tuple[float, float]:
    low, high = 0.5, 2.5
    t = kw.lower()
    high_intent = ["buy","price","cost","deal","hire","service","software","tool","platform",
                   "reviews","reputation","listing","monitoring","survey","feedback","sentiment"]
    if any(w in t for w in high_intent): low, high = 0.9, 4.0
    if any(loc in t for syns in LOCATION_SYNONYMS.values() for loc in syns) or "near me" in t:
        low *= 1.1; high *= 1.1
    if len(t.split()) >= 4: low *= 0.9; high *= 0.9
    return round(low, 2), round(high, 2)

def _estimate_competition(kw: str) -> float:
    t = kw.lower(); c = 0.35
    if any(w in t for w in ["buy","price","hire","service","software","tool","platform","reviews","best","top","compare","vs"]): 
        c += 0.25
    if len(t.split()) >= 4: c -= 0.08
    return max(0.05, min(0.95, c))

def fallback_keyword_ideas(cfg: Dict[str, Any], seeds: Dict[str, Any]) -> pd.DataFrame:
    geo = cfg.get("fallback",{}).get("geo","IN")
    scale = float(cfg.get("fallback",{}).get("scale_per_100", 20000))   # boosted default
    min_vol = int(cfg.get("fallback",{}).get("min_volume", 200))        # softer for testing
    cap = int(cfg.get("fallback",{}).get("max_keywords", 800))

    pool = _expand_seeds(cfg, seeds)
    print(f"[fallback] seed-pool size: {len(pool)} (sample: {pool[:10]})")
    if not pool:
        raise RuntimeError("Fallback: no seeds generated.")

    scores = _trends_volumes(pool[:cap], geo=geo)
    print(f"[fallback] trends returned: {len(scores)} keywords")

    if not scores:
        # Failsafe volumes when Trends is empty or blocked
        syn = {}
        intent_boosters = ["review","reviews","rating","reputation","listing","google",
                           "monitoring","feedback","survey","nps","csat","social","listening",
                           "automation","workflow","ai","outreach","sales"]
        for kw in pool[:cap]:
            base = 25
            if any(w in kw for w in intent_boosters): base += 40
            if len(kw.split()) >= 3: base += 15
            syn[kw] = min(100, base)
        scores = syn
        print("[fallback] trends empty → using synthesized volumes")

    rows = []
    for kw, s in scores.items():
        pseudo_vol = int((s/100.0) * scale)
        if pseudo_vol < min_vol: 
            continue
        low, high = _estimate_cpc_band(kw)
        comp = _estimate_competition(kw)
        rows.append([kw, pseudo_vol, low, high, comp])

    df = pd.DataFrame(rows, columns=["keyword","avg_monthly_searches","top_of_page_low","top_of_page_high","competition"])

    def sig(s: str) -> str:
        toks = [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOPWORDS]
        toks = sorted(toks)
        return " ".join(toks[:6])

    if not df.empty:
        df["sig"] = df["keyword"].apply(sig)
        df = df.sort_values(["avg_monthly_searches","top_of_page_high"], ascending=False).drop_duplicates(subset=["sig"])
        df = df.drop(columns=["sig"]).reset_index(drop=True)
    return df


# ======================== Scoring, Grouping, Themes ========================

def score_row(vol, bid_hi, comp):
    v = math.log1p(max(vol, 0))
    b = math.log1p(max(bid_hi, 0.01))
    c = comp
    return 0.6 * v + 0.55 * b - 0.55 * c

def classify_adgroup(term: str, brand: str, competitors: List[str], locations: List[str]) -> str:
    t = term.lower()
    if brand and brand.lower() in t: return "Brand Terms"
    if any(c.lower() in t for c in competitors if c): return "Competitor Terms"
    if "near me" in t or any(any(loc in t for loc in syns) for syns in LOCATION_SYNONYMS.values()):
        return "Location-based Queries"
    if len(t.split()) >= 4 or any(x in t for x in ["how","what","why","best","guide","vs","compare"]):
        return "Long-Tail Informational Queries"
    return "Category Terms"

def suggest_match_types(term: str, brand: str, competitors: List[str], locations: List[str]) -> List[str]:
    t = term.lower()
    if brand and brand.lower() in t: return ["exact","phrase"]
    if any(c.lower() in t for c in competitors if c): return ["exact"]
    if "near me" in t or any(any(loc in t for loc in syns) for syns in LOCATION_SYNONYMS.values()):
        return ["phrase","exact"]
    if len(t.split()) >= 4: return ["phrase","exact"]
    return ["broad","phrase","exact"]

def prune_and_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["avg_monthly_searches"] >= 200]  # <-- softer for testing; set 500 for final
    df["score"] = df.apply(lambda r: score_row(r["avg_monthly_searches"], r["top_of_page_high"], r["competition"]), axis=1)
    df.sort_values(["score","avg_monthly_searches"], ascending=False, inplace=True)
    return df.reset_index(drop=True)

def enrich_groups_and_matches(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    brand = cfg["brand"].get("name","")
    competitors = [c.get("name","") for c in cfg.get("competitors", [])]
    locations = cfg.get("service_locations", [])
    df = df.copy()
    df["ad_group"] = df["keyword"].apply(lambda k: classify_adgroup(k, brand, competitors, locations))
    df["match_types"] = df["keyword"].apply(lambda k: ",".join(suggest_match_types(k, brand, competitors, locations)))
    df["suggested_cpc"] = ((df["top_of_page_low"] + df["top_of_page_high"]) / 2.0) * (1 + 0.2 * df["competition"])
    return df

def build_pmax_themes(df: pd.DataFrame, cfg: Dict[str, Any], per_theme: int = 8) -> pd.DataFrame:
    rules: Dict[str, List[str]] = cfg.get("pmax_theme_rules", {})
    out = []
    used = set()
    for theme, needles in rules.items():
        patt = "|".join([re.escape(n.lower()) for n in needles])
        mask = df["keyword"].str.contains(patt, case=False, regex=True)
        picks = df[mask].sort_values("score", ascending=False).head(per_theme)
        if not picks.empty:
            used.update(picks["keyword"].tolist())
            out.append({"theme_name": theme, "seed_keywords": ", ".join(picks["keyword"].tolist())})
    for ag, g in df.groupby("ad_group"):
        g2 = g[~g["keyword"].isin(used)].sort_values("score", ascending=False).head(max(4, per_theme//2))
        if not g2.empty:
            out.append({"theme_name": f"{ag} — Theme", "seed_keywords": ", ".join(g2["keyword"].tolist())})
    return pd.DataFrame(out)

def shopping_cpc_suggestions(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if not cfg.get("enable_shopping", False):
        return pd.DataFrame(columns=["cluster","target_cpc","notes"])
    cr = float(cfg.get("conversion_rate", 0.02))
    shop_budget = float(cfg.get("budgets",{}).get("shopping", 0.0))
    shop_df = df[df["ad_group"].isin(["Category Terms", "Brand Terms"])].copy()
    if shop_df.empty or shop_budget <= 0:
        return pd.DataFrame(columns=["cluster","target_cpc","notes"])

    def bigram(s):
        toks = re.findall(r"[a-z0-9]+", s.lower())
        return " ".join(toks[:2]) if len(toks) >= 2 else s.lower()

    shop_df["cluster"] = shop_df["keyword"].apply(bigram)
    agg = shop_df.groupby("cluster").agg(
        vol=("avg_monthly_searches","sum"),
        bid_low=("top_of_page_low","median"),
        bid_high=("top_of_page_high","median"),
    ).reset_index()

    agg["est_cpc_mid"] = (agg["bid_low"] + agg["bid_high"]) / 2.0
    agg["weight"] = agg["vol"] / max(agg["vol"].sum(), 1)
    agg["alloc_budget"] = agg["weight"] * shop_budget
    agg["target_cpa"] = np.maximum(agg["est_cpc_mid"] * 20, 10.0)
    agg["target_cpc"] = agg["target_cpa"] * cr
    agg["target_cpc"] = np.clip(agg["target_cpc"], agg["bid_low"] * 0.8, agg["bid_high"] * 1.05)
    agg["notes"] = "Volume-weighted; clipped to market bid band"
    return agg[["cluster","target_cpc","notes"]].sort_values("target_cpc", ascending=False)


# ======================== Runner ========================

def gather_all_ideas(client: Optional[GoogleAdsClient], cfg: Dict[str, Any]) -> pd.DataFrame:
    brand_url = cfg["brand"].get("website","")
    comp_urls = [c.get("website","") for c in cfg.get("competitors", []) if c.get("website")]
    seed_kw = cfg.get("seed_keywords", []) or []
    mode = (cfg.get("keyword_discovery_mode") or "auto").lower()

    dfs = []
    if mode in ("auto","site"):
        if brand_url:
            dfs.append(safe_fetch_keyword_ideas(client, cfg, {"url_seed": brand_url}))
        for cu in comp_urls:
            dfs.append(safe_fetch_keyword_ideas(client, cfg, {"url_seed": cu}))
    if mode in ("auto","seeds+competitor"):
        if seed_kw:
            dfs.append(safe_fetch_keyword_ideas(client, cfg, {"keyword_seed": seed_kw}))
        for cu in comp_urls:
            dfs.append(safe_fetch_keyword_ideas(client, cfg, {"url_seed": cu}))

    if not dfs:
        raise RuntimeError("No keyword sources produced data. Check config.")
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["keyword"])
    print(f"[gather_all_ideas] collected rows: {len(df)}")
    return df

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    out_dir = pathlib.Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)
    client = None
    if "gkp" in (cfg.get("mode",{}).get("source_priority") or ["gkp","fallback"]):
        client = build_ads_client_from_env(cfg)

    ideas = gather_all_ideas(client, cfg)
    if ideas.empty:
        print("⚠️ No ideas collected. Exiting.")
        return

    ideas = prune_and_score(ideas)
    print(f"[prune_and_score] kept rows: {len(ideas)}")

    ideas = enrich_groups_and_matches(ideas, cfg)

    ideas_out = ideas[[
        "ad_group","keyword","match_types","avg_monthly_searches",
        "top_of_page_low","top_of_page_high","competition","suggested_cpc","score"
    ]].sort_values(["ad_group","score"], ascending=False)

    safe_to_csv(ideas_out, out_dir / "search_campaign_keywords.csv")
    pmax = build_pmax_themes(ideas, cfg, per_theme=8)
    safe_to_csv(pmax, out_dir / "pmax_themes.csv")
    shop = shopping_cpc_suggestions(ideas, cfg)
    safe_to_csv(shop, out_dir / "shopping_cpc_bids.csv")

    print("✅ Done. Files in ./outputs:")
    print(" - search_campaign_keywords.csv")
    print(" - pmax_themes.csv")
    print(" - shopping_cpc_bids.csv")

if __name__ == "__main__":
    main()
