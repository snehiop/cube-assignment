# Cube AI Engineer Assignment — SEM Plan Builder

This repo contains my submission for the **AI Engineer Intern** role at Cube.  
The goal of this project was to automate the process of building a **Search Engine Marketing (SEM) plan** that covers:
- Search campaigns
- Performance Max campaigns
- Manual shopping campaigns

The script takes in brand & competitor details, budgets, and optional seed keywords, then:
1. Collects relevant keywords (via Google Keyword Planner API or a fallback process)
2. Scores and filters them
3. Groups them into logical ad groups
4. Suggests match types
5. Generates PMax themes
6. Calculates shopping CPC suggestions

The output is a set of ready-to-use CSV files that can go straight into campaign planning.

---

## What it does

- **Keyword Collection**
  - Primary: Google Keyword Planner API (if credentials available)
  - Fallback:  
    - Extracts terms from brand & competitor websites  
    - Expands with Google Autocomplete  
    - Estimates search volumes from Google Trends
- **Keyword Scoring**
  - Based on search volume, CPC ranges, and competition
- **Ad Grouping**
  - Brand Terms, Category Terms, Competitor Terms, Location-based Queries, Long-Tail Informational Queries
- **Match Type Suggestions**
  - Exact, Phrase, or Broad depending on keyword intent
- **PMax Themes**
  - Groups keywords into asset themes by category or intent
- **Shopping CPC Suggestions**
  - Target CPCs calculated from budget, CPC benchmarks, and expected conversion rate

---

## Folder structure

```
cube-assignment/
│-- main.py              # Main script
│-- config.yaml          # Configurable inputs (brand, competitors, budgets, etc.)
│-- requirements.txt     # Python dependencies
│-- .gitignore           # Keeps .env, outputs/, venv/ out of Git
│-- README.md            # Project documentation
│-- outputs/             # Generated CSV outputs (ignored in Git)
│   ├── search_campaign_keywords.csv
│   ├── pmax_themes.csv
│   └── shopping_cpc_bids.csv
│-- .env                 # API credentials (ignored in Git)
```

---

## Setup instructions

### 1. Clone the repo
```bash
git clone https://github.com/snehiop/cube-assignment.git
cd cube-assignment
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Google Ads API credentials  
Create a `.env` file in the project root:
```
GOOGLE_ADS_DEVELOPER_TOKEN=your_token
GOOGLE_ADS_LOGIN_CUSTOMER_ID=your_login_id
GOOGLE_ADS_CUSTOMER_ID=your_customer_id
GOOGLE_ADS_OAUTH_CLIENT_ID=your_client_id
GOOGLE_ADS_OAUTH_CLIENT_SECRET=your_client_secret
GOOGLE_ADS_REFRESH_TOKEN=your_refresh_token
```

### 5. Edit `config.yaml`
Fill in:
- Brand name & website
- Competitor name(s) & website(s)
- Service locations
- Ad budgets (shopping, search, pmax)
- Optional seed keywords

---

## Running the script

```bash
python main.py config.yaml
```

When it finishes, you’ll see:
```
✅ Done. Files in ./outputs:
 - search_campaign_keywords.csv
 - pmax_themes.csv
 - shopping_cpc_bids.csv
```

---

## Output details

1. **search_campaign_keywords.csv**
   - Ad group, keyword, match types, search volume, CPC ranges, competition, suggested CPC, score.

2. **pmax_themes.csv**
   - Theme name and seed keywords for each Performance Max asset group.

3. **shopping_cpc_bids.csv**
   - Target CPC per keyword cluster, calculated from budgets and market benchmarks.

---

## Notes
- If GKP API access is blocked, fallback mode will still produce usable results.
- API keys are stored in `.env` and are never committed to GitHub.
- `outputs/` folder is ignored to avoid exposing campaign strategy.


