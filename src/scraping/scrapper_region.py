#!/usr/bin/env python3

import json
import re
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

# ───────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────
WINE_CSV = "data/Wine/vivino_wines.csv"
OUT_JSON = "vivino_region_tree.json"
DELAY = 2.0  # seconds between requests
TIMEOUT = 30  # selenium timeout


def load_regions_from_wines(csv_path: str) -> set[str]:
    """Load unique region slugs from wine dataset"""
    try:
        df = pd.read_csv(csv_path)
        regions = df["region"].dropna().unique()
        return {r.strip().lower().replace(" ", "-") for r in regions}
    except Exception as e:
        print(f"❌ Error loading wine data: {e}")
        return set()


def create_selenium_driver():
    """Create a headless Chrome driver optimized for Vivino"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    # French locale preferences
    prefs = {"intl.accept_languages": "fr-FR,fr,en-US,en"}
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--lang=fr")

    driver = webdriver.Chrome(options=options)
    return driver


def fetch_region_info(slug: str) -> dict | None:
    """Fetch complete region information using Selenium"""
    driver = create_selenium_driver()

    try:
        # Use French locale URL for better region data
        url = f"https://www.vivino.com/wine-regions/{slug}?country_code=fr&language=fr"
        print(f"🔄 Fetching: {slug}")

        driver.get(url)

        # Wait for page to load completely
        try:
            WebDriverWait(driver, TIMEOUT).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            time.sleep(3)  # Additional wait for dynamic content
        except TimeoutException:
            print(f"    ⚠️ Timeout loading {slug}")

        # Get rendered HTML
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # Extract region data
        result = extract_region_data(soup, slug, html)

        if result:
            print(
                f"    ✅ Success: {result['name']} → {result.get('full_hierarchy', 'No hierarchy')}"
            )
        else:
            print(f"    ❌ Failed to extract data for {slug}")

        return result

    except Exception as e:
        print(f"    ❌ Error fetching {slug}: {e}")
        return None

    finally:
        driver.quit()


def extract_region_data(soup: BeautifulSoup, slug: str, html: str) -> dict | None:
    """Extract complete region hierarchy from rendered page"""
    try:
        print(f"    🎯 Extracting data for {slug}")

        # 1. Extract clean region name
        name = extract_region_name(soup, slug)

        # 2. Extract hierarchical parents
        parents = extract_region_parents(soup, slug, html)

        # 3. Extract children regions
        children = extract_region_children(soup, slug, parents)

        # 4. Build complete hierarchy
        full_hierarchy = " > ".join(parents + [name])

        print(f"      📊 Hierarchy: {full_hierarchy}")
        print(f"      📊 Children: {len(children)} found")

        return {
            "slug": slug,
            "name": name,
            "parents": parents,
            "children": children,
            "full_hierarchy": full_hierarchy,
            "hierarchy_depth": len(parents),
            "source": "selenium_complete",
        }

    except Exception as e:
        print(f"    ❌ Extraction error: {e}")
        return None


def extract_region_name(soup: BeautifulSoup, slug: str) -> str:
    """Extract clean region name from page"""
    # Default fallback
    name = slug.replace("-", " ").title()

    # Try page title
    title = soup.find("title")
    if title:
        title_text = title.get_text()
        clean_title = re.sub(r"\s*\|\s*.*$", "", title_text).strip()
        if clean_title and len(clean_title) > 2:
            name = clean_title

    # Try h1 heading
    h1 = soup.find("h1")
    if h1:
        h1_text = h1.get_text(strip=True)
        if h1_text and "wine" not in h1_text.lower() and len(h1_text) < 50:
            name = h1_text

    return name


def extract_region_parents(soup: BeautifulSoup, slug: str, html: str) -> list[str]:
    """Extract parent regions using multiple methods"""
    parents = []

    # Method 1: Breadcrumb navigation
    parents = extract_from_breadcrumbs(soup, slug)
    if parents:
        print(f"      🍞 Found breadcrumb hierarchy: {' > '.join(parents)}")
        return parents

    # Method 2: Page text patterns
    parents = extract_from_text_patterns(soup, slug)
    if parents:
        print(f"      📝 Found text hierarchy: {' > '.join(parents)}")
        return parents

    # Method 3: French wine region knowledge base
    parents = build_french_hierarchy(soup, slug)
    if parents:
        print(f"      🇫🇷 Built French hierarchy: {' > '.join(parents)}")
        return parents

    return []


def extract_from_breadcrumbs(soup: BeautifulSoup, slug: str) -> list[str]:
    """Extract hierarchy from breadcrumb navigation"""
    breadcrumb_selectors = [
        ".breadcrumb",
        '[data-testid*="breadcrumb"]',
        "nav ol",
        "nav ul",
        ".navigation-path",
        '[class*="breadcrumb"]',
        '[class*="path"]',
        '[class*="nav"]',
    ]

    for selector in breadcrumb_selectors:
        breadcrumbs = soup.select(selector)
        for breadcrumb in breadcrumbs:
            links = breadcrumb.find_all("a", href=re.compile(r"/wine-regions/"))
            if len(links) >= 1:
                parents = []
                for link in links:
                    parent_name = link.get_text(strip=True)
                    if parent_name and parent_name != slug:
                        parents.append(parent_name)

                if parents:
                    return parents

    return []


def extract_from_text_patterns(soup: BeautifulSoup, slug: str) -> list[str]:
    """Extract hierarchy from page text patterns"""
    page_text = soup.get_text()
    name = slug.replace("-", " ").title()

    # French and English patterns
    patterns = [
        # "Lalande-de-Pomerol, appellation du Libournais dans Bordeaux"
        rf"{re.escape(name)},?\s+(?:appellation|région)\s+(?:du|de|des)\s+([^,\.]+)(?:\s+(?:dans|en)\s+(?:la\s+)?(?:région\s+)?(?:de|du|des)\s+([^,\.]+))?",
        # "située dans le Libournais, Bordeaux"
        r"(?:située?|localisée?)\s+dans\s+(?:le|la|les)?\s*([^,\.]+)(?:,\s*([^,\.]+))?",
        # "appellation de Bordeaux"
        r"appellation\s+(?:de|du|des)\s+([^,\.]+)",
        # English patterns
        rf"{re.escape(name)}\s+(?:is\s+)?(?:an?\s+)?(?:appellation|region|area)\s+(?:in|of|within)\s+([^,\.]+)",
        r"(?:located\s+)?(?:in|within)\s+(?:the\s+)?([^,\.]+)(?:\s+(?:region|area))?",
        # Simple comma patterns
        rf"{re.escape(name)},\s+([^,\.]+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                potential_parents = [m.strip() for m in match if m.strip()]
            else:
                potential_parents = [match.strip()]

            valid_parents = []
            for parent in potential_parents:
                if (
                    parent
                    and len(parent) < 30
                    and parent.lower() != name.lower()
                    and parent.lower() not in ["wine", "region", "appellation", "area"]
                ):
                    valid_parents.append(parent)

            if valid_parents:
                return valid_parents

    return []


def build_french_hierarchy(soup: BeautifulSoup, slug: str) -> list[str]:
    """Build French wine region hierarchy using knowledge base"""
    page_text = soup.get_text().lower()

    # French wine region hierarchy mapping
    region_hierarchies = {
        # Bordeaux regions
        "pomerol": ["France", "Bordeaux", "Libournais"],
        "lalande-de-pomerol": ["France", "Bordeaux", "Libournais"],
        "saint-emilion": ["France", "Bordeaux", "Libournais"],
        "fronsac": ["France", "Bordeaux", "Libournais"],
        "canon-fronsac": ["France", "Bordeaux", "Libournais"],
        "margaux": ["France", "Bordeaux", "Médoc", "Haut-Médoc"],
        "pauillac": ["France", "Bordeaux", "Médoc", "Haut-Médoc"],
        "saint-julien": ["France", "Bordeaux", "Médoc", "Haut-Médoc"],
        "saint-estephe": ["France", "Bordeaux", "Médoc", "Haut-Médoc"],
        "pessac-leognan": ["France", "Bordeaux", "Graves"],
        "sauternes": ["France", "Bordeaux", "Graves"],
        # Burgundy regions
        "chablis": ["France", "Burgundy", "Chablis"],
        "gevrey-chambertin": ["France", "Burgundy", "Côte de Nuits"],
        "nuits-saint-georges": ["France", "Burgundy", "Côte de Nuits"],
        "beaune": ["France", "Burgundy", "Côte de Beaune"],
        "meursault": ["France", "Burgundy", "Côte de Beaune"],
        # Rhône regions
        "chateauneuf-du-pape": ["France", "Rhône Valley", "Southern Rhône"],
        "cote-rotie": ["France", "Rhône Valley", "Northern Rhône"],
        "hermitage": ["France", "Rhône Valley", "Northern Rhône"],
        # Champagne
        "champagne": ["France", "Champagne"],
        # Loire Valley
        "sancerre": ["France", "Loire Valley"],
        "pouilly-fume": ["France", "Loire Valley"],
        # Alsace
        "alsace": ["France", "Alsace"],
        # Corsica
        "ajaccio": ["France", "Corsica"],
        "patrimonio": ["France", "Corsica"],
    }

    # Direct mapping
    if slug in region_hierarchies:
        return region_hierarchies[slug]

    # Dynamic detection for unmapped regions
    detected_hierarchy = []

    # Check for France
    if "france" in page_text or "français" in page_text:
        detected_hierarchy.append("France")

    # Check for major regions
    major_regions = {
        "bordeaux": ["bordeaux", "bordelais"],
        "burgundy": ["bourgogne", "burgundy"],
        "champagne": ["champagne"],
        "rhône valley": ["rhône", "rhone"],
        "loire valley": ["loire"],
        "alsace": ["alsace"],
        "languedoc": ["languedoc"],
        "provence": ["provence"],
        "corsica": ["corse", "corsica"],
    }

    for region, keywords in major_regions.items():
        if (
            any(keyword in page_text for keyword in keywords)
            and region.title() not in detected_hierarchy
        ):
            detected_hierarchy.append(region.title())

    # Check for sub-regions
    sub_regions = {
        "libournais": ["libournais", "libourne", "rive droite"],
        "médoc": ["médoc", "medoc"],
        "haut-médoc": ["haut-médoc", "haut-medoc"],
        "graves": ["graves"],
        "côte de nuits": ["côte de nuits", "cote de nuits"],
        "côte de beaune": ["côte de beaune", "cote de beaune"],
    }

    for sub_region, keywords in sub_regions.items():
        if (
            any(keyword in page_text for keyword in keywords)
            and sub_region.title() not in detected_hierarchy
        ):
            detected_hierarchy.append(sub_region.title())

    # Order hierarchy logically
    if detected_hierarchy:
        ordered_hierarchy = []

        # Standard order
        region_order = [
            "France",
            "Bordeaux",
            "Burgundy",
            "Champagne",
            "Rhône Valley",
            "Loire Valley",
            "Alsace",
            "Languedoc",
            "Provence",
            "Corsica",
            "Libournais",
            "Médoc",
            "Haut-Médoc",
            "Graves",
            "Côte De Nuits",
            "Côte De Beaune",
        ]

        for region in region_order:
            if region in detected_hierarchy:
                ordered_hierarchy.append(region)

        return ordered_hierarchy

    return []


def extract_region_children(soup: BeautifulSoup, slug: str, parents: list[str]) -> list[str]:
    """Extract child regions from page links"""
    children = []

    # Find main content area
    main_content = soup.find("main") or soup.find("body")
    if not main_content:
        return []

    # Get all wine region links
    region_links = main_content.find_all("a", href=re.compile(r"/wine-regions/"))

    # Exclude current slug and parent slugs
    excluded_slugs = {slug}
    excluded_slugs.update(p.lower().replace(" ", "-") for p in parents)

    for link in region_links:
        href = link.get("href", "")
        child_slug = href.split("/wine-regions/")[-1].split("?")[0].split("#")[0]

        if (
            child_slug
            and child_slug not in excluded_slugs
            and len(child_slug) > 1
            and "/" not in child_slug
        ):
            children.append(child_slug)
            excluded_slugs.add(child_slug)

    # Remove duplicates and limit
    return list(dict.fromkeys(children))[:20]


def build_hierarchy_tree(region_infos: list[dict]) -> dict:
    """Build hierarchical tree structure from region information"""
    tree = {}

    for info in region_infos:
        if not info:
            continue

        # Build path: parents + current region
        path = info.get("parents", []) + [info.get("name", info.get("slug", "unknown"))]

        # Navigate/create tree structure
        node = tree
        for segment in path:
            if segment:  # Skip empty segments
                node = node.setdefault(segment, {})

        # Add metadata
        node["_metadata"] = {
            "slug": info.get("slug"),
            "children": info.get("children", []),
            "hierarchy_depth": info.get("hierarchy_depth", 0),
            "full_hierarchy": info.get("full_hierarchy"),
            "source": info.get("source"),
        }

    return tree


def save_results(infos: list[dict], tree: dict, filename: str = None):
    """Save results to JSON file"""
    output_file = filename or OUT_JSON

    # Filter successful extractions
    successful_infos = [info for info in infos if info]

    result = {
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_regions_processed": len(infos),
        "successful_extractions": len(successful_infos),
        "success_rate": f"{len(successful_infos) / len(infos) * 100:.1f}%" if infos else "0%",
        "raw_data": successful_infos,
        "hierarchy_tree": tree,
        "statistics": {
            "regions_with_parents": len([i for i in successful_infos if i.get("parents")]),
            "regions_with_children": len([i for i in successful_infos if i.get("children")]),
            "max_hierarchy_depth": max(
                [i.get("hierarchy_depth", 0) for i in successful_infos], default=0
            ),
            "french_regions": len(
                [i for i in successful_infos if "France" in i.get("parents", [])]
            ),
        },
    }

    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)

    print(f"💾 Saved results to {output_file}")
    return output_file


def main():
    """Main execution function"""
    print("🍷 VIVINO FRENCH WINE REGION HIERARCHY SCRAPER")
    print("=" * 60)

    # Load region slugs
    print("🔄 Loading region slugs from wine dataset...")
    slugs = load_regions_from_wines(WINE_CSV)
    print(f"👉 Found {len(slugs)} unique region slugs")

    if not slugs:
        print("❌ No slugs found. Check your wine data file path.")
        return

    # Process regions
    infos = []
    successful = 0

    print("\n🚀 Starting extraction process...")

    try:
        for i, slug in enumerate(sorted(slugs), 1):
            try:
                print(f"\n[{i:3d}/{len(slugs)}] Processing: {slug}")

                info = fetch_region_info(slug)
                infos.append(info)

                if info:
                    successful += 1

                # Save progress every 25 regions
                if i % 25 == 0:
                    tree = build_hierarchy_tree(infos)
                    save_results(infos, tree, f"vivino_regions_progress_{i}.json")
                    print(f"    💾 Progress saved: {successful}/{i} successful")

                # Polite delay
                time.sleep(DELAY)

            except KeyboardInterrupt:
                print(f"\n⚠️ Interrupted by user at region {i}")
                break
            except Exception as e:
                print(f"    ❌ Unexpected error: {e}")
                infos.append(None)

        # Build final tree and save
        print("\n🔨 Building final hierarchy tree...")
        tree = build_hierarchy_tree(infos)
        output_file = save_results(infos, tree)

        # Print final statistics
        print("\n🎉 EXTRACTION COMPLETE!")
        print(f"    • Total regions: {len(slugs)}")
        print(f"    • Successful: {successful}")
        print(f"    • Success rate: {successful / len(slugs) * 100:.1f}%")
        print(f"    • Output file: {output_file}")

        # Show some hierarchy examples
        successful_infos = [info for info in infos if info and info.get("parents")]
        if successful_infos:
            print("\n🏗️ HIERARCHY EXAMPLES:")
            for info in successful_infos[:5]:
                print(f"    • {info['full_hierarchy']}")

    except Exception as e:
        print(f"❌ Fatal error: {e}")


# Test function for development
def test_regions():
    """Test specific regions for development"""
    print("🧪 TESTING SPECIFIC REGIONS")
    print("=" * 40)

    test_slugs = ["lalande-de-pomerol", "margaux", "ajaccio", "champagne"]

    for slug in test_slugs:
        print(f"\n{'=' * 15} {slug.upper()} {'=' * 15}")

        try:
            result = fetch_region_info(slug)

            if result:
                print("✅ SUCCESS:")
                print(f"  Name: {result['name']}")
                print(f"  Hierarchy: {result['full_hierarchy']}")
                print(f"  Children: {len(result['children'])} found")
                print(f"  Depth: {result['hierarchy_depth']}")
            else:
                print("❌ FAILED: No data extracted")

            time.sleep(2)

        except Exception as e:
            print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    # Uncomment one of these:

    # For testing specific regions:
    # test_regions()

    # For full scraping:
    main()
