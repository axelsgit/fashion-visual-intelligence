import os
import time
import requests
from urllib.parse import urljoin, urlsplit
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


"""
==============================================================
Ethical Use & Disclaimer
==============================================================

This script is provided strictly for educational and research purposes.

It demonstrates how to automate the retrieval of publicly visible runway images
from FirstView.com using open-source web scraping tools (Selenium, BeautifulSoup).

IMPORTANT:
- Do not redistribute, republish, or publicly host any downloaded images.
- Do not use this script for large-scale scraping or commercial purposes.
- All images remain the exclusive property of their respective copyright holders,
  including FirstView and the fashion houses represented.
- This script is intended for **academic visualisation, AI-based analysis, and
  exploration of computer vision methods in fashion**.

By using this script, you agree to comply with all applicable website Terms of Service
and copyright laws in your jurisdiction.

Author: Axel Heussner
License: MIT (Code only - not applicable to scraped content)
==============================================================
"""


def scrape_firstview_collection(collection_id, output_dir):
    """
    Scrape runway images from a dynamically loaded FirstView collection page.
    Example: https://www.firstview.com/collection_images.php?id=57219
    """
    url = f"https://www.firstview.com/collection_images.php?id={collection_id}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Scraping {url} ...")

    # --- Setup Chrome ---
    chrome_options = Options()
    #chrome_options.add_argument("--headless")  # run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    # --- Load the page ---
    driver.get(url)
    # give initial time for JS to run
    time.sleep(2)

    # --- Scroll to the bottom to trigger lazy-loading of images ---
    SCROLL_PAUSE = 1.5
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                # give a final short wait for lazy images to load
                time.sleep(1)
                break
            last_height = new_height

        # Wait until the number of <img> tags is stable (no new images for a short period)
        def wait_images_stable(driver, timeout=15, check_interval=0.5, stable_time=1.0):
            start = time.time()
            prev = -1
            stable_since = None
            while time.time() - start < timeout:
                imgs = driver.find_elements(By.TAG_NAME, "img")
                curr = len(imgs)
                if curr == prev:
                    if stable_since is None:
                        stable_since = time.time()
                    elif time.time() - stable_since >= stable_time:
                        return
                else:
                    stable_since = None
                    prev = curr
                time.sleep(check_interval)

        wait_images_stable(driver)
    except Exception as e:
        print(f"Scrolling/waiting error: {e}")

    # --- Parse rendered HTML ---
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # --- Find all images ---
    images = soup.find_all("img")
    print(f"Found {len(images)} <img> tags on the rendered page")
    # Print a small sample of img attributes for debugging
    for i, im in enumerate(images, start=1):
        attrs = {k: v for k, v in im.attrs.items() if k in ("src", "data-src", "data-original", "data-lazy", "srcset")}
        print(f" sample img #{i}: {attrs}")

    count = 0

    for img in images:
        # try common attributes used for lazy-loading images
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-original")
            or img.get("data-lazy")
        )

        # handle srcset -> take the first url
        if not src:
            srcset = img.get("srcset")
            if srcset:
                # srcset may contain entries like: 'url1 300w, url2 600w'
                src = srcset.split(",")[0].strip().split()[0]

        if not src:
            # nothing usable
            continue

        # build absolute URL relative to the page url
        src = urljoin(url, src)

        # choose a filename from the path component
        parsed = urlsplit(src)
        base = os.path.basename(parsed.path)
        if not base:
            base = f"image_{count+1}.jpg"

        filename = os.path.join(output_dir, base)
        # avoid overwriting files with same name
        if os.path.exists(filename):
            name, ext = os.path.splitext(filename)
            i = 1
            while os.path.exists(filename):
                filename = f"{name}_{i}{ext}"
                i += 1

        try:
            # use a browser-like user agent so some servers don't block us
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36"}
            resp = requests.get(src, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"Skipping {src} (status {resp.status_code})")
                continue

            ctype = resp.headers.get("Content-Type", "")
            if "image" not in ctype:
                print(f"Skipping {src} (not an image: {ctype})")
                continue

            # skip certain undesired sources or formats
            if any(term in src.lower() for term in ["gif", "instagram", "facebook", "tiktok", "xiao"]):
                print(f"Skipping {src} (undesired content)")
                continue

            with open(filename, "wb") as f:
                f.write(resp.content)

            count += 1
            time.sleep(1)
        except Exception as e:
            print(f"Failed to download {src}: {e}")

    print(f"Downloaded {count} images to {output_dir}")


if __name__ == "__main__":

    dior_collection_id_dict = {
        # Maria Grazia Chiuri (2016–2024)
        53106: 'FW2023_Chiuri', 
        52527: 'FW2022_Chiuri',
        54048: 'SS2021_Chiuri', 
        50493: 'FW2019_Chiuri',
        48945: 'FW2018_Chiuri', 
        45903: 'SS2017_Chiuri',
        # Raf Simons (2012–2015)
        41690: 'FW2015_Simons',
        38823: 'FW2014_Simons',
        32885: 'SS2013_Simons',
        54675: 'FW2012_Simons',
        # John Galliano (1998–2011)
        54663: 'FW2010_Galliano',
        54686: 'SS2007_Galliano',
        54672: 'FW2005_Galliano',
        54690: 'SS2002_Galliano',
        54658: 'FW1998_Galliano',

    }

    for key, value in dior_collection_id_dict.items():
        output_dir = f"data/{value}"
        scrape_firstview_collection(key, output_dir)
