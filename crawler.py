from playwright.async_api import async_playwright
import asyncio
import os
import re

BASE_URL = "https://www.glitexsolutions.co.ke/" 
OUTPUT_DIR = "./scraped_content"
MAX_PAGES = 7
WAIT_TIME = 2000  


def url_to_filename(url):
    """
    convert URL to a valid filename.
    """
    filename = url.replace(BASE_URL, "").lstrip("/")

    # replace non-alphanumeric characters with underscore
    filename = re.sub(r"[^a-zA-Z0-9]", "_", filename)

    filename = re.sub(r"_+", "_", filename)
    filename = filename.strip("_")

    return filename or "home"


async def scrape_website():
    """
    main function to scrape the website.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # track visited URLs 
        visited_urls = set()
        urls_to_visit = [BASE_URL]

        print(f"Starting to scrape {BASE_URL}")

        while urls_to_visit and len(visited_urls) < MAX_PAGES:
            current_url = urls_to_visit.pop(0)

            # skip if already visited
            if current_url in visited_urls:
                continue

            try:
                print(f"Visiting: {current_url}")

                await page.goto(current_url, wait_until="networkidle")

                await page.wait_for_timeout(WAIT_TIME)

                # extract  content
                page_content = await page.evaluate("""() => {
                    // Remove script and style tags
                    const scripts = document.querySelectorAll('script, style');
                    scripts.forEach(script => script.remove());
                    
                    // Get text content from the body
                    return document.body.innerText;
                }""")

                # save content to file
                filename = f"{url_to_filename(current_url)}.txt"
                file_path = os.path.join(OUTPUT_DIR, filename)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(page_content.strip())
                
                print(f"Saved content to {filename}")

                # mark as visited
                visited_urls.add(current_url)

                # find all links on the page that belong to the same domain
                links = await page.evaluate(
                    """(baseUrl) => {
                        const anchors = Array.from(document.querySelectorAll('a[href]'));
                        return anchors
                            .map(a => a.href)
                            .filter(href => 
                                href.startsWith(baseUrl) && 
                                !href.includes('#') && 
                                !href.endsWith('.pdf') && 
                                !href.endsWith('.jpg') && 
                                !href.endsWith('.png')
                            );
                    }""",
                    BASE_URL,
                )

                # add new links to the queue
                for link in links:
                    if link not in visited_urls and link not in urls_to_visit:
                        urls_to_visit.append(link)

                print(f"Found {len(links)} links on the page")
                print(f"Queue size: {len(urls_to_visit)}, Visited: {len(visited_urls)}")

            except Exception as e:
                print(f"Error scraping {current_url}: {str(e)}")

        summary_path = os.path.join(OUTPUT_DIR, "scraped_urls.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(visited_urls))

        print(f"Scraping completed. Visited {len(visited_urls)} pages.")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(scrape_website())