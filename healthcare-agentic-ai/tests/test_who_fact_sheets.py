"""Offline fixtures; no WHO requests or healthcare runtime imports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from scripts import crawl_who_fact_sheets as who

URL = who.INDEX_URL + "/detail/anaemia"
ARTICLE = """<html><head><title>A fallback title</title></head><body>
<nav>Navigation junk</nav><div class="sf-item-header-wrapper"><h1>Anaemia</h1></div>
<article class="sf-detail-body-wrapper"><!-- hidden editorial comment -->
<h2>Key facts</h2><ul><li>Anaemia is a <strong>condition</strong>.</li>
<li>Original medical wording.</li></ul><h2>Overview</h2><p>First paragraph.</p>
<h3>Treatment</h3><p>Second paragraph with <a href="/reference">a reference</a>.</p>
<table><tr><th>Group</th><th>Value</th></tr><tr><td>Adults</td><td>10</td></tr></table>
<div class="social-share">Share junk</div><script>bad script</script>
<footer>Footer junk</footer></article><aside>Related junk</aside></body></html>"""


def catalogue(*topics):
    return "".join(f'<a href="/news-room/fact-sheets/detail/{topic}">{topic}</a>' for topic in topics)


class URLTests(unittest.TestCase):
    def test_filter(self):
        for value in ("https://evil.test/news-room/fact-sheets/detail/anaemia",
                      "https://www.who.int.evil.test/news-room/fact-sheets/detail/anaemia",
                      "https://me@www.who.int/news-room/fact-sheets/detail/anaemia",
                      "/news-room/fact-sheets", "/news-room/fact-sheets/detail/",
                      "/news-room/fact-sheets/detail/../escape",
                      "/news-room/fact-sheets/detail/a%2fb",
                      "/news-room/fact-sheets/detail/a%5cb",
                      "/news-room/fact-sheets/detail/a/b", "javascript:alert(1)",
                      "https://www.who.int:123/news-room/fact-sheets/detail/a"):
            with self.subTest(value=value):
                self.assertIsNone(who.normalize_url(value))

    def test_normalize(self):
        for value in (URL, URL + "/?source=index#facts",
                      "http://WHO.INT/news-room/fact-sheets/detail/%61naemia",
                      "/news-room/fact-sheets/detail/anaemia", "detail/anaemia"):
            base = who.INDEX_URL + "/" if value == "detail/anaemia" else who.INDEX_URL
            self.assertEqual(who.normalize_url(value, base), URL)

    def test_unicode_and_path_case(self):
        url = who.INDEX_URL + "/detail/guillain-barré-syndrome"
        self.assertEqual(who.normalize_url(url), who.normalize_url(url.replace("é", "%C3%A9")))
        self.assertEqual(who.filename_for_url(url), "guillain-barré-syndrome.txt")
        self.assertEqual(who.topic_from_url(who.INDEX_URL + "/detail/community-CBHI"), "community-CBHI")

    def test_filenames(self):
        self.assertEqual(who.filename_for_url(URL), "anaemia.txt")
        self.assertEqual(who.filename_for_url(who.INDEX_URL + "/detail/ambient-(outdoor)"), "ambient-(outdoor).txt")
        self.assertEqual(who.filename_for_url(who.INDEX_URL + "/detail/CON"), "_CON.txt")
        with self.assertRaises(ValueError):
            who.filename_for_url("https://evil.test/../../escape")

    def test_discovery_dedup_and_sort(self):
        html = catalogue("zika-virus", "anaemia", "anaemia/?x=1#top") + '<a href="/about">About</a>'
        urls, duplicates = who.discover_urls(html)
        self.assertEqual(urls, [URL, who.INDEX_URL + "/detail/zika-virus"])
        self.assertEqual(duplicates, 1)


class ExtractionTests(unittest.TestCase):
    def test_structure_and_wording(self):
        title, text = who.extract_article(ARTICLE)
        self.assertEqual(title, "Anaemia")
        for part in ("Key facts\n=========", "Overview\n========", "Treatment\n---------",
                     "- Anaemia is a condition.", "Original medical wording.",
                     "Second paragraph with a reference.", "Adults | 10"):
            self.assertIn(part, text)
        self.assertLess(text.index("Key facts"), text.index("Overview"))
        for junk in ("Navigation", "Share junk", "Footer junk", "Related junk", "bad script", "editorial comment"):
            self.assertNotIn(junk, text)

    def test_fallbacks(self):
        for wrapper in ("article", "main", 'div id="content"', 'div itemprop="articleBody"'):
            html = f'<{wrapper}><h1>Asthma</h1><h2>Symptoms</h2><p>Original text.</p></{wrapper.split()[0]}>'
            with self.subTest(wrapper=wrapper):
                title, body = who.extract_article(html)
                self.assertEqual(title, "Asthma")
                self.assertIn("Symptoms\n========", body)
                self.assertIn("Original text.", body)
                self.assertNotIn("Asthma", body)

    def test_title_fallback_and_multiple_sections(self):
        title, body = who.extract_article('<title>Topic</title><main><p>First.</p><section><h2>Last</h2><p>Final.</p></section></main>')
        self.assertEqual(title, "Topic")
        self.assertIn("First.", body)
        self.assertIn("Final.", body)

    def test_who_legacy_key_facts_outside_article(self):
        html = """<h1>Topic</h1><div class='row sf-detail-content'>
        <div data-placeholder-label='Body'><div class='list-bold separator-line'>
        <h2>Key facts</h2><ul><li>Important fact.</li></ul></div>
        <article class='sf-detail-body-wrapper'><h2>Overview</h2><p>Body.</p></article>
        </div><div>Unrelated sidebar</div></div>"""
        _, body = who.extract_article(html)
        self.assertIn("Key facts\n=========", body)
        self.assertIn("Important fact.", body)
        self.assertLess(body.index("Key facts"), body.index("Overview"))
        self.assertNotIn("Unrelated sidebar", body)

    def test_article_header_title_and_numbered_list(self):
        html = """<article><header><h1>Topic</h1></header><h2>References</h2>
        <ol start='3'><li>First reference.</li><li value='7'>Second reference.</li>
        <li>Third reference.</li></ol></article>"""
        title, body = who.extract_article(html)
        self.assertEqual(title, "Topic")
        for part in ("3. First reference.", "7. Second reference.", "8. Third reference."):
            self.assertIn(part, body)

    def test_no_whole_page_fallback(self):
        for html in ("<h1>Error</h1><nav>Only navigation</nav>", "<main><h1>Empty</h1></main>"):
            with self.assertRaises(ValueError):
                who.extract_article(html)

    def test_long_content_not_truncated(self):
        text = "Original wording. " * 10000
        _, body = who.extract_article(f'<article><h1>Topic</h1><p>{text}</p><h2>End</h2><p>Last sentence.</p></article>')
        self.assertEqual(body.count("Original wording."), 10000)
        self.assertTrue(body.endswith("Last sentence."))


class CrawlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def run_crawl(self, side_effect, **kwargs):
        with patch.object(who.PoliteFetcher, "get", side_effect=side_effect) as fetch:
            report = who.crawl(self.output, delay=0, **kwargs)
        return report, fetch

    def test_index_and_rerun(self):
        report, _ = self.run_crawl([catalogue("anaemia"), ARTICLE])
        self.assertEqual(report["successful_downloads"], 1)
        index = json.loads((self.output / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(index, [{"topic": "anaemia", "title": "Anaemia", "url": URL, "file": "anaemia.txt"}])
        self.assertEqual(json.loads((self.output / "urls.json").read_text()), [URL])
        text = (self.output / index[0]["file"]).read_text(encoding="utf-8")
        self.assertTrue(text.startswith("WHO FACT SHEET\nTitle: Anaemia\nURL: " + URL + "\nRetrieved: "))
        report, fetch = self.run_crawl([catalogue("anaemia")])
        self.assertEqual(report["skipped_existing"], 1)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(who.generate_index(self.output), index)
        report, fetch = self.run_crawl([catalogue("anaemia"), ARTICLE], force=True)
        self.assertEqual(report["successful_downloads"], 1)
        self.assertEqual(fetch.call_count, 2)

    def test_failure_continues(self):
        report, fetch = self.run_crawl([catalogue("a", "b", "c", "d"), requests.HTTPError("503"),
                                        requests.Timeout("timed out"), "<h1>No body</h1>", ARTICLE])
        self.assertEqual(report["failed_downloads"], 3)
        self.assertEqual(report["successful_downloads"], 1)
        self.assertEqual(len(report["failed_urls"]), 3)
        self.assertEqual(fetch.call_count, 5)
        self.assertEqual([x["file"] for x in who.generate_index(self.output)], ["d.txt"])

    def test_limit_preserves_earlier_index_entries(self):
        self.run_crawl([catalogue("z"), ARTICLE])
        report, fetch = self.run_crawl([catalogue("a", "b", "z", "a"), ARTICLE], limit=1)
        self.assertEqual(report["total_urls_discovered"], 3)
        self.assertEqual(report["duplicate_urls"], 1)
        self.assertEqual(report["skipped_by_limit"], 2)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(len(json.loads((self.output / "urls.json").read_text())), 3)
        self.assertEqual([e["topic"] for e in who.generate_index(self.output)], ["a", "z"])

    def test_failed_force_retains_previous_file(self):
        self.run_crawl([catalogue("anaemia"), ARTICLE])
        original = (self.output / "anaemia.txt").read_bytes()
        report, _ = self.run_crawl([catalogue("anaemia"), requests.HTTPError("503")], force=True)
        self.assertEqual(report["failed_downloads"], 1)
        self.assertEqual((self.output / "anaemia.txt").read_bytes(), original)
        self.assertEqual(len(who.generate_index(self.output)), 1)

    def test_discovery_failure_keeps_index(self):
        self.run_crawl([catalogue("anaemia"), ARTICLE])
        original = (self.output / "index.json").read_bytes()
        for value in (requests.ConnectionError("offline"), "<nav>No detail links</nav>"):
            report, _ = self.run_crawl([value])
            self.assertIsNotNone(report["discovery_error"])
            self.assertEqual((self.output / "index.json").read_bytes(), original)
            self.assertTrue((self.output / "crawl_report.json").is_file())

    def test_invalid_options(self):
        for options in ({"limit": 0}, {"delay": -1}, {"timeout": 0}):
            with self.assertRaises(ValueError):
                who.crawl(self.output, **options)


class FetchTests(unittest.TestCase):
    def response(self, status=200, location=None):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = status
        response.headers = {"Location": location} if location else {}
        response.content = b"<article>test</article>"
        return response

    @patch.object(who.time, "sleep")
    def test_delay_timeout_and_manual_redirects(self, sleep):
        session = MagicMock()
        response = self.response()
        session.get.return_value = response
        fetcher = who.PoliteFetcher(session, 1, 30)
        fetcher.get(who.INDEX_URL)
        fetcher.get(URL)
        sleep.assert_called_once_with(1)
        session.get.assert_called_with(URL, timeout=30, allow_redirects=False)
        self.assertEqual(response.raise_for_status.call_count, 2)

    def test_refuse_unrelated_redirect(self):
        for target in ("https://evil.test", "https://www.who.int/about"):
            session = MagicMock()
            session.get.return_value = self.response(302, target)
            with self.assertRaises(ValueError):
                who.PoliteFetcher(session, 0, 30).get(URL)
            self.assertEqual(session.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
