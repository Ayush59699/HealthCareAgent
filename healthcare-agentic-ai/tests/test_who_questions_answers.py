"""Offline Q&A fixtures, including WHO's dynamic catalogue and collapsed answers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from scripts import crawl_who_questions_answers as who

URL = who.INDEX_URL + "/item/anaemia"
ARTICLE = """<html><title>Fallback</title><nav>Navigation junk</nav>
<div class="qa-details__wrapper"><div class="qa-details__header"><h1>Anaemia</h1><span>Date | Questions and answers</span></div>
<div class="qa-details__content"><p>Medical introduction.</p><div class="sf-accordion">
<div class="sf-accordion__panel" itemtype="https://schema.org/Question">
<div class="sf-accordion__trigger-panel"><a href="#"><span>What is anaemia?</span></a></div>
<div class="sf-accordion__content" hidden aria-hidden="true" style="display:none"><p>Original <strong>answer</strong>.</p>
<ul><li>First fact.</li><li>Second fact.</li></ul></div></div>
<div class="sf-accordion__panel"><div class="sf-accordion__trigger-panel">How is it treated?</div>
<div class="sf-accordion__content"><h3>Options</h3><p>Full answer.</p><ol><li>First step.</li><li>Second step.</li></ol></div></div>
</div><p>Final reference.</p><div class="qa-details__related">Related junk</div>
<div class="social-share">Share junk</div><script>Script junk</script><footer>Footer junk</footer>
</div></div></html>"""
DYNAMIC = """<script type="text/x-kendo-tmpl">
<a href="https://www.who.int/news-room/questions-and-answers/item#:ItemDefaultUrl #">#:Title#</a>
</script><script>hubsfiltering("20", "widget-id", "/api/hubs/qandagroups?sf_culture=en&$select=Title,ItemDefaultUrl");</script>"""


def catalogue(*topics):
    return "".join(f'<a href="/news-room/questions-and-answers/item/{topic}">{topic}</a>' for topic in topics)


def page(*topics, total=None):
    value = {"value": [{"ItemDefaultUrl": "/" + topic} for topic in topics]}
    if total is not None:
        value["@odata.count"] = total
    return json.dumps(value).encode()


class DiscoveryTests(unittest.TestCase):
    def test_filter(self):
        for value in ("https://evil.test/news-room/questions-and-answers/item/a",
                      "https://www.who.int.evil.test/news-room/questions-and-answers/item/a",
                      "https://me@www.who.int/news-room/questions-and-answers/item/a",
                      "/news-room/fact-sheets/detail/a", who.INDEX_URL,
                      "/news-room/questions-and-answers/item/a%2fb",
                      "/news-room/questions-and-answers/item/a%5cb",
                      "/news-room/questions-and-answers/item/../escape", "javascript:alert(1)"):
            with self.subTest(value=value):
                self.assertIsNone(who.normalize_url(value))

    def test_normalize_and_safe_filename(self):
        for value in (URL, URL + "/?source=index#top", "http://WHO.INT/news-room/questions-and-answers/item/%61naemia"):
            self.assertEqual(who.normalize_url(value), URL)
        self.assertEqual(who.filename_for_url(URL), "anaemia.txt")
        self.assertEqual(who.filename_for_url(who.INDEX_URL + "/item/CON"), "_CON.txt")
        unicode_url = who.INDEX_URL + "/item/guillain-barré-syndrome"
        self.assertEqual(who.normalize_url(unicode_url), who.normalize_url(unicode_url.replace("é", "%C3%A9")))
        with self.assertRaises(ValueError):
            who.filename_for_url("https://evil.test/escape")

    def test_static_discovery(self):
        urls, duplicates = who.discover_urls(catalogue("z", "anaemia", "anaemia/?x=1#top"))
        self.assertEqual(urls, [URL, who.INDEX_URL + "/item/z"])
        self.assertEqual(duplicates, 1)

    def test_api_config_from_page_not_guessed(self):
        self.assertTrue(who.api_configuration(DYNAMIC).startswith("https://www.who.int/api/hubs/qandagroups?"))
        self.assertIsNone(who.api_configuration("<script>other()</script>"))
        with self.assertRaises(ValueError):
            who.api_configuration(DYNAMIC.replace("ItemDefaultUrl #", "UnknownField #"))

    def test_paginated_discovery_and_duplicates(self):
        fetcher = MagicMock()
        fetcher.get.side_effect = [DYNAMIC + catalogue("a"), page("z", "a", total=3), page("b", total=3)]
        urls, duplicates = who.discover_collection(fetcher)
        self.assertEqual(urls, [who.INDEX_URL + "/item/" + topic for topic in ("a", "b", "z")])
        self.assertEqual(duplicates, 1)
        calls = [c.args[0] for c in fetcher.get.call_args_list]
        self.assertIn("%24skip=0", calls[1])
        self.assertIn("%24skip=2", calls[2])
        self.assertTrue(all("/api/hubs/qandagroups?" in u for u in calls[1:]))

    def test_no_count_paginates_until_empty(self):
        fetcher = MagicMock()
        fetcher.get.side_effect = [DYNAMIC, page("a"), page("b"), page()]
        self.assertEqual(len(who.discover_collection(fetcher)[0]), 2)
        self.assertEqual(fetcher.get.call_count, 4)

    def test_broken_or_repeated_api_page_fails(self):
        for pages in ([page("a", total=2), page(total=2)],
                      [page("a"), page("a")], [b'{"value": "bad"}'],
                      [page("a", total=2), page("b", total=3)],
                      [b'{"value": [{}]}'], [b'invalid JSON']):
            fetcher = MagicMock()
            fetcher.get.side_effect = [DYNAMIC] + pages
            with self.subTest(pages=pages), self.assertRaises(ValueError):
                who.discover_collection(fetcher)

    def test_real_dotted_topic_from_api(self):
        topic = "what-s-super-about-super-gonorrhea-a-q-a-with-who-s-dr.-teodora-wi"
        fetcher = MagicMock()
        fetcher.get.side_effect = [DYNAMIC, page(topic, total=1)]
        urls, duplicates = who.discover_collection(fetcher)
        self.assertEqual(who.filename_for_url(urls[0]), topic + ".txt")
        self.assertEqual(duplicates, 0)
        self.assertEqual(who.filename_for_url(who.INDEX_URL + "/item/CON.any"), "_CON.any.txt")

    def test_static_catalogue_needs_no_api(self):
        fetcher = MagicMock()
        fetcher.get.return_value = catalogue("anaemia")
        self.assertEqual(who.discover_collection(fetcher), ([URL], 0))
        fetcher.get.assert_called_once_with(who.INDEX_URL)


class ExtractionTests(unittest.TestCase):
    def test_collapsed_questions_answers_and_intro(self):
        title, text = who.extract_article(ARTICLE)
        self.assertEqual(title, "Anaemia")
        for part in ("What is anaemia?\n===============", "Original answer.",
                     "- First fact.", "- Second fact.", "How is it treated?",
                     "Options\n-------", "Full answer.", "1. First step.",
                     "2. Second step.", "Medical introduction.", "Final reference."):
            self.assertIn(part, text)
        self.assertLess(text.index("Original answer."), text.index("How is it treated?"))
        for junk in ("Navigation junk", "Related junk", "Share junk", "Script junk", "Footer junk", "Date |"):
            self.assertNotIn(junk, text)

    def test_semantic_faq_fallback(self):
        html = '<h1>Topic</h1><div itemtype="https://schema.org/FAQPage"><div itemtype="https://schema.org/Question"><span itemprop="name">Question?</span><div itemprop="acceptedAnswer" hidden>Answer.</div></div></div>'
        title, text = who.extract_article(html)
        self.assertEqual(title, "Topic")
        self.assertIn("Question?\n=========", text)
        self.assertIn("Answer.", text)

    def test_main_and_native_disclosure_fallback(self):
        title, text = who.extract_article('<title>Topic</title><main><details><summary>Question?</summary><p>Full answer.</p></details><h3>References</h3><p>Source.</p></main>')
        self.assertEqual(title, "Topic")
        self.assertIn("Question?\n=========", text)
        self.assertIn("Source.", text)

    def test_no_truncation(self):
        long_answer = "Exact wording. " * 10000
        _, text = who.extract_article(ARTICLE.replace("Full answer.", long_answer))
        self.assertEqual(text.count("Exact wording."), 10000)
        self.assertIn("Final reference.", text)

    def test_fail_on_missing_content_or_incomplete_panel(self):
        for html in ("<h1>Title</h1><nav>Junk</nav>", "<main><h1>Title</h1></main>",
                     ARTICLE.replace('class="sf-accordion__content"', 'class="missing-answer"'),
                     ARTICLE.replace('class="sf-accordion__trigger-panel"', 'class="missing-question"')):
            with self.subTest(html=html[:60]), self.assertRaises(ValueError):
                who.extract_article(html)


class FetchTests(unittest.TestCase):
    @patch.object(who.time, "sleep")
    def test_delay_timeout_and_redirect_scope(self, sleep):
        session = MagicMock()
        response = session.get.return_value.__enter__.return_value
        response.status_code = 200
        response.content = b"content"
        fetcher = who.PoliteFetcher(session, 1, 30)
        fetcher.get(who.INDEX_URL)
        fetcher.get("https://www.who.int/api/hubs/qandagroups?$skip=0")
        fetcher.get(URL)
        self.assertEqual(sleep.call_count, 2)
        session.get.assert_called_with(URL, timeout=30, allow_redirects=False)
        response.status_code = 302
        for target in ("https://evil.test", "https://www.who.int/about", "https://www.who.int/news-room/fact-sheets/detail/anaemia", "https://www.who.int/api/hubs/qandagroups"):
            response.headers = {"Location": target}
            with self.assertRaises(ValueError):
                fetcher.get(URL)

    def test_http_error(self):
        session = MagicMock()
        response = session.get.return_value.__enter__.return_value
        response.status_code = 503
        response.raise_for_status.side_effect = requests.HTTPError("503")
        with self.assertRaises(requests.HTTPError):
            who.PoliteFetcher(session, 0, 30).get(URL)

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
        self.assertTrue(text.startswith("WHO QUESTIONS AND ANSWERS\nTitle: Anaemia\nURL: " + URL + "\nRetrieved: "))
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



if __name__ == "__main__":
    unittest.main()
