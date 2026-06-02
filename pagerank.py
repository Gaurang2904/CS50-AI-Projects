import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a set of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue

        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(
                r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"",
                contents
            )
            pages[filename] = set(links) - {filename}

    # Only include links to pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    """

    n = len(corpus)

    # Base probability from random jump
    probabilities = {
        p: (1 - damping_factor) / n
        for p in corpus
    }

    links = corpus[page]

    # If page has no outgoing links
    if len(links) == 0:
        return {
            p: 1 / n
            for p in corpus
        }

    # Distribute damping probability among linked pages
    for link in links:
        probabilities[link] += damping_factor / len(links)

    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values by sampling n pages according to
    transition model.
    """

    pages = list(corpus.keys())

    counts = {
        page: 0
        for page in pages
    }

    # First sample chosen randomly
    current_page = random.choice(pages)
    counts[current_page] += 1

    for _ in range(n - 1):

        model = transition_model(
            corpus,
            current_page,
            damping_factor
        )

        current_page = random.choices(
            population=list(model.keys()),
            weights=list(model.values()),
            k=1
        )[0]

        counts[current_page] += 1

    ranks = {
        page: counts[page] / n
        for page in pages
    }

    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values by iteratively updating
    PageRank values until convergence.
    """

    n = len(corpus)

    # Initial rank = 1/N
    ranks = {
        page: 1 / n
        for page in corpus
    }

    while True:

        new_ranks = {}

        for page in corpus:

            total = 0

            for possible_page in corpus:

                links = corpus[possible_page]

                # Page with no links counts as linking to all pages
                if len(links) == 0:
                    links = set(corpus.keys())

                if page in links:
                    total += ranks[possible_page] / len(links)

            new_ranks[page] = (
                (1 - damping_factor) / n
                + damping_factor * total
            )

        # Check convergence
        converged = True

        for page in corpus:
            if abs(new_ranks[page] - ranks[page]) > 0.001:
                converged = False
                break

        ranks = new_ranks

        if converged:
            break

    return ranks


if __name__ == "__main__":
    main()