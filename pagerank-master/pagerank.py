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
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
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

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    links = corpus[page]
    model = dict()
    total_pages = len(corpus)
    initial_probab = 1/total_pages
    rand_probab = (1 - damping_factor) * initial_probab

    if len(links) == 0:
        for p in corpus:
            model[p] = initial_probab

        return model

    for p in corpus:
        if p in links:
            model[p] = rand_probab + damping_factor * (1/len(links))
        else:
            model[p] = rand_probab

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page = list(corpus.keys())[random.randint(0, len(corpus) - 1)]
    pagerank = {}

    for p in corpus:
        pagerank[p] = 0

    for _ in range(0, n):
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(list(model.keys()), list(model.values()))[0]
        pagerank[page] += 1/n

    for page in pagerank:
        pagerank[page] = round(pagerank[page], 3)

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {}
    total_pages = len(corpus)
    initial_probab = 1 / total_pages
    rand_probab = (1-damping_factor) * initial_probab

    for p in corpus:
        pagerank[p] = initial_probab
        if len(corpus[p]) == 0:
            corpus[p] = set(corpus.keys())

    while True:
        max_change = 0

        for p in pagerank:
            prev_rank = pagerank[p]
            pagerank[p] = rand_probab

            for x in corpus:
                if p in corpus[x]:
                    pagerank[p] = round(pagerank[p] + damping_factor *
                                        pagerank[x] / len(corpus[x]), 3)

            change = abs(pagerank[p] - prev_rank)
            max_change = change if change > max_change else max_change

        if max_change < 0.001:
            break

    return pagerank


if __name__ == "__main__":
    main()
