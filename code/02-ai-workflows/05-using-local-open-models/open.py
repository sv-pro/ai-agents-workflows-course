import enum
import json
import time
import os

# Run "uv sync" to install the below packages
from dotenv import load_dotenv
import requests

load_dotenv()

# there are 3 older models ("old" generation)
# "gemma3:1b-it-qat" is used for summarization
# "gemma3:4b-it-qat" is used for extraction
# "gemma3:27b-it-qat" is used for post-generation

# there are 2 newer models ("new" generation)
# "gemma3n:e2b" is used for summarization
# "gemma3n:e4b" is used for extraction and post-generation

# the `generation` switch determines which model to use for each task
# GENERATION is passed as an env variable, the default value is "new"
VARIATION = os.getenv("VARIATION", "new")

print(f"Using variation: {VARIATION}")

LIGHT_MODEL = "gemma3n:e2b"
HEAVY_MODEL = "gemma3n:e4b"
OLLAMA_GENERATE_ENDPOINT = "http://localhost:11434/api/generate"
SOURCE_URL = "https://maximilian-schwarzmueller.com/articles/gemma-3n-may-be-amazing/"

FASTEST_MODEL = "gemma3:1b-it-qat"

MODELS = {
    "old": {
        "summarization": "gemma3:1b-it-qat",
        "extraction": "gemma3:4b-it-qat",
        "post_generation": "gemma3:27b-it-qat"
    },
    "new": {
        "summarization": "gemma3n:e2b",
        "extraction": "gemma3n:e4b",
        "post_generation": "gemma3n:e4b"
    },
    "fastest": {
        "summarization": FASTEST_MODEL,
        "extraction": FASTEST_MODEL,
        "post_generation": FASTEST_MODEL
    }
}

SUMMARIZATION_MODEL = MODELS[VARIATION]["summarization"]
EXTRACTION_MODEL = MODELS[VARIATION]["extraction"]
POST_GENERATION_MODEL = MODELS[VARIATION]["post_generation"]


def get_ai_response(prompt: str, model: str = LIGHT_MODEL, ctx: int = 4000) -> str:
    response = requests.post(
        OLLAMA_GENERATE_ENDPOINT,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": ctx
            }
        }
    )

    try:    
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
    
    data = response.json()

    if "response" not in data:
        print("Unexpected response format:", data)
        return ""
    
    return data["response"]



def get_website_html(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")
        return ""


def extract_core_website_content(html: str) -> str:
    
    response = get_ai_response(
        # model="gemma3:4b-it-qat",
        model=EXTRACTION_MODEL,
        prompt=f"""
            You are an expert web content extractor. Your task is to extract the core content from a given HTML page.
            The core content should be the main text, excluding navigation, footers, and other non-essential elements like scripts etc.

            Here is the HTML content:
            <html>
            {html}
            </html>

            Please extract the core content and return it as plain text.
        """,
        ctx=8192
    )

    return response


def summarize_content(content: str) -> str:
    response = get_ai_response(
        # model="gemma3:1b-it-qat",
        model=SUMMARIZATION_MODEL,
        prompt=f"""
            You are an expert summarizer. Your task is to summarize the provided content into a concise and clear summary.

            Here is the content to summarize:
            <content>
            {content}
            </content>

            Please provide a brief summary of the main points in the content. Prefer bullet points and avoid unncessary explanations.
        """
    )

    return response


def generate_x_post(summary: str) -> str:
    with open("post-examples.json", "r") as f:
        examples = json.load(f)

    examples_str = ""
    for i, example in enumerate(examples, 1):
        examples_str += f"""
        <example-{i}>
            <topic>
            {example['topic']}
            </topic>

            <generated-post>
            {example['post']}
            </generated-post>
        </example-{i}>
        """

    prompt = f"""
        You are an expert social media manager, and you excel at crafting viral and highly engaging posts for X (formerly Twitter).

        Your task is to generate a post based on a short text summary.
        Your post must be concise and impactful.
        Avoid using hashtags and lots of emojis (a few emojis are okay, but not too many).

        Keep the post short and focused, structure it in a clean, readable way, using line breaks and empty lines to enhance readability.

        Here's the text summary which you should use to generate the post:
        <summary>
        {summary}
        </summary>

        Here are some examples of topics and generated posts:
        <examples>
            {examples_str}
        </examples>

        Please use the tone, language, structure , and style of the examples provided above to generate a post that is engaging and relevant to the topic provided by the user.
        Don't use the content from the examples!
"""
    response = get_ai_response(
        # model="gemma3:27b-it-qat",
        model=POST_GENERATION_MODEL,
        prompt=prompt
    )

    return response


def main():
    # website_url = input("Website URL: ")
    website_url = SOURCE_URL
    
    print("Fetching website HTML...")
    try:
        html_content = get_website_html(website_url)
    except Exception as e:
        print(f"An error occurred while fetching the website: {e}")
        return

    if not html_content:
        print("Failed to fetch the website content. Exiting.")
        return

    t0 = time.time()

    print("---------")
    print("Extracting core content from the website...")
    core_content = extract_core_website_content(html_content)
    print("Extracted core content:")
    print(core_content)

    print("---------")
    print("Summarizing the core content...")
    summary = summarize_content(core_content)
    print("Generated summary:")
    print(summary)

    print("---------")
    print("Generating X post based on the summary...")
    x_post = generate_x_post(summary)
    print("Generated X post:")
    print(x_post)

    t1 = time.time()
    elapsed_time = t1 - t0
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
