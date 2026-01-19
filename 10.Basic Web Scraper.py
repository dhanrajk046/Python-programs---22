import requests
from bs4 import BeautifulSoup

# URL to scrape
url = "https://www.example.com"

# Send HTTP GET request
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, "html.parser")

# Extract and print the page title
print("Page Title:", soup.title.string)

# Extract and print all the links on the page
print("\nAll links on the page:")
for link in soup.find_all('a'):
    href = link.get('href')
    if href:
        print(href)