"""Prueba de concepto de *scraping* impulsado por IA para Falabella Colombia.

Este script muestra un pipeline compacto que realiza las siguientes tareas:

1. Construye la URL de búsqueda para la palabra clave y el número de página.
2. **(Async)** Renderiza el HTML con Playwright y recoge todos los enlaces absolutos.
3. Pide a la API de OpenAI un patrón de expresión regular capaz de distinguir páginas de producto.
4. Filtra las URLs de producto usando dicho patrón.
5. Descarga cada página de producto, elimina contenido innecesario y extrae texto e imágenes.
6. Delega al *function calling* de OpenAI la obtención de un objeto `Product` estructurado.
7. Persiste el resultado en *scrapping_producto.json*.

Se requiere tener definida la variable de entorno **OPENAI_API_KEY**.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import urllib.parse
import asyncio

import dotenv
import requests
from bs4 import BeautifulSoup
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from openai import OpenAI
from playwright.async_api import async_playwright  # <- versión asíncrona
from pydantic import BaseModel, Field

dotenv.load_dotenv(dotenv.find_dotenv())


# ---------------------------------------------------------------------------
# Utilidades de URL
# ---------------------------------------------------------------------------


def build_search_url(term: str, page: int = 1) -> str:
    """Devuelve la URL completa de búsqueda para *term* y página *page*."""
    base = "https://www.falabella.com.co/falabella-co/search"
    query = urllib.parse.urlencode({"Ntt": term, "page": page})
    return f"{base}?{query}"


# ---------------------------------------------------------------------------
# Scraping de la página de resultados (ahora asíncrono)
# ---------------------------------------------------------------------------


async def get_urls_search_scrapping(term: str, page_number: int = 1) -> list[str]:
    """Renderiza los resultados y recopila URLs absolutas de forma asíncrona.

    Utiliza Playwright asincrónico para ejecutar el JS del cliente y obtener el
    DOM final generado por Falabella.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            url = build_search_url(term, page_number)
            print("🌐 Navegando…", url)
            await page.goto(url, timeout=15_000)

            # Falabella puede redirigir eliminando el parámetro *page*; se repone.
            final_url = page.url
            if f"&page={page_number}" not in final_url and page_number > 1:
                parsed = urllib.parse.urlparse(final_url)
                qs = urllib.parse.parse_qs(parsed.query)
                qs["page"] = [str(page_number)]
                final_url = parsed._replace(query=urllib.parse.urlencode(qs, doseq=True)).geturl()
                await page.goto(final_url, timeout=15_000)

            await page.wait_for_timeout(3_000)
            html = await page.content()
            print(f"✅ HTML descargado: {len(html):,} caracteres")

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(("script", "style")):
                tag.decompose()

            return [a["href"] for a in soup.find_all("a", href=True) if a["href"].startswith("http")]

        finally:
            await browser.close()


# ---------------------------------------------------------------------------
# Búsqueda asistida por IA del patrón regex
# ---------------------------------------------------------------------------


class RegexPattern(BaseModel):
    pattern: str = Field(description="Patrón de expresión regular que identifica URLs de producto.")


def get_pattern(urls: list[str]) -> str:
    """Solicita al LLM un patrón regex que seleccione páginas de producto."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = inspect.cleandoc(
        f"""
        Analiza las siguientes URLs y proporciona únicamente el patrón de
        expresión regular que identifique las páginas de producto del dominio
        Falabella.

        URLs:
        {urls}
        """
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=0,
        response_format=RegexPattern,
    )

    return completion.choices[0].message.parsed.pattern


# ---------------------------------------------------------------------------
# Filtrado de URLs
# ---------------------------------------------------------------------------


def get_url_products(urls: list[str], pattern: str) -> list[str]:
    product_pattern = re.compile(pattern)
    return [url for url in urls if product_pattern.search(url)]


# ---------------------------------------------------------------------------
# Descarga y transformación de páginas de producto
# ---------------------------------------------------------------------------


def get_product_scrapping(url_product: str, timeout: int = 10) -> tuple[str, list[str]]:
    """Descarga *url_product* y entrega texto limpio + URLs de imágenes."""
    try:
        resp = requests.get(url_product, timeout=timeout)
        resp.raise_for_status()
    except requests.Timeout:
        print("⏰ Timeout:", url_product)
        return "", []
    except requests.RequestException as exc:
        print("⚠️  Error de red:", exc)
        return "", []

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(("script", "style")):
        tag.decompose()

    image_urls = [img["src"] for img in soup.find_all("img", src=True) if img["src"].startswith("http")]

    html2text = Html2TextTransformer()
    page_text = html2text.transform_documents([Document(page_content=resp.text)])[0].page_content

    return page_text, image_urls


# ---------------------------------------------------------------------------
# Modelos y extracción estructurada
# ---------------------------------------------------------------------------


class Product(BaseModel):
    id: str
    title: str
    price: float
    image_url: str
    description: str


def get_product_info(page_text: str, image_urls: list[str], url_product: str) -> Product:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = inspect.cleandoc(
        f"""
        Extrae id, título, precio, url de imagen y una breve descripción.

        URL del producto: {url_product}
        Texto de la página:\n{page_text}
        URLs de imágenes: {image_urls}
        """
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        response_format=Product,
    )

    return completion.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Funciones de alto nivel
# ---------------------------------------------------------------------------


def get_products(urls_products: list[str], n_products: int) -> list[dict[str, str | float]]:
    resultados = []
    for url in urls_products[:n_products]:
        texto, imagenes = get_product_scrapping(url)
        producto = get_product_info(texto, imagenes, url)
        resultados.append(
            {
                "ID": producto.id,
                "TITLE": producto.title,
                "PRICE": producto.price,
                "IMAGE_URL": producto.image_url,
                "DESCRIPTION": producto.description,
            }
        )
    return resultados


# ---------------------------------------------------------------------------
# Punto de entrada CLI
# ---------------------------------------------------------------------------


async def main() -> None:
    query = "celulares"
    page_number = 1
    n_products = 3

    urls_search = await get_urls_search_scrapping(query, page_number)
    urls_products = []
    while len(urls_products) == 0:
        pattern_search_urls = get_pattern(urls_search)
        urls_products = get_url_products(urls_search, pattern_search_urls)

    print(f"🔎 {len(urls_products)} productos identificados")

    print("\nProcesando productos…")
    productos = get_products(urls_products, n_products)

    with open("scrapping_producto.json", "w", encoding="utf-8") as fp:
        json.dump(productos, fp, ensure_ascii=False, indent=4)

    print("✅ Resultado guardado en scrapping_producto.json")


if __name__ == "__main__":
    asyncio.run(main())
