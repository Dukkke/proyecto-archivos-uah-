from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import json

def scrape_archivo_uah(max_pages=5, output_file="archivo_uah_consolidado.json"):
    # Configuración de Selenium (modo headless)
    options = Options()
    options.add_argument("--headless")  
    driver = webdriver.Chrome(options=options)

    base_url = "https://archivopatrimonial.uahurtado.cl/index.php/informationobject/browse"
    documentos = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}"
        print(f"📄 Procesando página {page}: {url}")
        driver.get(url)

        # Esperar a que cargue el contenido dinámico
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.result"))
            )
        except:
            print(f"⚠️ No se encontraron resultados en la página {page}")
            continue

        # Parsear HTML renderizado
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Ajusta el selector según el HTML real del sitio
        for item in soup.select("div.result a"):
            titulo = item.get_text(strip=True)
            enlace = item.get("href")
            if titulo and enlace:
                documentos.append({"title": titulo, "url": enlace})

        time.sleep(2)  # pequeña pausa entre páginas

    driver.quit()

    # Guardar resultados en JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"documentos": documentos}, f, ensure_ascii=False, indent=2)

    print(f"✅ Scraping completado: {len(documentos)} documentos extraídos en total")
    print(f"📂 Guardado en: {output_file}")


if __name__ == "__main__":
    scrape_archivo_uah(max_pages=10)  # ajusta el número de páginas según lo que quieras recorrer
