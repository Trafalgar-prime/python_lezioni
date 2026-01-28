# **`asyncio` in Python: Cos'è e perché dovresti usarlo?**

import asyncio
import time
import aiohttp  # Libreria per richieste HTTP asincrone

# Funzione sincrona che simula un'operazione lenta
def operazione_lenta(nome):
    print(f"Inizio {nome}...")
    time.sleep(3)
    print(f"Fine {nome}.")

# Esecuzione sincrona
def esecuzione_sincrona():
    t0 = time.time()
    operazione_lenta("Task 1")
    operazione_lenta("Task 2")
    print(f"Tempo totale: {time.time() - t0:.2f} secondi")

# Funzione asincrona che simula un'operazione lenta
async def operazione_lenta_async(nome):
    print(f"Inizio {nome}...")
    await asyncio.sleep(3)
    print(f"Fine {nome}.")

# Esecuzione asincrona con asyncio.gather()
async def esecuzione_asincrona():
    t0 = time.time()
    await asyncio.gather(
        operazione_lenta_async("Task 1"),
        operazione_lenta_async("Task 2")
    )
    print(f"Tempo totale: {time.time() - t0:.2f} secondi")

# Funzione per testare asyncio.create_task()
async def test_create_task():
    task = asyncio.create_task(operazione_lenta_async("Operazione in background"))
    print("Nel frattempo, il codice continua...")
    await asyncio.sleep(1)
    print("Ancora in esecuzione...")
    await task

# Funzione per scaricare più pagine web contemporaneamente
async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def scarica_pagine():
    urls = ["https://www.python.org", "https://www.wikipedia.org", "https://www.github.com"]
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print("Pagine scaricate:", len(results))

# Esecuzione dei test
if __name__ == "__main__":
    print("--- Esecuzione sincrona ---")
    esecuzione_sincrona()
    
    print("\n--- Esecuzione asincrona ---")
    asyncio.run(esecuzione_asincrona())
    
    print("\n--- Test asyncio.create_task() ---")
    asyncio.run(test_create_task())
    
    print("\n--- Scaricamento pagine web ---")
    asyncio.run(scarica_pagine())
