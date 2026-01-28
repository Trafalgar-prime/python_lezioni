def decoratore(f):
    def wrapper():
        print("Prima dell'esecuzione")
        f()
        print("Dopo l'esecuzione")
    return wrapper

@decoratore #applica il decoratore alla funzione, altrimenti non vale il decoratore
def saluta():
    print("ciao")

saluta()

def decoratore(f):
    def wrapper(*args , **kwargs): #in questo modo creo una funzione generica che può accettare ogni parametro, quindi utile per ogni caso
        print(f"Chiamata a {f.__name__} con argomenti : {args} , {kwargs}")
        risultato = f(*args , **kwargs)
        print(f"Risultato = {risultato}")
        return risultato
    return wrapper

@decoratore
def somma(a,b,**kwargs):# quando definisco la funzione in f ho gia dato tutti i possibili argomenti, quindi cosi mi chiamo solo quelli che mi servono realmente
    return a + b

somma(3,5)
somma(6,7,nome = 'Luca',età=17)

def decoratore1(f):
    def wrapper(*args, **kwargs):
        print("Decoratore 1 - Prima")
        risultato = f(*args, **kwargs)
        print("Decoratore 1 - Dopo")
        return risultato
    return wrapper

def decoratore2(f):
    def wrapper(*args, **kwargs):
        print("Decoratore 2 - Prima")
        risultato = f(*args, **kwargs)
        print("Decoratore 2 - Dopo")
        return risultato
    return wrapper

@decoratore1
@decoratore2
def saluta():
    print("Ciao!")

saluta()

import functools

def decoratore(f):
    @functools.wraps(f)  # Mantiene il nome e la docstring della funzione originale
    def wrapper(*args, **kwargs):
        print(f"Eseguendo {f.__name__}")
        return f(*args, **kwargs)
    return wrapper

@decoratore
def esempio():
    """Questa è una funzione di esempio"""
    print("Funzione originale")

print(esempio.__name__)  # ✅ Output corretto: esempio
print(esempio.__doc__)   # ✅ Output corretto: Questa è una funzione di esempio
esempio()

import time

def tempo_di_esecuzione(f):
    def wrapper(*args, **kwargs):
        inizio = time.time()
        print("Inizio tempo")
        risultato = f(*args, **kwargs)
        fine = time.time()
        print("Fine tempo")
        print(f"Tempo di esecuzione: {fine - inizio:.4f} secondi")
        return risultato
    return wrapper

@tempo_di_esecuzione
def operazione_lenta():
    time.sleep(2)  # Simula un'operazione lunga
    print("Operazione completata!")

operazione_lenta()


def richiede_autenticazione(f):
    def wrapper(*args, **kwargs):
        utente_autenticato = True  # Cambia in True per testare
        if not utente_autenticato:
            print("Accesso negato!")
            return
        return f(*args, **kwargs)
    return wrapper

@richiede_autenticazione
def area_riservata():
    print("Benvenuto nell'area riservata!")

area_riservata()

import functools

def cache(f):
    memoria = {}

    @functools.wraps(f)
    def wrapper(*args):
        if args in memoria:
            print("Restituisco valore dalla cache")
            return memoria[args]
        risultato = f(*args)
        memoria[args] = risultato
        return risultato
    return wrapper

@cache
def quadrato(n):
    print(f"Calcolando quadrato di {n}")
    return n * n

print(quadrato(4))  # Calcolo
print(quadrato(2))  # Calcolo
print(quadrato(2))  # Cache


