import threading

def stampa_messaggio():
    print("Ciao dal thread!\n")

#CREIAMO UN THREAD
mio_thread = threading.Thread(target = stampa_messaggio)

#AVVIAMO IL THREAD
mio_thread.start()

print("Questo è il thread principale\n") #l'output è causale tra i due print perche i due processi avvengono contemporaneamente


def stampa_nome(nome):
    print(f"{nome}")

thread1 = threading.Thread(target = stampa_nome, args = ("Alice",)) #in questo modo è sbagliato, perche passa prima questo di tutti gli altri con args=
thread2 = threading.Thread(target = stampa_nome("Bob"))
thread3 = threading.Thread(target = stampa_nome , args = ("Carlotta",)) #la virgola in args è obbligatoria

thread1.start()
thread2.start()
thread3.start()


class MioThread(threading.Thread):
    def __init__(self,nome):
        super().__init__()
        self.nome = nome

    def run(self):
        print(f"\nThread {self.nome} in esecuzione")

#Creiamo e avviamo il thread

mio_thread = MioThread("Alice")
mio_thread.start()

import time

def lavora():
    time.sleep(1) #anche con sleep(0) il programma potrebbe terminare prima che il thread finisca
    print("Lavoro completato")

thread = threading.Thread(target = lavora)
thread.start()

print("\nAspettando che il thread finisca")
thread.join()   #Senza join(), il programma principale potrebbe terminare prima che il thread finisca
print("Il thread è terminato")


#Se più thread modificano la stessa variabile contemporaneamente, possono verificarsi race conditions.
#Possiamo usare Lock per evitare problemi di concorrenza.


saldo = 100
lock = threading.Lock()

def preleva(quantità):
    global saldo
    lock.acquire()
    if saldo >= quantità:
        saldo -= quantità
        print(f"Prelevati {quantità}, saldo rimanente: {saldo}")

    else:
        print("Saldo insufficiente")
    lock.release()

thread1 = threading.Thread(target = preleva , args = (50,))
#thread2 = threading.Thread(target = preleva(80))  # ❌ ERRORE
thread2 = threading.Thread(target = preleva, args = (80,))

thread1.start()
thread2.start()
thread1.join()
thread2.join()

from concurrent.futures import ThreadPoolExecutor

def saluta(nome):
    print(f"saluta {nome}")

with ThreadPoolExecutor(max_workers=3) as executor:
    executor.submit(saluta, "alice")
    executor.submit(saluta, "bob")
    executor.submit(saluta, "charlie")
