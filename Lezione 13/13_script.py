import multiprocessing
import time

def stampa_messaggio():
    time.sleep(1)
    print("\nCiao dal processo!", flush = True)

if __name__ == "__main__":

    #creiamo un nuovo processo
    mio_processo = multiprocessing.Process(target = stampa_messaggio)

    #avviamo il processo
    mio_processo.start()


    #il processo principale continua l'esecuzione
    print("\nQuesto è il processo principale")

    # Aspettiamo che il processo termini
    mio_processo.join() #non leggerà nulla perchè il multiprocessing su idle su windows da problemi, se lo runno dal terminale cmd ottengo il risultato giusto

import multiprocessing
if __name__ == "__main__":
    def stampa_nome(nome):
        print(f"Processo eseguito da: {nome}")

    processo1 = multiprocessing.Process(target=stampa_nome, args=("Alice",))
    processo2 = multiprocessing.Process(target=stampa_nome, args=("Bob",))

    processo1.start()
    processo2.start()

    processo1.join()
    processo2.join()


