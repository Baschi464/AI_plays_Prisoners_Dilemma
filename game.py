import random

def game():

    #numero di turni parametrizzabile
    roundnumber=20
    count=1
    punteggioroundA=0
    punteggioroundB=0

    for i in range(roundnumber):
        
        #estrae scelta di a e b
        # 0=defect
        # 1=cooperate

        risultatoA= random.randint(0,1)
        risultatoB=random.randint(0,1)

        #valuto il risultato del round in base agli output
        if (risultatoA==0) and (risultatoB==0):
            punteggioroundA+=1
            punteggioroundB+=1
            
        if (risultatoA==1) and (risultatoB==1):
            punteggioroundA+=3
            punteggioroundB+=3
            
        if (risultatoA==1) and (risultatoB==0):
            punteggioroundA+=0
            punteggioroundB+=5
            
        if (risultatoA==0) and (risultatoB==1):
            punteggioroundA+=5
            punteggioroundB+=0
        
        print("round numero "+ str(count) + ":\n", risultatoA, risultatoB)
        count+=1
        roundnumber-=1
    
        

        
    print(punteggioroundA, punteggioroundB)    
game()
