# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 19:24:12 2018

@author: djerr
"""

import random
def hcf(a, b):
    a=abs(a)
    b=abs(b)
    if a<b:
        a,b=b,a
        
    while b != 0:
        a, b = b, a % b
    return a

'''
Euclid's extended algorithm for finding the
 multiplicative inverse of two numbers
'''
def multiplicative_inverse(e, phi):
   
    x = 0
    y = 1
    lx = 1
    ly = 0
    oa = e  
    ob = phi 
    while phi != 0:
        q = e // phi
        (e, phi) = (phi, e % phi)
        (x, lx) = ((lx - (q * x)), x)
        (y, ly) = ((ly - (q * y)), y)
    if lx < 0:
        lx += ob  
    if ly < 0:
        ly += oa  
    return lx

def is_prime(num):
    if num == 2:
        return True
    if num < 2 or num % 2 == 0:
        return False
    for n in range(3, int(num**0.5)+2, 2):
        if num % n == 0:
            return False
    return True

def generate_keypair(p, q):
    if not (is_prime(p) and is_prime(q)):
        raise ValueError('Both numbers must be prime.')
    elif p == q:
        raise ValueError('p and q cannot be equal')
    
    n = p * q
    phi = (p-1) * (q-1)

    #chose coprime
    e = random.randrange(1, phi)

    g = hcf(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = hcf(e, phi)
    d = multiplicative_inverse(e, phi)
    print(d)
    
    return ((e, n), (d, n))

def encrypt(pk, plaintext):
    key, n = pk
    cipher = [(ord(char) ** key) % n for char in plaintext]
    return cipher

def decrypt(pk, ciphertext):
    key, n = pk
    plain = [chr((char ** key) % n) for char in ciphertext]


    return ''.join(plain)



if __name__ == '__main__':
    
    print ("RSA Encrypter/ Decrypter")
    p = int(input("Enter a prime number (17, 19, 23, etc): "))
    q = int(input("Enter another prime number (Not one you entered above):"))
    print ("Generating your public/private keypairs now . . .")
    public, private = generate_keypair(p, q)
    print ("Your public key is ", public ," and your private key is ", private)
    message = input("Enter a message to encrypt with your private key: ")
    encrypted_msg = encrypt(private, message)
    print ("Your encrypted message is: ")
    print (''.join(map(lambda x: str(x), encrypted_msg)))
    print ("Decrypting message with public key ", public ," . . .")
    print ("Your message is:")
    print (decrypt(public, encrypted_msg))